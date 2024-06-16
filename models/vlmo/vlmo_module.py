import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.vlmo import multiway_transformer, heads, objectives, vlmo_utils

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
# from pytorch_lightning.utilities.distributed import rank_zero_info
from scipy import interpolate
from timm.models import create_model


def convert_to_textpt_ckpt(state_dict, module):
    new_state_dict = {}

    # Merge relative_position_bias_table from all layer into one tensor, 
    # so we can use one op for gather the relative position bias for speed up
    relative_position_bias_tables = {}

    for key in state_dict:
        value = state_dict[key]

        if "relative_position_bias_table" in key:
            # transformer.blocks.0.attn.relative_position_bias_table
            layer_idx = int(key.split(".attn.")[0].split('.')[-1])
            relative_position_bias_tables[layer_idx] = value
            continue

        if "mlp" in key:
            key_imag = "transformer." + key.replace("mlp", "mlp_imag")
            new_state_dict[key_imag] = value
        elif "norm2" in key:
            key_imag = "transformer." + key.replace("norm2", "norm2_imag")
            new_state_dict[key_imag] = value
        else:
            new_key = "transformer." + key
            new_state_dict[new_key] = value
    
    if len(relative_position_bias_tables) > 0:
        tensor_list = []
        for layer_idx in sorted(relative_position_bias_tables.keys()):
            tensor_list.append(relative_position_bias_tables[layer_idx])
        relative_position_bias_table = torch.cat(tensor_list, dim=1)

        num_distence, _ = relative_position_bias_table.shape
        all_relative_position_bias_table = module.relative_position_bias_table.data.clone()
        all_relative_position_bias_table[:num_distence, :] = relative_position_bias_table

        new_state_dict["relative_position_bias_table"] = all_relative_position_bias_table
        
    return new_state_dict


def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        return pos_embed_checkpoint


def convert_deepspeed_ckpt(state_dict):
    new_state_dict = {}
    for key in state_dict:
        if key.startswith("module."):
            new_key = key[len("module."):]
            value = state_dict[key]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = state_dict[key]

    return new_state_dict


class VLMo(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.save_hyperparameters()
        self.config = config
        # momentum value
        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.distill = config['distill']
        self.alpha = config['alpha']

        # backbone & patch projection
        self.img_size = config["image_size"]
        self.transformer = create_model(
            config["model_arch"],
            img_size=self.img_size,
            pretrained=False,
            drop_rate=0,
            drop_path_rate=config["drop_path_rate"],
            attn_drop_rate=0,
            drop_block_rate=None,
            config=self.config,
        )
        self.patch_size = self.transformer.patch_size
        self.vlffn_start_layer_index = self.transformer.vlffn_start_layer_index
        self.num_layers = len(self.transformer.blocks)
        self.num_features = self.transformer.num_features
        self.build_relative_position_embed(config)
        
        # language embedding
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=self.num_features,
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_path_rate"],
            position_embedding_type="rel_pos" if self.transformer.need_relative_position_embed else "absolute", 
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, self.num_features)
        self.token_type_embeddings.apply(objectives.init_weights)

        # task layers        
        self.pooler = heads.Pooler(self.num_features)
        self.pooler.apply(objectives.init_weights)



        ## language modeling
        if config["loss_names"]["mlm"] > 0 or config["loss_names"]["textmlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        ## image-text matching (global hard negative)
        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(self.num_features)
            self.itm_score.apply(objectives.init_weights)
        
        ## contrastive loss (or sampling for global hard negative)
        if config["loss_names"]["itc"] > 0:
            self.itc_text_proj = heads.ITCHead(self.num_features)
            self.itc_image_proj = heads.ITCHead(self.num_features)
            self.itc_text_proj.apply(objectives.init_weights)
            self.itc_image_proj.apply(objectives.init_weights)

            self.itc_vl_text_proj = heads.ITCHead(self.num_features)
            self.itc_vl_image_proj = heads.ITCHead(self.num_features)
            self.itc_vl_text_proj.apply(objectives.init_weights)
            self.itc_vl_image_proj.apply(objectives.init_weights)

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.logit_vl_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))



        ## retrieval task ft
        if config["loss_names"]["irtr"] > 0:
            self.itc_text_proj = heads.ITCHead(self.num_features)
            self.itc_image_proj = heads.ITCHead(self.num_features)
            self.itc_text_proj.apply(objectives.init_weights)
            self.itc_image_proj.apply(objectives.init_weights)

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.load_pretrained_weight()

        if self.distill:
            # create momentum models
            self.transformer_m = create_model(
                config["model_arch"],
                img_size=self.img_size,
                pretrained=False,
                drop_rate=0,
                drop_path_rate=config["drop_path_rate"],
                attn_drop_rate=0,
                drop_block_rate=None,
                config=self.config,
            )
            self.text_embeddings_m = BertEmbeddings(bert_config)
            self.token_type_embeddings_m = nn.Embedding(2, self.num_features)
            self.pooler_m = heads.Pooler(self.num_features)
            if config["loss_names"]["mlm"] > 0 or config["loss_names"]["textmlm"] > 0:
                self.mlm_score_m = heads.MLMHead(bert_config)
            if config["loss_names"]["itm"] > 0:
                self.itm_score_m = heads.ITMHead(self.num_features)
            if config["loss_names"]["itc"] > 0:
                self.itc_text_proj_m = heads.ITCHead(self.num_features)
                self.itc_image_proj_m = heads.ITCHead(self.num_features)
                self.itc_vl_text_proj_m = heads.ITCHead(self.num_features)
                self.itc_vl_image_proj_m = heads.ITCHead(self.num_features)
            if config["loss_names"]["irtr"] > 0:
                self.itc_text_proj_m = heads.ITCHead(self.num_features)
                self.itc_image_proj_m = heads.ITCHead(self.num_features)



            self.model_pairs = [[self.transformer, self.transformer_m],
                                [self.text_embeddings, self.text_embeddings_m],
                                [self.token_type_embeddings, self.token_type_embeddings_m],
                                [self.pooler, self.pooler_m],
                                [self.mlm_score, self.mlm_score_m],
                                [self.itm_score, self.itm_score_m],
                                [self.itc_text_proj, self.itc_text_proj_m],
                                [self.itc_image_proj, self.itc_image_proj_m],
                                [self.itc_vl_text_proj, self.itc_vl_text_proj_m],
                                [self.itc_vl_image_proj, self.itc_vl_image_proj_m],
                                ]

            self.copy_params()
            # self.num_features
            self.register_buffer("image_queue", torch.randn(self.num_features, self.queue_size))
            self.register_buffer("text_queue", torch.randn(self.num_features, self.queue_size))
            self.register_buffer("image_vlffn_queue", torch.randn(self.num_features, self.queue_size))
            self.register_buffer("text_vlffn_queue", torch.randn(self.num_features, self.queue_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
            self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        # ===================== Downstream ===================== #
        ## VQAv2
        if self.config["loss_names"]["vqa"] > 0:
            vs = self.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(self.num_features, self.num_features * 2),
                nn.LayerNorm(self.num_features * 2),
                nn.GELU(),
                nn.Linear(self.num_features * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)


        ## NLVR2 (Visual reasoning)
        if self.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(self.num_features * 2, self.num_features * 2),
                nn.LayerNorm(self.num_features * 2),
                nn.GELU(),
                nn.Linear(self.num_features * 2, 2),
            )
            self.nlvr2_classifier.apply(objectives.init_weights)
            emb_data = self.token_type_embeddings.weight.data
            self.token_type_embeddings = nn.Embedding(3, self.num_features)
            self.token_type_embeddings.apply(objectives.init_weights)
            self.token_type_embeddings.weight.data[0, :] = emb_data[0, :]
            self.token_type_embeddings.weight.data[1, :] = emb_data[1, :]
            self.token_type_embeddings.weight.data[2, :] = emb_data[1, :]

        vlmo_utils.set_metrics(self)
        self.current_tasks = self.config["loss_names"]

        # ===================== load downstream (test_only) ======================

        if self.config["checkpoint"] != "" and self.config["test_only"]:
            print("Load ckpt from: {}".format(self.config["checkpoint"]))
            ckpt = torch.load(self.config["checkpoint"], map_location="cpu")

            state_dict = None
            
            for state_dict_key in ("state_dict", "module", "model"):
                if state_dict_key in ckpt:
                    print("Read state dict from ckpt[%s]. " % state_dict_key)
                    state_dict = ckpt[state_dict_key]
                    break
            if state_dict_key == "module":
                state_dict = convert_deepspeed_ckpt(state_dict)
            if state_dict is None:
                print("Read state dict from ckpt. ")
                state_dict = ckpt

            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print("missing_keys: {}".format(missing_keys))
            print("unexpected_keys: {}".format(unexpected_keys))

    def load_pretrained_weight(self):
        if self.config["checkpoint"] != "" and not self.config["test_only"]:
            config = self.config
            ckpt = torch.load(self.config["checkpoint"], map_location="cpu")
            print("Load ckpt from: {}".format(self.config["checkpoint"]))

            state_dict = None

            for state_dict_key in ("state_dict", "module", "model"):
                if state_dict_key in ckpt:
                    print("Read state dict from ckpt[%s]. " % state_dict_key)
                    state_dict = ckpt[state_dict_key]
                    break
            if state_dict_key == "module":
                state_dict = convert_deepspeed_ckpt(state_dict)
            if state_dict is None:
                print("Read state dict from ckpt. ")
                state_dict = ckpt

            for key in state_dict:
                var = state_dict[key]
                print("%s = %s" % (key, str(var.size())))

            print(config["loss_names"])
            if config["loss_names"]["textmlm"] > 0:
                print("convert to textpt")
                state_dict = convert_to_textpt_ckpt(state_dict, self)

            max_text_len = config["max_text_len"]
            if "text_embeddings.position_embeddings.weight" in state_dict and state_dict["text_embeddings.position_embeddings.weight"].size(0) != max_text_len:
                state_dict["text_embeddings.position_embeddings.weight"].data = state_dict["text_embeddings.position_embeddings.weight"].data[:max_text_len, :]
                state_dict["text_embeddings.position_ids"].data = state_dict["text_embeddings.position_ids"].data[:, :max_text_len]
                print("text position_embeddings size: {}".format(state_dict["text_embeddings.position_embeddings.weight"].size()))
                for check_key in ("relative_position_index", "text_relative_position_index", "text_imag_relative_position_index"):
                    if check_key in state_dict:
                        state_dict.pop(check_key)

            if "transformer.pos_embed" in state_dict:
                pos_embed_reshaped = interpolate_pos_embed(state_dict['transformer.pos_embed'], self.transformer)         
                state_dict['transformer.pos_embed'] = pos_embed_reshaped

            if "relative_position_bias_table" in state_dict:
                rel_pos_bias = state_dict["relative_position_bias_table"]
                src_num_pos, num_attn_heads = rel_pos_bias.size()
                dst_num_pos, _ = self.relative_position_bias_table.size()
                dst_patch_shape = self.transformer.patch_embed.patch_shape
                if dst_patch_shape[0] != dst_patch_shape[1]:
                    raise NotImplementedError()
                num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
                src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
                dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
                if src_size != dst_size:
                    #todo: fix this
                    state_dict.pop("relative_position_index")
                    state_dict.pop("text_relative_position_index")
                    state_dict.pop("text_imag_relative_position_index")
                    
                    print("Position interpolate from %dx%d to %dx%d" % (
                        src_size, src_size, dst_size, dst_size))
                    extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                    rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    print("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(num_attn_heads):
                        z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                        f = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(
                            torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                    rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                    new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                    state_dict["relative_position_bias_table"] = new_rel_pos_bias

            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print("missing_keys: {}".format(missing_keys))
            print("unexpected_keys: {}".format(unexpected_keys))

    def get_rel_pos_bias(self, relative_position_index):
        if self.relative_position_embed:
            relative_position_bias = F.embedding(relative_position_index.long().to(self.relative_position_bias_table.device),
                                                    self.relative_position_bias_table)
            all_relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # nH, x, y
            relative_position_bias_list = torch.chunk(all_relative_position_bias, self.num_layers, dim=0)
            return relative_position_bias_list
        else:
            return [None] * self.num_layers

    def build_relative_position_embed(self, config):
        if not self.transformer.need_relative_position_embed:
            self.relative_position_embed = False
            self.text_imag_relative_position_index = None
            self.text_relative_position_index = None
            self.relative_position_index = None
            return

        self.relative_position_embed = True
        window_size = (int(self.img_size / self.patch_size), int(self.img_size / self.patch_size)) #(14, 14)
        print("window_size: {}".format(window_size))
        num_heads = self.transformer.num_heads
        max_text_len_of_initckpt = config["max_text_len_of_initckpt"] #196
        max_text_len = config["max_text_len"] #40
        max_imag_len = window_size[0] * window_size[1] + 1 #197
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.text_num_relative_distance = 2 * max_text_len_of_initckpt
        self.all_num_relative_distance = self.num_relative_distance + self.text_num_relative_distance + 2

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.all_num_relative_distance, num_heads * self.num_layers))
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.relative_position_index = relative_position_index
        
        text_position_ids = torch.arange(max_text_len-1)
        text_rel_pos_mat = text_position_ids.unsqueeze(-2) - text_position_ids.unsqueeze(-1)
        min_distance = int(2-max_text_len_of_initckpt) #-194
        # print("min_distance: {}".format(min_distance))
        text_rel_pos_mat = text_rel_pos_mat - min_distance
        text_rel_pos_mat += (self.num_relative_distance + 2)
        text_relative_position_index = \
            torch.zeros(size=(max_text_len, ) * 2, dtype=relative_coords.dtype)
        text_relative_position_index[1:, 1:] = text_rel_pos_mat
        text_relative_position_index[0, 0:] = self.all_num_relative_distance - 3
        text_relative_position_index[0:, 0] = self.all_num_relative_distance - 2
        text_relative_position_index[0, 0] = self.all_num_relative_distance - 1
        self.text_relative_position_index = text_relative_position_index
        
        text2imag_relative_position_index = torch.ones(max_text_len, max_imag_len) * (self.num_relative_distance)
        imag2text_relative_position_index = torch.ones(max_imag_len, max_text_len) * (self.num_relative_distance + 1)

        text_row_relative_position_index = torch.cat((text_relative_position_index, text2imag_relative_position_index), 1)
        imag_row_relative_position_index = torch.cat((imag2text_relative_position_index, relative_position_index), 1)
        text_imag_relative_position_index = torch.cat((text_row_relative_position_index, imag_row_relative_position_index), 0)
        self.text_imag_relative_position_index = text_imag_relative_position_index

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"].to(self.config['device'])
        text_labels = batch[f"text_labels{do_mlm}"].to(self.config['device'])
        text_masks = batch[f"text_masks"].to(self.config['device'])
        img = batch[imgkey][0]
        img = img.to(self.config['device'], non_blocking=True)


        text_embeds = self.text_embeddings(text_ids)


        image_embeds, image_masks = self.transformer.visual_embed(img)

        image_masks = image_masks.long().to(device=img.get_device())
        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds
        relative_position_bias_list = self.get_rel_pos_bias(self.text_imag_relative_position_index)

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, modality_type="vl", relative_position_bias=relative_position_bias_list[i])

        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)

        # get momentum features
        if self.distill:
            with torch.no_grad():
                # self.temp.clamp_(0.001, 0.5)
                # self._momentum_update()
                text_embeds_m = self.text_embeddings_m(text_ids)
                image_embeds_m, image_masks_m = self.transformer_m.visual_embed(img)
                image_masks_m = image_masks_m.long().to(device=img.get_device())
                text_embeds_m, image_embeds_m = (
                    text_embeds_m + self.token_type_embeddings_m(torch.zeros_like(text_masks)),
                    image_embeds_m
                    + self.token_type_embeddings_m(
                        torch.full_like(image_masks_m, image_token_type_idx)
                    ),
                )

                co_embeds_m = torch.cat([text_embeds_m, image_embeds_m], dim=1)
                co_masks_m = torch.cat([text_masks, image_masks_m], dim=1)
                x_m = co_embeds_m

                for i, blk in enumerate(self.transformer_m.blocks):
                    x_m= blk(x_m, mask=co_masks, modality_type="vl", relative_position_bias=relative_position_bias_list[i])

                x_m = self.transformer_m.norm(x_m)
                text_feats_m, image_feats_m = (
                    x_m[:, : text_embeds_m.shape[1]],
                    x_m[:, text_embeds_m.shape[1] :],
                )
                cls_feats_m = self.pooler_m(x_m)

                # cls_feats = self.temp * cls_feats + (1 - self.temp) * cls_feats_m
                # text_feats = self.temp * text_feats + (1 - self.temp) * text_feats_m
                # image_feats = self.temp * image_feats + (1 - self.temp) * image_feats_m
                ret = {
                    "text_feats": text_feats,
                    "image_feats": image_feats,
                    "cls_feats": cls_feats,
                    "text_feats_m": text_feats_m,
                    "image_feats_m": image_feats_m,
                    "cls_feats_m": cls_feats_m,
                    "raw_cls_feats": x[:, 0],
                    "raw_cls_feats_m": x_m[:, 0],
                    "image": img,
                    "text_labels": text_labels,
                    "text_ids": text_ids,
                    "text_masks": text_masks,
                }
                return ret


        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image": img,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret

    def infer_text(
        self,
        batch,
        mask_text=False,
    ):
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        co_embeds = text_embeds
        co_masks = text_masks

        x = co_embeds
        all_hidden_states = []
        relative_position_bias_list = self.get_rel_pos_bias(self.text_relative_position_index)

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, modality_type="text", relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)
        
        vlffn_hiddens = all_hidden_states[self.vlffn_start_layer_index-1]
        for vlffn_index in range(self.vlffn_start_layer_index, self.num_layers):
            vlffn_hiddens = self.transformer.blocks[vlffn_index](vlffn_hiddens, mask=co_masks, modality_type="vl", relative_position_bias=relative_position_bias_list[vlffn_index])

        lffn_hiddens = all_hidden_states[-1]

        lffn_hiddens = self.transformer.norm(lffn_hiddens)
        text_feats, image_feats = (
            lffn_hiddens,
            None,
        )

        cls_feats = self.itc_text_proj(lffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        vlffn_hiddens = self.transformer.norm(vlffn_hiddens)
        cls_vlffn_feats = self.itc_vl_text_proj(vlffn_hiddens[:, 0])
        cls_vlffn_feats = cls_vlffn_feats / cls_vlffn_feats.norm(dim=-1, keepdim=True)


        if self.distill:
            with torch.no_grad():
                text_embeds_m = self.text_embeddings_m(text_ids)

                co_embeds_m = text_embeds_m
                co_masks_m = text_masks
                x_m = co_embeds_m
                all_hidden_states_m = []
                for i, blk in enumerate(self.transformer_m.blocks):
                    x_m = blk(x_m, mask=co_masks_m, modality_type="text", relative_position_bias=relative_position_bias_list[i])
                    all_hidden_states_m.append(x_m)

                vlffn_hiddens_m = all_hidden_states_m[self.vlffn_start_layer_index-1]
                for vlffn_index in range(self.vlffn_start_layer_index, self.num_layers):
                    vlffn_hiddens_m = self.transformer_m.blocks[vlffn_index](vlffn_hiddens_m, mask=co_masks_m, modality_type="vl", relative_position_bias=relative_position_bias_list[vlffn_index])

                lffn_hiddens_m = all_hidden_states_m[-1]
                lffn_hiddens_m = self.transformer_m.norm(lffn_hiddens_m)

                text_feats_m, image_feats_m = (
                    lffn_hiddens_m,
                    None,
                )

                cls_feats_m = self.itc_text_proj_m(lffn_hiddens_m[:, 0])
                cls_feats_m = cls_feats_m / cls_feats_m.norm(dim=-1, keepdim=True)

                vlffn_hiddens_m = self.transformer_m.norm(vlffn_hiddens_m)
                cls_vlffn_feats_m = self.itc_vl_text_proj_m(vlffn_hiddens_m[:, 0])
                cls_vlffn_feats_m = cls_vlffn_feats_m / cls_vlffn_feats_m.norm(dim=-1, keepdim=True)

                ret = {
                    "text_feats": text_feats,
                    "image_feats": image_feats,
                    "cls_feats": cls_feats,
                    "cls_vlffn_feats": cls_vlffn_feats,
                    "raw_cls_feats": x[:, 0],
                    "image_masks": None,
                    "text_labels": text_labels,
                    "text_ids": text_ids,
                    "text_masks": text_masks,
                    "text_feats_m": text_feats_m,
                    "image_feats_m": image_feats_m,
                    "cls_feats_m": cls_feats_m,
                    "cls_vlffn_feats_m": cls_vlffn_feats_m,
                }
                return ret



        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": cls_vlffn_feats,
            "raw_cls_feats": x[:, 0],
            "image_masks": None,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret

    def infer_text_ft(
        self,
        batch,
        mask_text=False,
    ):
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"].to(self.config['device'])
        text_labels = batch[f"text_labels{do_mlm}"].to(self.config['device'])
        text_masks = batch[f"text_masks"].to(self.config['device'])
        text_embeds = self.text_embeddings(text_ids)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        co_embeds = text_embeds
        co_masks = text_masks

        x = co_embeds
        all_hidden_states = []
        relative_position_bias_list = self.get_rel_pos_bias(self.text_relative_position_index)

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, modality_type="text", relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)

        lffn_hiddens = all_hidden_states[-1]

        lffn_hiddens = self.transformer.norm(lffn_hiddens)
        text_feats, image_feats = (
            lffn_hiddens,
            None,
        )

        cls_feats = self.itc_text_proj(lffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": None,
            "raw_cls_feats": x[:, 0],
            "image_masks": None,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret

    def infer_text_mlm(
        self,
        batch,
        mask_text=False,
    ):
        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        co_embeds = text_embeds
        co_masks = text_masks

        x = co_embeds
        all_hidden_states = []
        relative_position_bias_list = self.get_rel_pos_bias(self.text_relative_position_index)

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, modality_type="text", relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)

        lffn_hiddens = all_hidden_states[-1]

        lffn_hiddens = self.transformer.norm(lffn_hiddens)
        text_feats, image_feats = (
            lffn_hiddens,
            None,
        )

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": None,
            "cls_vlffn_feats": None,
            "raw_cls_feats": x[:, 0],
            "image_masks": None,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret

    def infer_image(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        img = batch[imgkey][0]
        image_embeds, image_masks = self.transformer.visual_embed(img)

        image_masks = image_masks.long().to(device=img.get_device())
        image_embeds = image_embeds + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            )

        co_embeds = image_embeds
        co_masks = image_masks

        x = co_embeds
        all_hidden_states = []
        relative_position_bias_list = self.get_rel_pos_bias(self.relative_position_index)

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, modality_type="image", relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)
        
        vlffn_hiddens = all_hidden_states[self.vlffn_start_layer_index-1]

        for vlffn_index in range(self.vlffn_start_layer_index, self.num_layers):
            vlffn_hiddens = self.transformer.blocks[vlffn_index](vlffn_hiddens, mask=co_masks, modality_type="vl", relative_position_bias=relative_position_bias_list[vlffn_index])
        
        vffn_hiddens = all_hidden_states[-1]
        vffn_hiddens = self.transformer.norm(vffn_hiddens)
        text_feats, image_feats = (
            None,
            vffn_hiddens,
        )

        cls_feats = self.itc_image_proj(vffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)


        cls_vlffn_feats = self.itc_vl_image_proj(vlffn_hiddens[:, 0])
        cls_vlffn_feats = cls_vlffn_feats / cls_vlffn_feats.norm(dim=-1, keepdim=True)

        if self.distill:
            with torch.no_grad():
                image_embeds_m, image_masks_m = self.transformer_m.visual_embed(img)
                image_masks_m = image_masks_m.long().to(device=img.get_device())
                image_embeds_m = image_embeds_m + self.token_type_embeddings(
                    torch.full_like(image_masks_m, image_token_type_idx)
                )
                co_embeds_m = image_embeds_m
                co_masks_m = image_masks_m
                x_m = co_embeds_m
                all_hidden_states_m = []
                for i, blk in enumerate(self.transformer_m.blocks):
                    x_m = blk(x_m, mask=co_masks_m, modality_type="image", relative_position_bias=relative_position_bias_list[i])
                    all_hidden_states_m.append(x_m)

                vlffn_hiddens_m = all_hidden_states_m[self.vlffn_start_layer_index-1]
                for vlffn_index in range(self.vlffn_start_layer_index, self.num_layers):
                    vlffn_hiddens_m = self.transformer_m.blocks[vlffn_index](vlffn_hiddens_m, mask=co_masks_m, modality_type="vl", relative_position_bias=relative_position_bias_list[vlffn_index])

                vffn_hiddens_m = all_hidden_states_m[-1]
                vffn_hiddens_m = self.transformer_m.norm(vffn_hiddens_m)
                text_feats_m, image_feats_m = (
                    None,
                    vffn_hiddens_m,
                )
                cls_feats_m = self.itc_image_proj_m(vffn_hiddens_m[:, 0])
                cls_feats_m = cls_feats_m / cls_feats_m.norm(dim=-1, keepdim=True)

                cls_vlffn_feats_m = self.itc_vl_image_proj_m(vlffn_hiddens_m[:, 0])
                cls_vlffn_feats_m = cls_vlffn_feats_m / cls_vlffn_feats_m.norm(dim=-1, keepdim=True)

                ret = {
                    "text_feats": text_feats,
                    "image_feats": image_feats,
                    "cls_feats": cls_feats,
                    "cls_vlffn_feats": cls_vlffn_feats,
                    "raw_cls_feats": x[:, 0],
                    "image_masks": image_masks,
                    "text_labels": None,
                    "text_ids": None,
                    "text_masks": None,
                    "text_feats_m": text_feats_m,
                    "image_feats_m": image_feats_m,
                    "cls_feats_m": cls_feats_m,
                    "cls_vlffn_feats_m": cls_vlffn_feats_m,
                }
                return ret

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": cls_vlffn_feats,
            "raw_cls_feats": x[:, 0],
            "image_masks": image_masks,
            "text_labels": None,
            "text_ids": None,
            "text_masks": None,
        }

        return ret

    def infer_image_ft(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        img = batch[imgkey][0]
        img = img.to(self.config['device'], non_blocking=True)
        image_embeds, image_masks = self.transformer.visual_embed(img)

        image_masks = image_masks.long().to(device=img.get_device())
        image_embeds = image_embeds + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            )

        co_embeds = image_embeds
        co_masks = image_masks

        x = co_embeds
        all_hidden_states = []
        relative_position_bias_list = self.get_rel_pos_bias(self.relative_position_index)

        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, modality_type="image", relative_position_bias=relative_position_bias_list[i])
            all_hidden_states.append(x)

        vffn_hiddens = all_hidden_states[-1]

        vffn_hiddens = self.transformer.norm(vffn_hiddens)
        text_feats, image_feats = (
            None,
            vffn_hiddens,
        )

        cls_feats = self.itc_image_proj(vffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "cls_vlffn_feats": None,
            "raw_cls_feats": x[:, 0],
            "image_masks": image_masks,
            "text_labels": None,
            "text_ids": None,
            "text_masks": None,
        }

        return ret

    def forward(self, batch):
        ret = dict()

        if self.distill:
            with torch.no_grad():
                self.temp.clamp_(0.001, 0.5)
            self._momentum_update()

        batch["text_ids"] = batch["text_ids"].to(self.config['device'])
        batch["text_masks"] = batch["text_masks"].to(self.config['device'])
        batch["text_labels"] = batch["text_labels"].to(self.config['device'])
        batch["image"][0] = batch["image"][0].to(self.config['device'])
        batch["text_labels_mlm"] = batch["text_labels_mlm"].to(self.config['device'])
        batch["text_ids_mlm"] = batch["text_ids_mlm"].to(self.config['device'])


        if self.current_tasks['embed'] > 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if self.current_tasks['mlm'] > 0:
            ret.update(objectives.compute_mlm(self, batch))

        # Textonly Masked Language Modeling phase two text pretraining
        if self.current_tasks['textmlm'] > 0:
            ret.update(objectives.compute_textonly_mlm(self, batch))

        # Contrastive loss for pretraining
        if self.current_tasks['itc'] > 0 :
            ret.update(objectives.compute_itc(self, batch))

        # Contrastive loss for finetuning
        if self.current_tasks['irtr'] > 0:
            if self.config['cluster'] is True:
                ret.update(objectives.compute_irtr(self, batch, aggregate=True))
            else:
                ret.update(objectives.compute_irtr(self, batch))

        # Image Text Matching with global hard negative, must use with itc
        if self.current_tasks['itm'] > 0:
            ret.update(objectives.compute_itm_hardneg(self, batch, ret["itc_i2t_logits"], ret["itc_t2i_logits"]))

        # Visual Question Answering
        if self.current_tasks['vqa'] > 0:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if self.current_tasks['nlvr2'] > 0:
            ret.update(objectives.compute_nlvr2(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        vlmo_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vlmo_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vlmo_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vlmo_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        vlmo_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.config["checkpoint"].split("/")[-1][:-5]

        if self.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name, self.config["log_dir"])
        vlmo_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vlmo_utils.set_schedule(self)

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, image_vl_feat, text_vl_feat):
        # gather keys before updating queue
        if self.config['cluster']:
            image_feats = concat_all_gather(image_feat)
            text_feats = concat_all_gather(text_feat)
            image_vl_feats = concat_all_gather(image_vl_feat)
            text_vl_feats = concat_all_gather(text_vl_feat)
        else:
            image_feats = image_feat
            text_feats = text_feat
            image_vl_feats = image_vl_feat
            text_vl_feats = text_vl_feat

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        # print(f"self.queue_size{self.queue_size}, batch_size{batch_size}")
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.image_vlffn_queue[:, ptr:ptr + batch_size] = image_vl_feats.T
        self.text_vlffn_queue[:, ptr:ptr + batch_size] = text_vl_feats.T

        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output