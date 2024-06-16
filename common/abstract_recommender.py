# coding: utf-8
# @email  : enoche.chow@gmail.com

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models import clip
from utils.dataset import create_dataset, create_loader
from models.vlmo.vlmo_module import VLMo

from timm.models import create_model
from models.beit3 import modeling_finetune
# from models.beit3.beit3_datasets import create_downstream_dataset
from models.beit3 import utils


class AbstractRecommender(nn.Module):
    r"""Base class for all models
    """

    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        r"""full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
        """
        raise NotImplementedError

    #
    # def __str__(self):
    #     """
    #     Model prints with number of trainable parameters
    #     """
    #     model_parameters = filter(lambda p: p.requires_grad, self.parameters())
    #     params = sum([np.prod(p.size()) for p in model_parameters])
    #     return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class GeneralRecommender(AbstractRecommender):
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """

    def __init__(self, config, dataloader):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']

        # load encoded features here
        self.v_feat, self.t_feat = None, None
        if not config['end2end'] and config['is_multimodal_model']:
            dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
            # if file exist?
            v_feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
            t_feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
            if os.path.isfile(v_feat_file_path):
                self.v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)
            if os.path.isfile(t_feat_file_path):
                self.t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                    self.device)

            assert self.v_feat is not None or self.t_feat is not None, 'Features all NONE'


class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


from models.beit3.beit3_datasets import create_recsys_dataset


class MultiModalEndtoEndRecommender(AbstractRecommender):
    def __init__(self, config, dataloader):
        self.config_class = Dict2Class(config.final_config_dict)
        super(MultiModalEndtoEndRecommender, self).__init__()

        self.config = config
        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = 'cuda'
        self.config = config
        # build encoder

        self.train_data_loader = self.build_dataloader(config)
        self.multi_modal_encoder = self.build_multiModalEncoder(config)

        # self.device = self.config_class.device
        self.multi_modal_encoder.to(self.device)

        self.build_test_dataloader()

        # build embeddings

        if 'beit3' in self.config['multi_modal_encoder'] and self.config['using_cls_feature']:
            self.cls_trian_feat \
                = self.build_embeds(config,
                                    self.multi_modal_encoder,
                                    self.train_data_loader)

        else:
            self.v_train_feat, self.t_train_feat \
                = self.build_embeds(config,
                                    self.multi_modal_encoder,
                                    self.train_data_loader)


        self.config = config

    def detach_train_feat(self):
        if 'beit3' in self.config['multi_modal_encoder'] and self.config['using_cls_feature']:
            self.cls_train_feat.detach().cpu()
        else:
            self.v_train_feat.detach().cpu()
            self.t_train_feat.detach().cpu()

    def detach_test_feat(self):
        if 'beit3' in self.config['multi_modal_encoder'] and self.config['using_cls_feature']:
            self.cls_test_feat.detach().cpu()
        else:
            self.v_test_feat.detach().cpu()
            self.t_test_feat.detach().cpu()

    def build_test_dataloader(self):
        if 'beit3' in self.config['multi_modal_encoder'] :
            test_data_loader = create_recsys_dataset(self.config_class, is_train=False)
            self.test_dataloader = test_data_loader


    def build_test_embeds(self):
        if 'beit3' in self.config['multi_modal_encoder'] and self.config['using_cls_feature']:
            self.cls_feat \
                = self.build_embeds(self.config,
                                    self.multi_modal_encoder,
                                    self.test_dataloader)
        else:

            self.v_test_feat, self.t_test_feat \
                = self.build_embeds(self.config,
                                    self.multi_modal_encoder,
                                    self.test_dataloader)


    def build_dataloader(self, config):
        if 'beit3' in self.config['multi_modal_encoder']:
            train_data_loader = create_recsys_dataset(self.config_class)
            self.item_num = train_data_loader.dataset.item_num

            # test_data_loader = create_recsys_dataset(self.config_class, is_train=False)
            return train_data_loader

        # creat dataset
        datasets = [create_dataset('movie_dataset', config)]
        samplers = [None]
        # create dataloader
        data_loader = \
            create_loader(datasets, samplers,
                          batch_size=[16],
                          num_workers=[2], is_trains=[False],
                          collate_fns=[None])[0]
        return data_loader

    def build_multiModalEncoder(self, config):
        # build vision encoder
        if 'beit3' in config['multi_modal_encoder']:

            if not config['multi_modal_encoder'].endswith(config['task']):
                if config['task'] in ("recsys"):
                    model_config = "%s_recsys" % config['multi_modal_encoder']
            else:
                model_config = config['multi_modal_encoder']

            if not self.config['using_id_as_vl_input']:
                multimodal_encoder = create_model(
                    model_config,
                    pretrained=False,
                    drop_path_rate=config['drop_path'],
                    vocab_size=config['vocab_size'],
                    checkpoint_activations=False,
                )
            else:
                multimodal_encoder = create_model(
                    model_config,
                    pretrained=False,
                    drop_path_rate=config['drop_path'],
                    vocab_size=config['vocab_size'],
                    checkpoint_activations=False,
                    item_num=self.item_num,
                )

            if config['finetune']:
                utils.load_model_and_may_interpolate(self.config_class.finetune,
                                                     multimodal_encoder, self.config_class.model_key,
                                                     self.config_class.model_prefix)

        elif 'vlmo' in config['multi_modal_encoder']:
            print('Using vlmo as multi encoder')
            multimodal_encoder = VLMo(config)

            if config['checkpoint'] is not None:
                # msg = multimodal_encoder.load_state_dict(torch.load(config['checkpoint']))
                # print(msg)
                print('load checkpoint from {}'.format(config['checkpoint']))

            multimodal_encoder = multimodal_encoder.to(config['device'])

        elif 'clip' in config['multi_modal_encoder']:
            print('Using clip as multi encoder')
            multimodal_encoder, self.clip_preprocess = clip.load("ViT-B/16")

        return multimodal_encoder

    def build_embeds(self, config, multimodal_encoder, data_loader):
        if 'beit3' in self.config['multi_modal_encoder'] and self.config['using_cls_feature']:
            # self.device = 'cuda'

            cls_reps = []
            for batch in data_loader:
                batch['image'] = batch['image'].to(self.device)
                batch['language_tokens'] = batch['language_tokens'].to(self.device)
                batch['padding_mask'] = batch['padding_mask'].to(self.device)

                multimodal_encoder.to(self.device)

                cls_rep = multimodal_encoder(
                    image=batch['image'], text_description=batch['language_tokens'],
                    padding_mask=batch['padding_mask'], only_infer=True)

                for embed in cls_rep.unbind():
                    cls_reps.append(embed.detach().cpu().numpy())

            return torch.FloatTensor(cls_reps).to(
                self.device)

        elif 'beit3' in self.config['multi_modal_encoder']:
            # self.device = 'cuda'
            multimodal_encoder.zero_grad()
            text_embeddings = []
            image_embeddings = []
            self.vl_loss = 0
            for batch in data_loader:

                batch['image'] = batch['image'].to(self.device)
                batch['language_tokens'] = batch['language_tokens'].to(self.device)
                batch['padding_mask'] = batch['padding_mask'].to(self.device)
                if self.config['using_id_as_vl_input']:
                    batch['item_id'] = batch['image_id'].to(self.device)


                multimodal_encoder.to(self.device)
                if self.config['using_vl_loss']:
                    if not self.config['using_id_as_vl_input']:
                        vision_cls, language_cls, vl_loss = multimodal_encoder(
                            image=batch['image'], text_description=batch['language_tokens'],
                            padding_mask=batch['padding_mask'], only_infer=False)
                    else:
                        vision_cls, language_cls, vl_loss = multimodal_encoder(
                            image=batch['image'], text_description=batch['language_tokens'],
                            padding_mask=batch['padding_mask'], only_infer=False,
                            item_id=batch['item_id'])
                    self.vl_loss += vl_loss.detach().cpu()
                else:
                    if not self.config['using_id_as_vl_input']:
                        vision_cls, language_cls = multimodal_encoder(
                            image=batch['image'], text_description=batch['language_tokens'],
                            padding_mask=batch['padding_mask'], only_infer=True)
                    else:
                        vision_cls, language_cls = multimodal_encoder(
                            image=batch['image'], text_description=batch['language_tokens'],
                            padding_mask=batch['padding_mask'], only_infer=True,
                            item_id=batch['item_id'])

                for embed in vision_cls.unbind():
                    image_embeddings.append(embed.detach().cpu().numpy())
                for embed in language_cls:
                    text_embeddings.append(embed.detach().cpu().numpy())

            self.vl_loss = self.vl_loss / len(data_loader)
            self.vl_loss = self.vl_loss.to(self.device)

            return torch.FloatTensor(image_embeddings).to(
                self.device), torch.FloatTensor(text_embeddings).to(
                self.device)

        if 'clip' in config['multi_modal_encoder']:
            text_embeddings = []
            image_embeddings = []
            for batch in data_loader:
                images = batch['image'][0]
                images = images.cuda()
                texts = batch['text']
                image_input = images  # torch.tensor(np.stack(images)).cuda()
                text_tokens = clip.tokenize([desc[0:77] for desc in texts]).cuda()

                with torch.no_grad():
                    image_features = multimodal_encoder.encode_image(image_input).float()
                    text_features = multimodal_encoder.encode_text(text_tokens).float()

                # with vl layers
                for embed in image_features.unbind():
                    image_embeddings.append(embed.detach().cpu().numpy())
                for embed in text_features:
                    text_embeddings.append(embed.detach().cpu().numpy())

            return torch.FloatTensor(image_embeddings).to(
                self.device), torch.FloatTensor(text_embeddings).to(
                self.device)

        elif 'vlmo' in config['multi_modal_encoder']:
            text_embeddings = []
            image_embeddings = []
            cls_features = []
            raw_cls_features = []
            text_embeddings_infer_text = []
            text_embeddings_infer_text_vl = []
            text_embeddings_infer_text_ft = []
            image_embeddings_infer_image = []
            image_embeddings_infer_image_vl = []
            image_embeddings_infer_image_ft = []

            for batch in data_loader:

                ret_dict = multimodal_encoder.infer_text_ft(batch)
                for embed in ret_dict["cls_feats"].unbind():
                    text_embeddings_infer_text_ft.append(embed.detach().cpu().numpy())

                ret_dict = multimodal_encoder.infer_image_ft(batch)
                for embed in ret_dict["cls_feats"].unbind():
                    image_embeddings_infer_image_ft.append(embed.detach().cpu().numpy())

            return torch.FloatTensor(image_embeddings_infer_image_ft).to(
                self.device), torch.FloatTensor(text_embeddings_infer_text_ft).to(
                self.device)
        else:
            print(f'Not implemented yet for this model'
                  f'{config["multi_modal_encoder"]}')
            raise NotImplementedError

    def update_train_embeddings(self):
        if 'beit3' in self.config['multi_modal_encoder'] and self.config['using_cls_feature']:
            self.cls_train_feat \
                = self.build_embeds(self.config,
                                    self.multi_modal_encoder,
                                    self.train_data_loader)

        else:
            self.v_train_feat, self.t_train_feat \
                = self.build_embeds(self.config,
                                    self.multi_modal_encoder,
                                    self.train_data_loader)
    def update_test_embeddings(self):
        if 'beit3' in self.config['multi_modal_encoder']and self.config['using_cls_feature']:
            self.cls_test_feat \
                = self.build_embeds(self.config,
                                    self.multi_modal_encoder,
                                    self.test_dataloader)
        else:
            self.v_test_feat, self.t_test_feat \
                = self.build_embeds(self.config,
                                    self.multi_modal_encoder,
                                    self.test_dataloader)


    def clean_train_embeddings(self):
        if 'beit3' in self.config['multi_modal_encoder']:
            self.cls_train_feat = None
            torch.cuda.empty_cache()
        else:
            self.v_train_feat = None
            self.t_train_feat = None
            torch.cuda.empty_cache()


    def clean_test_embeddings(self):
        if 'beit3' in self.config['multi_modal_encoder']:
            self.cls_test_feat = None
            torch.cuda.empty_cache()

        else:
            self.v_test_feat = None
            self.t_test_feat = None
            torch.cuda.empty_cache()