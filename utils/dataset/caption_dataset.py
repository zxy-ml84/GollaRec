import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)


def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            BertTokenizer.from_pretrained(
                from_pretrained, do_lower_case="uncased" in from_pretrained
            )
        torch.distributed.barrier()
    return BertTokenizer.from_pretrained(
        from_pretrained, do_lower_case="uncased" in from_pretrained
    )


class vl_pretrain_dataset(Dataset):
    def __init__(self, _config, ann_file, transform, max_words=40):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words

        print('loading json file done!')

        tokenizer = 'bert-base-uncased'
        self.tokenizer = get_pretrained_tokenizer(tokenizer)

        collator = (
            DataCollatorForWholeWordMask
            if _config["whole_word_masking"]
            else DataCollatorForLanguageModeling
        )

        self.mlm_collator = collator(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=_config["mlm_prob"]
        )

        # if _config['loss_name']['irtr']:

        self.all_texts, self.index_mapper = self.get_all_texts()

    def get_text(self, raw_index):

        index, caption_index = self.index_mapper[raw_index]

        text = self.all_texts[index][caption_index]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_words,
            return_special_tokens_mask=True,
        )

        mlms = self.mlm_collator([encoding])

        mlm_ids = mlms["input_ids"]
        mlm_labels = mlms["labels"]

        # input_ids = torch.zeros_like(mlm_ids)
        # attention_mask = torch.zeros_like(mlm_ids)

        input_ids, attention_mask = (
            torch.tensor(encoding["input_ids"]),
            torch.tensor(encoding["attention_mask"]),
        )

        dict_batch = {}
        txt_key = "text"
        dict_batch[f"{txt_key}_ids"] = input_ids
        dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
        dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids[0]
        dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels[0]
        dict_batch[f"{txt_key}_masks"] = attention_mask
        dict_batch['img_index'] = index
        dict_batch['caption_index'] = caption_index
        dict_batch['raw_index'] = raw_index
        return dict_batch

    def get_all_texts(self):
        all_texts = []
        index_mapper = dict()

        j = 0
        for i in range(len(self.ann)):
            ann = self.ann[i]

            if type(ann['caption']) == list:
                for _j in range(len(ann['caption'])):
                    all_texts.append(ann['caption'][_j])
                    index_mapper[j] = (i, _j)
                    j += 1
            else:
                all_texts.append(ann['caption'])
                index_mapper[j] = (i, 0)
                j += 1



        # for i, texts in enumerate(all_texts):
        #     for _j in range(len(texts)):
        #         index_mapper[j] = (i, _j)
        #         j += 1

        return all_texts, index_mapper



    def __len__(self):
        return len(self.index_mapper)

    def __getitem__(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]

        ann = self.ann[index]

        # if type(ann['caption']) == list:
        #     caption = pre_caption(random.choice(ann['caption']), self.max_words)
        # else:
        #     caption = pre_caption(ann['caption'], self.max_words)

        batch = self.get_text(index)


        if 'visual-genome' in ann['image']:
            img_name= ann['image'].split('/')[-1]
            img_path = os.path.join('../datasets/VG_100K', img_name)
        elif 'coco' in ann['image']:
            img_name = ann['image'].split('/')[-1]
            year = ann['image'].split('/')[-2]
            img_path = os.path.join('../datasets/coco', year , img_name)
        elif 'ConceptualCaptions' in ann['image']:
            # img_name = ann['image'].split('/')[-1]
            # split = ann['image'].split('/')[-2]
            # img_path = os.path.join('../datasets/DownloadConceptualCaptions', split, img_name)
            img_path = ann['image']
        elif 'SBU' in ann['image']:
            # img_name = ann['image'].split('/')[-1]
            # split = ann['image'].split('/')[-2]
            # img_path = os.path.join('../datasets/DownloadConceptualCaptions', split, img_name)
            img_path = ann['image']
        elif 'flickr30k' in ann['image']:
            img_name = ann['image'].split('/')[-1]
            # split = ann['image'].split('/')[-2]
            #Flickr
            img_path = os.path.join('../datasets/Flickr/flickr30k_images/flickr30k_images/', img_name)
            #img_path = os.path.join('../datasets/flickr30k_images/flickr30k_images/', img_name)
        else:
            print('Dataset is not supported')
            raise NotImplementedError

        image = Image.open(img_path).convert('RGB')
        image1 = self.transform(image)

        batch['image'] = [image1]

        return batch


class movie_dataset(Dataset):
    def __init__(self, _config, ann_file, transform, max_words=40):

        self.transform = transform
        self.max_words = max_words
        self.image_root = _config['image_root']
        self.is_1m = False
        if '1M' in ann_file:
            self.is_1m = True
        print('loading json file done!')

        tokenizer = 'bert-base-uncased'
        self.tokenizer = get_pretrained_tokenizer(tokenizer)

        collator = (
            DataCollatorForWholeWordMask
            if _config["whole_word_masking"]
            else DataCollatorForLanguageModeling
        )

        self.mlm_collator = collator(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=_config["mlm_prob"]
        )

        # if _config['loss_name']['irtr']:
        self.ann  = self.clean_json()


    def clean_json(self):
        new_ann = []
        for image_id, data in self.ann.items():
            try:
                Image.open(os.path.join(self.image_root, image_id+'.jpg'))
                new_data = dict()
                new_data['image_id'] = image_id
                new_data['caption'] = data
                new_ann.append(new_data)
            except:
                pass

        print('clean json done!')
        print('Valid json length: ', len(new_ann))
        return new_ann

    def get_text(self, index):


        if self.is_1m:
            text = str(self.ann[index]['caption']['name']) + ' ' + str(self.ann[index]['caption']['intro']) + ' ' + \
                   str(self.ann[index]['caption']['genre'])
        else:
            text = self.ann[index]['caption']

        # print(text)

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_words,
            return_special_tokens_mask=True,
        )

        mlms = self.mlm_collator([encoding])

        mlm_ids = mlms["input_ids"]
        mlm_labels = mlms["labels"]

        # input_ids = torch.zeros_like(mlm_ids)
        # attention_mask = torch.zeros_like(mlm_ids)

        input_ids, attention_mask = (
            torch.tensor(encoding["input_ids"]),
            torch.tensor(encoding["attention_mask"]),
        )

        dict_batch = {}
        txt_key = "text"
        dict_batch['text'] = text
        dict_batch[f"{txt_key}_ids"] = input_ids
        dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
        dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids[0]
        dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels[0]
        dict_batch[f"{txt_key}_masks"] = attention_mask
        # dict_batch['img_index'] = index
        return dict_batch


    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        batch = self.get_text(index)

        img_path = os.path.join(self.image_root, ann['image_id']+'.jpg')

        image = Image.open(img_path).convert('RGB')
        image1 = self.transform(image)

        batch['image'] = [image1]

        return batch
