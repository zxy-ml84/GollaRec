# coding: utf-8
"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""
import warnings

# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='UGNN_ete', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    # parser.add_argument('--image_size', type=int, default=224, help='image size')
    # parser.add_argument('--multi_modal_encoder', type=str, default='vlmo',  help='multi-modal encoder')
    # parser.add_argument('--movie_file', type=str, default='data/Amazon_Sports_Dataset/raw_text.json', help='movie file')
    # parser.add_argument('--image_root', type=str, default='data/Amazon_Sports_Dataset/raw_image_id', help='image root')
    # parser.add_argument('--whole_word_masking', type=bool, default=True, help='whole word masking')
    #


    config_dict = {
        'gpu_id': 1,
        'image_size': 224,
        'input_size': 224,
        'multi_modal_encoder': 'beit3_base_patch16_224',
        'json_path': 'data/Amazon_Baby_Dataset/raw_text.json',
        'image_root': 'data/Amazon_Baby_Dataset/raw_image_id',
        'whole_word_masking': True,
        'mlm_prob': 0.15,
        'task': 'recsys',
        'sentencepiece_model': 'beit3.spm',
        'num_max_bpe_tokens': 64,
        'captioning_mask_prob': 0.15,
        'num_workers': 4,
        'pin_mem': False,
        'dist_eval': False,
        'vocab_size': 64010,
        'finetune': '../ckpt/beit3/beit3_base_itc_patch16_224.pth',
        'model_key': 'model|module',
        'model_prefix': '',
        'encoder_batch_size': 64,
        'train_interpolation': 'bicubic',
        'randaug': False,
        'batch_update_embeddings': True,
        'enable_wandb': True,
        'using_cls_feature': False,
        'using_vl_loss': True,
        'using_id_as_vl_input': True,
    }

    args, _ = parser.parse_known_args()



    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


