import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from utils.dataset.caption_dataset import  vl_pretrain_dataset, movie_dataset
from utils.dataset.randaugment import RandomAugment
from utils.dataset.utils import GaussianBlur


def create_dataset(dataset, config):

    # transform from VLMO
    inception_normalize = transforms.Compose(
        [transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(config['image_size'], scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            inception_normalize,
        ])

    test_transform = transforms.Compose(
        [
            transforms.Resize((config['image_size'], config['image_size']), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            inception_normalize,
        ])


    if dataset=='vl_pretrain':
        return vl_pretrain_dataset(config, config['train_file'], train_transform)

    elif dataset=='movie_dataset':
        dataset = movie_dataset(config, config['movie_file'], test_transform)
        return dataset

    elif dataset=='re':     
        # train_dataset = re_train_dataset(config['train_file'], train_transform, config['image_root'])
        # val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])
        # test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        train_dataset = vl_pretrain_dataset(config, config['train_file'], train_transform)
        val_dataset = vl_pretrain_dataset(config, config['val_file'], test_transform)
        test_dataset = vl_pretrain_dataset(config, config['test_file'], test_transform)

        return [train_dataset, val_dataset, test_dataset]


    

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks,
                                                      rank=global_rank, shuffle=shuffle, drop_last=True)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    