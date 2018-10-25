# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Data provider"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self, data_path, data_split, 
        vocab, adapt_set=False, sigma=0.0):
        self.data_path = data_path
        self.split = data_split
        self.vocab = vocab
        loc = data_path + '/'
        self.adapt_set = adapt_set
        self.sigma = sigma

        data_split = data_split.replace('val', 'dev')
        # Captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index/self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        
        if self.adapt_set:
            image_adapt = add_noise(image, self.sigma)
            image = add_noise(image, self.sigma)
            image = (image, image_adapt)

        return image, target, index, img_id

    def __len__(self):
        return self.length


def add_noise(x, sigma=0.):
    return x+x.clone().normal_(0., sigma)


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    images_ema = None    
    if len(images[0]) == 2:
        images, images_ema = zip(*images)
        images_ema = torch.stack(images_ema, 0)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    if images_ema is not None:
        return images, images_ema, targets, lengths, ids

    return images, targets, lengths, ids


def get_precomp_loader(
        data_path, data_split, vocab, 
        opt, batch_size=100, shuffle=True, 
        num_workers=2, adapt_set=False, noise=0.0
    ):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(
        data_path, data_split, vocab, adapt_set, sigma=noise)

    data_loader = torch.utils.data.DataLoader(
        dataset=dset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return data_loader


def _get_loaders(data_name, vocab, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(
        dpath, 'train', vocab, opt,
        batch_size, True, workers
    )
    val_loader = get_precomp_loader(
        dpath, 'dev', vocab, opt,
        batch_size, False, workers
    )
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(
        dpath, split_name, vocab, opt,
        batch_size, False, workers
    )
    return test_loader

### EDIT BALLESTER ###
def get_loader(
        data_name, batch_size, workers, opt,
        split='train', adapt_set=False, vocab=None
    ):

    dpath = os.path.join(opt.data_path, data_name)

    if opt.data_name.endswith('_precomp'):

        if split in ['train', 'val', 'test']:
            loader = get_precomp_loader(
                data_path=dpath,
                data_split=split,                
                vocab=vocab,
                opt=opt,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=workers,                
                adapt_set=adapt_set,
                noise=opt.noise,
            )
        elif split == 'adapt':
            adapt_dataset = UnlabeledPrecompDataset(
                data_path=dpath,
                sigma=opt.noise,
            )
            loader = torch.utils.data.DataLoader(
                dataset=adapt_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
            )

    
    return loader
