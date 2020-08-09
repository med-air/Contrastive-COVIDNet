import collections
import os
import pprint
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.io import imread
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.segmentation import slic, mark_boundaries
import cv2
from util import read_filepaths, read_filepaths2


class COVID_CT_Dataset(Dataset):
    """
    Code for reading the COVIDxDataset
    """

    def __init__(self, args, mode, n_classes=2, dataset_path='./datasets', dim=(224, 224)):
        self.mode = mode
        self.CLASSES = n_classes
        self.dim = dim
        self.COVIDxDICT = {'normal': 0, 'COVID-19': 1}
        trainfile = os.path.join(dataset_path, 'COVID-CT', 'train_split.txt')
        testfile = os.path.join(dataset_path, 'COVID-CT', 'test_split.txt')

        newtrainpath, newtrainlabel = read_filepaths2('../data/SARS-Cov-2/train_split.txt')
        newtestpath, newtestlabel = read_filepaths2('../data/SARS-Cov-2/test_split.txt')

        if mode == 'train':
            self.paths, self.labels = read_filepaths(trainfile, self.mode)
            self.paths.extend(self.paths)
            self.labels.extend(self.labels)
            self.paths.extend(self.paths)
            self.labels.extend(self.labels)

            self.paths.extend(newtrainpath)
            self.labels.extend(newtrainlabel)

        elif mode == 'test':
            self.paths, self.labels = read_filepaths(testfile, self.mode)
            self.paths.extend(newtestpath)
            self.labels.extend(newtestlabel)

        print("{} examples =  {}".format(mode, len(self.paths)))
        self.mode = mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_tensor, site = self.load_image(self.paths[index])
        label_tensor = torch.tensor(self.COVIDxDICT[self.labels[index]], dtype=torch.long)

        return image_tensor, label_tensor, site

    def load_image(self, img_path):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))

        if img_path.split('/')[2] == 'COVID-CT':
            site = 'ucsd'
        else:
            site = 'new'

        image = Image.open(img_path).convert('RGB')

        inputsize = 224
        transform = {
            'train': transforms.Compose(
                [transforms.Resize(256),
                 transforms.RandomCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 ]),
            'test': transforms.Compose(
                [transforms.Resize([inputsize, inputsize]),
                 ])
        }

        transformtotensor = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        if self.mode == 'train':
            image = transform['train'](image)
        else:
            image = transform['test'](image)

        image_tensor = transformtotensor(image)

        return image_tensor, site
