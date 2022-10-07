# coding:utf8
import os
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

import cv2


from albumentations import (Flip, HorizontalFlip, VerticalFlip,
                            RandomRotate90, ShiftScaleRotate, Transpose,
                            RandomCrop, CenterCrop,
                            Cutout,
                            Blur, MedianBlur, GaussianBlur,
                            # RGBShift, ColorJitter,
                            # HueSaturationValue,
                            # RandomBrightnessContrast,
                            # RandomGamma,
                            # RandomRain, RandomFog,
                            # CoarseDropout, ChannelDropout,
                            # GridDistortion, ElasticTransform, CLAHE,
                            Resize,
                            # OneOf,
                            Compose
                            )


class dataset_green(Dataset):

    def __init__(self, root, train=False, val=False, test=False, train_test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.train = train
        self.val = val
        self.test = test
        self.train_test = train_test
        self.root = root

        train_path = self.root + 'train/'
        test_path = self.root + 'test/'

        train_paths = [os.path.join(train_path, train_img) for train_img in os.listdir(train_path) if '.DS_Store' not in train_img]
        test_paths = [os.path.join(test_path, test_img) for test_img in os.listdir(test_path) if '.DS_Store' not in test_img]

        self.imgs, self.masks = [], []

        if self.val:
            for test_path in test_paths:
                for test_names in os.listdir(test_path):
                    if '.DS_Store' not in test_names:
                        self.imgs.append(test_path + '/' + test_names + '/' + 'Image {}-2.png'.format(test_names))
                        self.masks.append(test_path + '/' + test_names + '/' + '2.1.png')
        elif self.test:
            for test_path in test_paths:
                for test_names in os.listdir(test_path):
                    if '.DS_Store' not in test_names:
                        self.imgs.append(test_path + '/' + test_names + '/' + 'Image {}-2.png'.format(test_names))
                        self.masks.append(test_path + '/' + test_names + '/' + '2.1.png')
        else:

            for train_path in train_paths:
                for train_names in os.listdir(train_path):
                    if '.DS_Store' not in train_names:
                        self.imgs.append(train_path + '/' + train_names + '/' + 'Image {}-2.png'.format(train_names))
                        self.masks.append(train_path + '/' + train_names + '/' + '2.1.png')

        # print(self.imgs)
        # print(self.masks)
        # print(len(self.imgs))
        # print(len(self.masks))

        self.transforms = Compose([
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomRotate90(p=0.5),
            Transpose(p=0.5)
        ])


    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        mask_path = self.masks[index]

        # if self.train:
        #     data = self.transforms(image=data)['image']

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        mask = mask[:, :, np.newaxis]

        mask[np.where(mask > 127)] = 255

        image = self.normlize(image)
        mask = mask / 255.

        # print(np.max(image), np.min(image))
        # print(np.max(mask), np.min(mask))

        # HWC to CHW
        image = image.transpose((2, 0, 1))
        image = torch.FloatTensor(image.copy())

        mask = mask.transpose((2, 0, 1))
        mask = torch.FloatTensor(mask.copy())


        return image, mask

    def __len__(self):
        return len(self.imgs)

    def normlize(self, x):

        return (x - x.min()) / (x.max() - x.min())


if __name__ == '__main__':
    train_data_root = '/home/lz/sgd/data/'  # 训练集存放路径
    test_data = dataset_green(train_data_root, test=True)

    train_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

    for ii, (data, mask) in enumerate(train_dataloader):
        print(data.size())
        print(mask.size())
