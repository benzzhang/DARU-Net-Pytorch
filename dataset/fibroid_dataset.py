import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from .util import rotate, translate

__all__ = ['FibroidDataset']


class FibroidDataset(Dataset):

    def __init__(self, img_list, transform_paras, prefix='data/', img_size=(512, 512), sigma=5):

        self.img_size = tuple(img_size)
        self.transform_paras = transform_paras
        self.prefix = prefix
        # read img_list and metas
        self.img_list = [l.strip() for l in open(img_list).readlines()]
        self.img_data_list = self.__readAllImgData__()
        self.mask_data_list = self.__readAllMaskData__()

    def __readAllImgData__(self):
        img_data_list = []
        for index in range(len(self.img_list)):
            img_path = os.path.join(os.path.join(self.prefix, 'img'), self.img_list[index].strip())
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.img_size)
            img_data_list.append(img)
        return img_data_list

    def __readAllMaskData__(self):
        mask_data_list = []
        for index in range(len(self.img_list)):
            mask_path = os.path.join(os.path.join(self.prefix, 'label'), self.img_list[index].strip())
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.img_size)
            mask_data_list.append(mask)
        return mask_data_list

    def __getitem__(self, index):
        img = self.img_data_list[index]
        img = np.expand_dims(img, axis=0)
        _, w, h, = img.shape

        mask = self.mask_data_list[index]
        mask = np.expand_dims(mask, axis=0)

        idx_name = self.img_list[index]

        # transform use rotate and translate
        angle = int(np.random.rand() * self.transform_paras['rotate_angle'])
        offset_x = int(np.random.rand() * self.transform_paras['offset'][0] * w)
        offset_y = int(np.random.rand() * self.transform_paras['offset'][1] * h)

        # rotate
        if np.random.rand() < 0.5:
            img = rotate(img, angle)
            mask = rotate(mask, angle)

        # translation
        if np.random.rand() < 0.5:
            img = translate(img, [offset_x, offset_y])
            mask = translate(mask, [offset_x, offset_y])

        # scale img pixel: 0 ~ 1
        if np.max(img) > 1:
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) 

        if np.max(mask) > 1:
            mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask)) 

        # shape (C,W,H)
        return torch.FloatTensor(img), torch.FloatTensor(mask), idx_name

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    prefix = '../data'
    img_list = '../data/KFold-train-0.txt'

    transform_list = {'rotate_angle': 15, 'offset': [0.2, 0.2]}
    fibroid_dataset = FibroidDataset(img_list, transform_list, prefix)

    for i in range(fibroid_dataset.__len__()):
        image, mask = fibroid_dataset.__getitem__(i)

        image, mask = image.numpy(), mask.numpy()
        # print(f'max: {np.max(image)}, min:{np.min(image)}')

        # 失真图像 image & mask
        image = image * 255
        image = image.astype(np.uint8)
        mask = mask * 255
        mask = mask.astype(np.uint8)

        # shape - (W,H,C) - (512,512,1)
        image = image.transpose(1, 2, 0)
        mask = mask.transpose(1, 2, 0)

        cv2.imshow('image', image)
        cv2.imshow('heatmap', mask)

        if cv2.waitKey() == ord('q'):
            break
        else:
            continue
