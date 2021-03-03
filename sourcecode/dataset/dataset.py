import sys
import os
import cv2
import torch
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from sourcecode.dataset.transform import custom_transform

FloorPlanDataset_cfg = {
    'type': 'FloorPlanDataset',
    'color_maps': [
        [0, 0, 0],  # background
        [192, 192, 224],  # closet
        [192, 255, 255],  # batchroom/washroom
        [224, 255, 192],  # livingroom/kitchen/dining room
        [255, 224, 128],  # bedroom
        [255, 160, 96],  # hall
        [255, 224, 224],  # balcony
        [255, 60, 128],  # extra label for opening (door&window)
        [255, 255, 255],  # extra label for wall line
        [77, 77, 77] # ignore
    ],
}

class FloorPlanDataset(Dataset):
    """
    Dataset for Floor Plan (r2v).
    """

    def __init__(self, opts, full_cfg):

        self.opts = opts
        self.root = self.opts.root
        self.mode = self.opts.mode
        self.color_maps = FloorPlanDataset_cfg['color_maps']
        self.layers = opts.layers
        self.data_path = opts.data_path
        self.data_list = self._read_data_list(self.data_path)
        self.full_cfg = full_cfg
        self.mean = full_cfg.DATA.mean
        self.std = full_cfg.DATA.std
        self.value_scale = full_cfg.DATA.value_scale

    def _read_data_list(self, data_path):
        data_list = []
        for data_file in data_path:
            with open(data_file, 'r') as r:
                for line in r.readlines():
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    img_path = line
                    mask_path = line[:-10] + '_rooms.png'
                    data_list.append((img_path, mask_path))
        return data_list

    def __len__(self):
        return len(self.data_list)

    def mask_convert(self, mask_img):
        # h, w = mask_img.shape([:2])
        res_mask = np.array(np.zeros(mask_img.shape[:2]), dtype=np.uint8)
        for i in range(len(self.color_maps)):
            res_mask[(mask_img == self.color_maps[i]).all(2)] = i
        # logger.info("{}".format(np.unique(res_mask)))
        return res_mask

    def to_color(self, mask_img):
        h, w = mask_img.shape[:2]
        res_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(len(self.color_maps)):
            res_mask[mask_img == i] = self.color_maps[i]
        return res_mask

    def __getitem__(self, item):
        img_path, mask_path = self.data_list[item]
        if self.root is not None:
            img_path = '{}/{}'.format(self.root, img_path)
            mask_path = '{}/{}'.format(self.root, mask_path)
        image = cv2.imread(img_path, 1)
        if image is None:
            logger.error("{} doesn't exist.".format(img_path))
            raise FileExistsError(img_path)
        if os.path.exists(mask_path):
            basename = img_path[:-10]
            room = cv2.cvtColor(cv2.imread(basename + '_rooms.png', 1), cv2.COLOR_BGR2RGB)
            wall = cv2.imread(basename + '_wall.png', 0)
            close = cv2.imread(basename + '_close.png', 0)

            if self.mode == 'training':
                # add ignore if training.
                room_sum = np.sum(room, axis=-1)
                binary = np.zeros(image.shape[:2]).astype(np.uint8)
                binary[room_sum>=6] = 255
                # cv2.imwrite('binary.png', np.hstack((cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), room)))
                dilate = cv2.dilate(binary, np.ones((5,5)), 2)
                binary = dilate - binary
                # cv2.imwrite('binary.png', binary)
                room[binary != 0 ] = [77, 77, 77]

            room[wall == 255] = [255, 255, 255]
            room[close == 255] = [255, 60, 128]
        
            mask = self.mask_convert(room)
        else:
            mask = np.array(np.zeros(image.shape[:2]), dtype=np.uint8)
        
        for func in self.full_cfg.DATA.preprocessor:
            image, mask = custom_transform(func, image, mask)

        if self.mode == 'training':
            for func in self.full_cfg.TRAIN.data_argumentation:
                image, mask = custom_transform(func, image, mask)

        org_image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        mask = torch.from_numpy(mask).long()

        image, mask = TorchNormalize(
            mean=self.mean, std=self.std)(org_image, mask)
        # print(np.unique(mask))
        return {
            'imgs':
            image.float(),
            'org_imgs':
            (org_image.float() / (self.value_scale / 255))[[2, 1, 0], :, :],
            'targets':
            mask.long(),
            'img_path':
            img_path
        }

class TorchNormalize:

    def __init__(self, mean, std, **kwargs):
        self.mean = mean if type(mean) is list else [mean]
        self.std = std if type(std) is list else [std]

    def __call__(self, img, mask, **kwargs):
        img = transforms.Normalize(self.mean, self.std)(img)
        return img, mask

def to_color(mask_img):
    color_maps = FloorPlanDataset_cfg['color_maps']
    h, w = mask_img.shape[:2]
    res_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(color_maps)):
        res_mask[mask_img == i] = color_maps[i]
    return res_mask
