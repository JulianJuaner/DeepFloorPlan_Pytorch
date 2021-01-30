import sys
import os

import torch
from torch.utils.data import Dataset

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
    'layers': 1,
    'mode': 'train'
}

class FloorPlanDataset(Dataset):
    """
    Dataset for Floor Plan (r2v).
    """

    def __init__(self, opts):
        """
        Args:
            data_name: dataset name
            data_path: list of data path
            root: the root path, it will be used as prefix for each image path and json path.
            **kwargs:
        """
        self.opts = opts
        self.color_maps = FloorPlanDataset_cfg['color_maps']
        self.layers = opts.floor_plan_layers
        self.mode = self.kwargs.get('mode')
        self.data_list = self._read_data_list(self.data_path)

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
                    # logger.info("{}".format(mask_path))
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

    def __getitem__(self, item):
        img_path, mask_path = self.data_list[item]
        if self.root is not None:
            img_path = '{}/{}'.format(self.root, img_path)
            mask_path = '{}/{}'.format(self.root, mask_path)
        image = cv2.imread(img_path, 0 if self.in_channels == 1 else 1)
        if image is None:
            logger.error("{} doesn't exist.".format(img_path))
            raise FileExistsError(img_path)
        if self.in_channels != 1 and self.in_chn_type == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if os.path.exists(mask_path):
            basename = img_path[:-10]
            room = cv2.cvtColor(cv2.imread(basename + '_rooms.png', 1), cv2.COLOR_BGR2RGB)
            wall = cv2.imread(basename + '_wall.png', 0)
            close = cv2.imread(basename + '_close.png', 0)

            if self.mode == 'training':
                # add ignore if training.
                binary = np.zeros(image.shape[:2])
                binary[(room != [0, 0, 0]).all(2)] = 255
                dilate = cv2.dilate(binary, np.ones(3,3), 2)
                binary = dilate - binary
                room[binary != 0 ] = [77, 77, 77]

            room[wall == 255] = [255, 255, 255]
            room[close == 255] = [255, 60, 128]
        
            mask = self.mask_convert(room)
        else:
            mask = np.array(np.zeros(image.shape[:2]), dtype=np.uint8)

        self._check_shape(image, mask, img_path, mask_path)
        for func in self.preprocessor_list:
            image, mask = func(image, mask)
        for func in self.argumentation_list:
            image, mask = func(image, mask)
        org_image, mask = self._to_tensor(image, mask)
        image, mask = TorchNormalize(
            mean=self.mean, std=self.std)(org_image, mask)
        return {
            'imgs':
            image.float(),
            'org_imgs':
            (org_image.float() / (self.value_scale / 255))[[2, 1, 0], :, :]
            if self.in_channels == 3 else org_image.float() /
            (self.value_scale / 255),
            'targets':
            mask.long(),
            'img_path':
            img_path
        }

