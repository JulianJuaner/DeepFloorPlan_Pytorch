from sourcecode.dataset.dataset import FloorPlanDataset_cfg, 
import cv2
import os
import numpy as np


CUBE_INDEX = [[2,1,0], [3,2,0],
              [4,5,6], [6,7,4],
              [5,1,2], [2,6,5],
              [0,1,5], [5,4,0]]
'''
-1 -1 -1 |  1 -1 -1 |  1  1 -1 | -1  1 -1
-1 -1  1 |  1 -1  1 |  1  1  1 | -1  1  1
'''

def vis2mesh(cfg):
    img_folder = os.path.join(cfg.FOLDER, 'vistest')
    mesh_folder = os.path.join(cfg.FOLDER, 'mesh')
    os.makedirs(mesh_folder)
    img_list = os.listdir(img_folder)
    for img in img_list:
        img = cv2.imread(os.path.join(img_folder, img))
        walls = np.zeros((img.shape[0], img.shape[1]))


if "__main__" in __name__:
    # initialize exp configs.
    parser = argparse.ArgumentParser()
    OptionInit = Options(parser)
    parser = OptionInit.initialize(parser)
    opt = parser.parse_args()
    folder_name = opt.exp
    print(folder_name)
    exp_cfg = make_config(os.path.join(folder_name, "exp.yaml"))
    print(exp_cfg)
    vis2mesh(exp_cfg)