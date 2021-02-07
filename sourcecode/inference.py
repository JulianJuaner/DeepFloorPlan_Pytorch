from sourcecode.model import HRNetW18SmallV2, build_head
from sourcecode.dataset import FloorPlanDataset, to_color
from sourcecode.configs import make_config, Options
from sourcecode.utils.optim_loss import adjust_learning_rate, compute_acc
from sourcecode.train import FloorPlanModel
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import argparse
import torch
import math
import os
import cv2


def inference(cfg):
    test_dataset = FloorPlanDataset(cfg.DATA.test_data, cfg)

    test_loader = DataLoader(
            test_dataset,
            batch_size=1, 
            shuffle=False,
            num_workers=0,
        )
    
    model = FloorPlanModel(cfg).eval()
    model.cuda()
    os.makedirs(os.path.join(cfg.FOLDER, 'vistest'), exist_ok=True)
    model_ckpt_path = os.path.join(cfg.FOLDER, 'ckpt' + str(cfg.EVAL.iter) + '.pth')
    model.load_state_dict(torch.load(model_ckpt_path), strict=False)
    test_iter = iter(test_loader)
    acc_sum = torch.zeros((cfg.MODEL.num_classes+1)).cuda()
    pixel_sum = torch.zeros((cfg.MODEL.num_classes+1)).cuda()
    print('start evaluation.')
    counter = 1
    for eval_step in range(len(test_iter)):
        eval_data = next(test_iter)
        res = model(eval_data['imgs'].cuda())
        if isinstance(res, list):
            res = res[0]
        acc_sum, pixel_sum = compute_acc(res, eval_data['targets'].cuda(), acc_sum, pixel_sum)
        _, mask = torch.max(res, dim=1)

        cv2.imwrite(os.path.join(cfg.FOLDER, 'vistest',str(counter) + '.png'), np.hstack((eval_data['org_imgs'][0].numpy().transpose(1,2,0),
                    to_color(mask[0].cpu().detach().numpy()))))
        counter += 1

    acc_value = []
    for i in range(res.shape[1]):
        acc_value.append((acc_sum[i].float()+1e-10)/(pixel_sum[i].float()+1e-10))
    print(acc_sum)
    print(pixel_sum)
    acc_class = sum(acc_value)/len(acc_value)
    acc_total = (acc_sum[-1].float()+1e-10)/(pixel_sum[-1].float()+1e-10)
    print('eval_class_acc: %.2f'%(acc_class.item()*100),
        'eval_overall_acc: %.2f'%(acc_total.item()*100))

        
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
    inference(exp_cfg)