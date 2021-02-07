from sourcecode.model import HRNetW18SmallV2, build_head
from sourcecode.dataset import FloorPlanDataset, to_color
from sourcecode.configs import make_config, Options
from sourcecode.utils.optim_loss import adjust_learning_rate, compute_acc
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import argparse
import torch
import math
import os
import cv2

class FloorPlanModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = HRNetW18SmallV2()
        self.backbone.load_state_dict(torch.load(cfg.MODEL.weights), strict=False)
        self.head = build_head(cfg.MODEL.head)
        self.opts = cfg

    def forward(self, x):
        backbone_out = self.backbone(x)
        return self.head(backbone_out, x.shape[2:])

def train(cfg):
    train_dataset = FloorPlanDataset(cfg.DATA.train_data, cfg)
    eval_dataset = FloorPlanDataset(cfg.DATA.eval_data, cfg)
    optimizer = 0
    train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.TRAIN.batch_size_per_gpu, 
            shuffle=True,
            num_workers=4,
            drop_last=True
        )

    eval_loader = DataLoader(
            eval_dataset,
            batch_size=1, 
            shuffle=False,
            num_workers=0,
        )
    
    model = FloorPlanModel(cfg).train()
    model.cuda()
    enc_optimizer = optim.SGD(model.backbone.parameters(), lr=cfg.TRAIN.optimizer.lr, momentum=0.9, weight_decay=0.0005)
    dec_optimizer = optim.SGD(model.head.parameters(), lr=cfg.TRAIN.optimizer.lr, momentum=0.9, weight_decay=0.0005)
    factors = [1429/1006, 1429/12.25, 1429/32.18, 1429/102.35,
                                    1429/80.46, 1429/18.42, 1429/22.86, 1429/35.06,
                                    1429/118.72]
    for i in range(len(factors)):
        factors[i] =  1 + math.log(factors[i])
    loss_weight = torch.FloatTensor(factors)
    loss_func = nn.CrossEntropyLoss(weight=loss_weight,ignore_index=9).cuda()
    num_epoch = cfg.TRAIN.max_iter//len(train_loader) + 1
    iteration = 0
    for epoch in range(num_epoch):
        train_iter = iter(train_loader)
        for step in range(len(train_iter)):
            adjust_learning_rate(enc_optimizer, iteration, cfg, cfg.TRAIN.enc_lr_factor)
            adjust_learning_rate(dec_optimizer, iteration, cfg, cfg.TRAIN.dec_lr_factor)

            data = next(train_iter)
            res = model(data['imgs'].cuda())
            
            if isinstance(res, list):
                anneal_factor = iteration / cfg.TRAIN.max_iter
                aux_loss = loss_func(res[1], data['targets'].cuda())*0.6*(1-anneal_factor) + 0.4
                total_loss = loss_func(res[0], data['targets'].cuda())*(0.5 + 0.5*anneal_factor)
                loss = aux_loss + total_loss
            else:
                loss = loss_func(res, data['targets'].cuda())
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()

            if iteration % cfg.TRAIN.eval_freq == 0 and iteration != 0:
                # evaluation.
                if iteration % cfg.TRAIN.ckpt_freq == 0:
                    print(iteration, 'saving model')
                    torch.save(model.state_dict(), '{}ckpt{}.pth'.format(
                            cfg.FOLDER,
                            iteration))

                model.eval()
                eval_iter = iter(eval_loader)
                acc_sum = torch.zeros((cfg.MODEL.num_classes+1)).cuda()
                pixel_sum = torch.zeros((cfg.MODEL.num_classes+1)).cuda()
                print('start evaluation.')
                counter = 1
                for eval_step in range(len(eval_iter)):
                    eval_data = next(eval_iter)
                    res = model(eval_data['imgs'].cuda())
                    if isinstance(res, list):
                        res = res[0]
                    acc_sum, pixel_sum = compute_acc(res, eval_data['targets'].cuda(), acc_sum, pixel_sum)
                    _, mask = torch.max(res, dim=1)
                    cv2.imwrite('./data/vis/' + str(counter) + '.png', np.hstack((eval_data['org_imgs'][0].numpy().transpose(1,2,0),
                                to_color(mask[0].cpu().detach().numpy()))))
                    counter += 1
                acc_value = []
                for i in range(res.shape[1]):
                    acc_value.append((acc_sum[i].float()+1e-10)/(pixel_sum[i].float()+1e-10))
                print(acc_sum)
                print(pixel_sum)
                acc_class = sum(acc_value)/len(acc_value)
                acc_total = (acc_sum[-1].float()+1e-10)/(pixel_sum[-1].float()+1e-10)
                print('iter', iteration,
                         'eval_class_acc: %.2f'%(acc_class.item()*100),
                         'eval_overall_acc: %.2f'%(acc_total.item()*100)
                         )
                model.train()


            elif iteration % cfg.TRAIN.print_freq == 0 and iteration != 0:
                if isinstance(res, list):
                    res = res[0]
                acc_sum, pixel_sum = compute_acc(res, data['targets'].cuda())
                acc_value = []
                for i in range(res.shape[1]):
                    acc_value.append((acc_sum[i].float()+1e-10)/(pixel_sum[i].float()+1e-10))
                    
                acc_class = sum(acc_value)/len(acc_value)
                acc_total = (acc_sum[-1].float()+1e-10)/(pixel_sum[-1].float()+1e-10)
                print('iter', iteration, 'train loss: %.4f'%(loss.item()),
                         'lr: %.5f'%(enc_optimizer.param_groups[0]['lr']),
                         'class_acc: %.2f'%(acc_class.item()*100),
                         'overall_acc: %.2f'%(acc_total.item()*100)
                         )
            iteration += 1

        


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
    train(exp_cfg)