from sourcecode.model import HRNetW18SmallV2, FloorHead
from sourcecode.dataset import FloorPlanDataset
from sourcecode.configs import make_config, Options
from sourcecode.utils.optim_loss import adjust_learning_rate
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import torch
import os

class FloorPlanModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = HRNetW18SmallV2()
        self.backbone.load_state_dict(torch.load(cfg.MODEL.weights), strict=False)
        self.head = FloorHead(cfg.MODEL.head)
        self.opts = cfg

    def forward(self, x):
        backbone_out = self.backbone(x)
        return self.head(backbone_out, [512, 512])

def train(cfg):
    train_dataset = FloorPlanDataset(cfg.DATA.train_data, cfg)
    eval_dataset = FloorPlanDataset(cfg.DATA.eval_data, cfg)
    optimizer = 0
    train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.TRAIN.batch_size_per_gpu, 
            shuffle=True,
            num_workers=4,
        )

    eval_loader = DataLoader(
            eval_dataset,
            batch_size=1, 
            shuffle=False,
            num_workers=0,
        )
    
    model = FloorPlanModel(cfg)
    model.cuda()
    enc_optimizer = optim.SGD(model.backbone.parameters(), lr=cfg.TRAIN.optimizer.lr, momentum=0.9)
    dec_optimizer = optim.SGD(model.head.parameters(), lr=cfg.TRAIN.optimizer.lr, momentum=0.9)
    loss_func = nn.CrossEntropyLoss(ignore_index=9).cuda()
    num_epoch = cfg.TRAIN.max_iter//len(train_loader)
    iteration = 0
    for epoch in range(num_epoch):
        train_iter = iter(train_loader)
        for step in range(len(train_iter)):
            model.zero_grad()
            adjust_learning_rate(enc_optimizer, iteration, cfg, cfg.TRAIN.enc_lr_factor)
            adjust_learning_rate(dec_optimizer, iteration, cfg, cfg.TRAIN.dec_lr_factor)
            data = next(train_iter)
            res = model(data['imgs'].cuda())
            res = nn.functional.softmax(res, dim=1)
            # print(res.shape, data['targets'].shape)
            loss = loss_func(res, data['targets'].cuda())
            loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            if iteration % cfg.TRAIN.ckpt_freq == 0 and iteration != 0:
                print(iteration, 'saving model')
                torch.save(model.state_dict(), '{}ckpt{}.pth'.format(
                        cfg.FOLDER,
                        iteration))
            if iteration % cfg.TRAIN.print_freq == 0 and iteration != 0:
                print('iter', iteration, 'train loss:', loss.item())
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