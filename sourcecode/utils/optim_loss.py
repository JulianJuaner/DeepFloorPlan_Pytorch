import torch

def adjust_learning_rate(optimizer, cur_iter, cfg, factor = 1.0):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iter) ** cfg.TRAIN.lr_scheduler.power)
    adjust_lr = cfg.TRAIN.optimizer.lr * scale_running_lr
    if cur_iter < cfg.TRAIN.lr_scheduler.warmup_iter:
        adjust_lr = cur_iter/cfg.TRAIN.lr_scheduler.warmup_iter*cfg.TRAIN.optimizer.lr
    for param_group in optimizer.param_groups:  
        param_group['lr'] = adjust_lr*factor
    return adjust_lr*factor

def compute_acc(pred, target, acc_sum=None, pixel_sum=None):
    ignore_index = pred.shape[1]
    _, preds = torch.max(pred, dim=1)
    if acc_sum != None and pixel_sum!= None:
        acc_sum = acc_sum
        pixel_sum = pixel_sum
    else:
        acc_sum = torch.zeros((pred.shape[1]+1)).cuda()
        pixel_sum = torch.zeros((pred.shape[1]+1)).cuda()

    for i in range(pred.shape[1]):
        valid = (target == i).long()
        acc_sum[i] += torch.sum(valid * (preds == i).long())
        pixel_sum[i] += torch.sum(valid)

    acc_sum[-1] = sum(acc_sum[:-1])
    pixel_sum[-1] = sum(pixel_sum[:-1])

    return acc_sum, pixel_sum