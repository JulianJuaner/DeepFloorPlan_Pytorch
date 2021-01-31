
def adjust_learning_rate(optimizer, cur_iter, cfg, factor = 1.0):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iter) ** cfg.TRAIN.lr_scheduler.power)
    adjust_lr = cfg.TRAIN.optimizer.lr * scale_running_lr
    if cur_iter < cfg.TRAIN.lr_scheduler.warmup_iter:
        adjust_lr = cur_iter/cfg.TRAIN.lr_scheduler.warmup_iter*cfg.TRAIN.optimizer.lr
    for param_group in optimizer.param_groups:  
        param_group['lr'] = adjust_lr*factor