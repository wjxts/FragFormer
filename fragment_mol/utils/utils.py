import random 
import torch 
import dgl 
import numpy as np
import os 
import math 

def set_random_seed(seed=22, n_threads=16):
    """Set random seed.

    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(n_threads)
    os.environ['PYTHONHASHSEED'] = str(seed) 


from torch.optim.lr_scheduler import _LRScheduler
class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, warmup_iters=0, last_epoch=-1):
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # self.last_epoch 是 last step的意思
        # if self.last_epoch % 100 == 0:
        #     print(f"last_epoch in WarmUpLR: {self.last_epoch}")
        ret = []
        for base_lr in self.base_lrs:
            linear_increase_lr = base_lr * self.last_epoch / self.warmup_iters
            cos_decay_lr = base_lr * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_iters) / (self.total_iters - self.warmup_iters))) / 2 
            current_lr = linear_increase_lr if self.last_epoch < self.warmup_iters else cos_decay_lr
            ret.append(current_lr)
        # ret = [base_lr * self.last_epoch / self.warmup_iters if self.last_epoch < self.warmup_iters else base_lr * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_iters) / (self.total_iters - self.warmup_iters))) / 2 for base_lr in self.base_lrs]
        return ret