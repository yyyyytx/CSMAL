#!/usr/bin/python
# -*- encoding: utf-8 -*-
import math

import torch
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
import numpy as np


class WarmupExpLrScheduler(_LRScheduler):
    def __init__(
            self,
            optimizer,
            power,
            step_interval=1,
            warmup_iter=500,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.power = power
        self.step_interval = step_interval
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupExpLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            ratio = self.power ** (real_iter // self.step_interval)
        return ratio

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio


class WarmupPolyLrScheduler(_LRScheduler):
    def __init__(
            self,
            optimizer,
            power,
            max_iter,
            warmup_iter,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.power = power
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupPolyLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            real_max_iter = self.max_iter - self.warmup_iter
            alpha = real_iter / real_max_iter
            ratio = (1 - alpha) ** self.power
        return ratio

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio


class WarmupCosineLrScheduler(_LRScheduler):
    '''
    This is different from official definition, this is implemented according to
    the paper of fix-match
    '''
    def __init__(
            self,
            optimizer,
            max_iter,
            warmup_iter,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupCosineLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            real_max_iter = self.max_iter - self.warmup_iter
            ratio = np.cos((7 * np.pi * real_iter) / (16 * real_max_iter))
        return ratio

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio


# from Fixmatch-pytorch
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        # return max(0., math.cos(math.pi * num_cycles * no_progress))

        return max(0., (math.cos(math.pi * num_cycles * no_progress) + 1) * 0.5)

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

if __name__ == "__main__":
    model = torch.nn.Conv2d(3, 16, 3, 1, 1)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    max_iter = 20000
    # lr_scheduler = WarmupCosineLrScheduler(optim, max_iter, 0)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optim, 0, max_iter)

    lrs = []
    for _ in range(max_iter):
        lr = lr_scheduler.get_lr()[0]
        print(lr)
        lrs.append(lr)
        lr_scheduler.step()
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    lrs = np.array(lrs)
    n_lrs = len(lrs)
    plt.plot(np.arange(n_lrs), lrs)
    plt.title('3')
    plt.grid()
    plt.show()


