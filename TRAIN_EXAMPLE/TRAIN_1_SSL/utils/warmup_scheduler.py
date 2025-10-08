import torch
import math

class WarmupLearningRateScheduler:
    def __init__(self, optimizers, warmup_steps, target_lrs, total_steps):
        self.optimizers = optimizers
        self.warmup_steps = warmup_steps
        self.target_lrs = target_lrs
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        if self.current_step < self.warmup_steps:
            # Linear warm-up
            for i, optimizer in enumerate(self.optimizers):
                lr = self.target_lrs[i] * (self.current_step / self.warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        else:
            # After warm-up, you can apply a decay strategy
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            for i, optimizer in enumerate(self.optimizers):
                lr = self.target_lrs[i] * (1.0 - progress)  # Linear decay as an example
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        self.current_step += 1


def set_warmup_and_decay_lr(parameters, init_lr, peak_lr, final_lr, warm_up_step=30000, decay_factor=0.95, weight_decay=1e-5, amsgrad=False):
    optimizer = torch.optim.Adam(parameters, lr=init_lr, weight_decay=weight_decay, amsgrad=amsgrad)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: (1 + step * (peak_lr / init_lr - 1) / warm_up_step) if step < warm_up_step \
                                                                                else max(decay_factor ** (step - warm_up_step) * peak_lr / init_lr, final_lr / init_lr))
    return optimizer, scheduler