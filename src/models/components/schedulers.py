import torch
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, max_lr, min_lr=0.0, last_epoch=-1):
        """
        Scheduler with linear warmup followed by cosine annealing decay.
        
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_steps (int): Number of warmup steps.
            max_steps (int): Total number of training steps.
            max_lr (float): Peak learning rate reached after warmup.
            min_lr (float): Minimum learning rate after cosine annealing.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.init_lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"warmup_steps={warmup_steps}, max_steps={max_steps}, min_lr={min_lr}, max_lr={max_lr}")

        # Internal cosine scheduler for post-warmup phase
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_steps - warmup_steps,
            eta_min=min_lr
        )

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        # Linear warmup
        if step <= self.warmup_steps:
            scale = step / self.warmup_steps
            return [
                base_lr + scale * (self.max_lr - base_lr)
                for base_lr in self.init_lrs
            ]
        else:
            # Delegate to cosine scheduler for post-warmup phase
            return [group['lr'] for group in self.cosine_scheduler.optimizer.param_groups]

    def step(self, epoch=None):
        """Advance one step and update learning rates."""
        self.last_epoch += 1
        step = self.last_epoch

        if step <= self.warmup_steps:
            scale = step / self.warmup_steps
            lrs = []
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.init_lrs[i] + scale * (self.max_lr - self.init_lrs[i])
                param_group['lr'] = lr
                lrs.append(lr)
            self._last_lr = lrs
        else:
            self.cosine_scheduler.step()
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

        return self._last_lr
