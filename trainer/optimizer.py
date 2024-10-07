import torch
from torch.optim import Adam


class ClippedAdam(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, max_grad_norm=None):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.max_grad_norm = max_grad_norm

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.param_groups[0]['params'], self.max_grad_norm)

        super().step(closure)

        return loss
