from contextlib import nullcontext

from torch.optim import AdamW as _AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.nn.parallel import DistributedDataParallel as _DistributedDataParallel


class DistributedDataParallel(_DistributedDataParallel):
    def no_sync_except_last(self, is_last=False):
        return nullcontext() if is_last else self.no_sync()


def range_first_last(total):
    if total == 0:
        return
    if total == 1:
        yield True, True
        return
    yield True, False
    for _ in range(total - 2):
        yield False, False
    yield False, True


class AdamW(_AdamW):
    def __init__(self, model, weight_decay, device, **kwargs) -> None:
        super().__init__(params=[
                {"params": [p for p in model.parameters() if p.requires_grad and p.dim() >= 2], "weight_decay": weight_decay},
                {"params": [p for p in model.parameters() if p.requires_grad and p.dim() < 2], "weight_decay": 0.0}
            ],
            fused=device.type=="cuda",
            **kwargs,
        )


class CosineAnnealingWithWarmupLR(SequentialLR):
    def __init__(self, optimizer, warmup_steps, lr_decay_steps, min_lr) -> None:
        super().__init__(
            optimizer=optimizer,
            schedulers=[
                LinearLR(optimizer=optimizer, start_factor=(1/(warmup_steps+1)), end_factor=1, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer=optimizer, T_max=(lr_decay_steps - warmup_steps), eta_min=min_lr),
            ],
            milestones=[warmup_steps],
        )


def estimate_transformer_mfu(num_params, cfg, fwdbwd_per_iter, dt):
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
    flops_per_token = 6 * num_params + 12 * L * H * Q * T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
    mfu = flops_per_iter / (flops_promised * dt)
    return mfu
