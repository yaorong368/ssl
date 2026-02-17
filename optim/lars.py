import torch
from torch.optim.optimizer import Optimizer


class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS).
    Matches common Barlow Twins usage:
    - weight_decay_filter: exclude 1D params (bias/BN) from weight decay
    - lars_adaptation_filter: exclude 1D params from LARS scaling
    """
    def __init__(
        self,
        params,
        lr: float,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        eta: float = 0.001,
        weight_decay_filter: bool = True,
        lars_adaptation_filter: bool = True,
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            mu = group["momentum"]
            eta = group["eta"]
            wdf = group["weight_decay_filter"]
            laf = group["lars_adaptation_filter"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                dp = p.grad

                if wd != 0 and (not wdf or p.ndim != 1):
                    dp = dp.add(p, alpha=wd)

                if not laf or p.ndim != 1:
                    p_norm = torch.norm(p)
                    dp_norm = torch.norm(dp)

                    one = torch.ones_like(p_norm)
                    q = torch.where(
                        p_norm > 0.0,
                        torch.where(dp_norm > 0.0, eta * p_norm / dp_norm, one),
                        one
                    )
                    dp = dp.mul(q)
                    # p_norm = torch.norm(p)
                    # dp_norm = torch.norm(dp)
                    # if p_norm > 0 and dp_norm > 0:
                    #     q = eta * p_norm / (dp_norm + eps)
                    #     dp = dp.mul(q)

                state = self.state[p]
                if "mu" not in state:
                    state["mu"] = torch.zeros_like(p)
                buf = state["mu"]
                buf.mul_(mu).add_(dp)

                p.add_(buf, alpha=-lr)

        return loss

