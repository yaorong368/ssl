import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision

class GlobalBatchNorm1d(nn.Module):
    """
    BatchNorm1d using GLOBAL (all-GPU) batch statistics in training.
    Uses only current-batch stats (no running stats).
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"GlobalBatchNorm1d expects [N,C], got {tuple(x.shape)}")

        x_dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            xf = x.float()  # fp32 for stats + normalize

            n = xf.shape[0]
            mean = xf.mean(dim=0)
            mean2 = (xf * xf).mean(dim=0)

            if _is_dist():
                n_tensor = xf.new_tensor([n], dtype=torch.float32)  # on same device
                sum1 = mean * n_tensor
                sum2 = mean2 * n_tensor

                dist.all_reduce(sum1, op=dist.ReduceOp.SUM)
                dist.all_reduce(sum2, op=dist.ReduceOp.SUM)
                dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM)

                mean = sum1 / n_tensor
                mean2 = sum2 / n_tensor

            var = (mean2 - mean * mean).clamp_min(0.0)
            x_hat = (xf - mean) / torch.sqrt(var + self.eps)

            if self.affine:
                x_hat = x_hat * self.weight.float() + self.bias.float()

        return x_hat.to(x_dtype)

def _subset_standardize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # x: [m, d], standardize over batch dim (m)
    mean = x.mean(dim=0)
    var = x.var(dim=0, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    '''
    remove diagonal return all off-diagonal as flatten()
    '''
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _barlow_core_loss(c: torch.Tensor, lambd: float) -> torch.Tensor:
    """
    c: (d, d) cross-correlation matrix already normalized by global batch.
    """
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag_sum = off_diagonal(c).pow_(2).sum()
    return on_diag + lambd * off_diag_sum

def barlow_loss(z1, z2, bn, lambd):
    world = dist.get_world_size() if _is_dist() else 1
    global_bs = z1.shape[0] * world

    with torch.cuda.amp.autocast(enabled=False):
        z1f = z1.float()
        z2f = z2.float()
        c = bn(z1f).T @ bn(z2f)
        if _is_dist():
            dist.all_reduce(c)
        c = c / float(global_bs)
        loss = _barlow_core_loss(c, lambd)
    return loss
# def barlow_loss(z1: torch.Tensor, z2: torch.Tensor, bn: nn.Module, lambd: float) -> torch.Tensor:
#     """
#     Standard (global) Barlow Twins loss.
#     """
#     world = dist.get_world_size() if _is_dist() else 1
#     global_bs = z1.shape[0] * world

#     c = bn(z1).T @ bn(z2)
#     if _is_dist():
#         dist.all_reduce(c)
#     c = c / float(global_bs)
#     return _barlow_core_loss(c, lambd)


def jdrx_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    bn: nn.Module,
    lambd: float,
    num_subsets: int = 4,
    sync_stats: bool = False,
) -> torch.Tensor:
    """
    JDRX subset-wise Barlow (partition-based):
      - permute local batch once
      - split into num_subsets contiguous subsets (nearly equal size)
      - compute Barlow core loss on each subset
      - average losses

    Notes:
    - Local-batch partitioning (per rank under DDP).
    - Each subset uses denom = subset_size (subset-local normalization).
    """
    b = z1.shape[0]

    k = max(1, int(num_subsets))
    # k = min(k, b)  # cannot have more subsets than samples
    assert k < b

    # One permutation, then split
    perm = torch.randperm(b, device=z1.device)
    z1p = bn(z1[perm])
    z2p = bn(z2[perm])

    # Split sizes as evenly as possible (first r subsets get +1)
    base = b // k
    r = b % k

    losses = []
    start = 0
    for s in range(k):
        size = base + (1 if s < r else 0)
        if size < 2:
            # if batch is tiny and some subsets become size=1, skip them
            start += size
            continue

        end = start + size
        z1s = z1p[start:end]
        z2s = z2p[start:end]
        start = end

        c = _subset_standardize(z1s).T @ _subset_standardize(z2s)   # [d,d]
       
        if sync_stats and _is_dist():
            # 2) sum c across ranks
            dist.all_reduce(c, op=dist.ReduceOp.SUM)

            # 3) also sum subset size across ranks
            sz = torch.tensor([size], device=z1.device, dtype=torch.float32)
            dist.all_reduce(sz, op=dist.ReduceOp.SUM)
            global_size = float(sz.item())
        else:
            global_size = float(size)

        # 4) normalize by GLOBAL subset size
        c = c / global_size

        losses.append(_barlow_core_loss(c, lambd))

    return torch.stack(losses).mean()


class my_models(nn.Module):
    """
    Self-contained model with two objectives:
      - objective="barlow": standard global Barlow Twins
      - objective="jdrx": subset-wise (JDRX-style) Barlow averaged over subsets
    """
    def __init__(
        self,
        projector: str = "512-512-512",
        lambd: float = 5e-3,
        objective: str = "barlow",
        num_subsets: int = 4,
        sync_jdrx_stats: bool = False,
    ):
        super().__init__()
        self.lambd = float(lambd)
        self.objective = str(objective).lower()
        self.num_subsets = int(num_subsets)
        self.sync_jdrx_stats = bool(sync_jdrx_stats)

        self.backbone = torchvision.models.resnet18(zero_init_residual=True)
        # CIFAR-style stem
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        sizes = [512] + list(map(int, projector.split("-")))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1], affine=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for representations
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        if self.objective == "barlow":
            return barlow_loss(z1, z2, self.bn, self.lambd)
        elif self.objective == "jdrx":
            return jdrx_loss(
                z1, z2, self.bn, self.lambd,
                num_subsets=self.num_subsets,
                sync_stats=self.sync_jdrx_stats,   # NEW
            )
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

