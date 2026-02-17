import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
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


def barlow_loss(z1: torch.Tensor, z2: torch.Tensor, bn: nn.BatchNorm1d, lambd: float) -> torch.Tensor:
    """
    Standard (global) Barlow Twins loss.
    """
    world = dist.get_world_size() if _is_dist() else 1
    global_bs = z1.shape[0] * world

    c = bn(z1).T @ bn(z2)
    if _is_dist():
        dist.all_reduce(c)
    c = c / float(global_bs)
    return _barlow_core_loss(c, lambd)


def jdrx_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    bn: nn.BatchNorm1d,
    lambd: float,
    num_subsets: int = 4,
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
    if b < 2:
        return barlow_loss(z1, z2, bn, lambd)

    k = max(1, int(num_subsets))
    k = min(k, b)  # cannot have more subsets than samples

    # One permutation, then split
    perm = torch.randperm(b, device=z1.device)
    z1p = z1[perm]
    z2p = z2[perm]

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

        c = bn(z1s).T @ bn(z2s)
        c = c / float(size)

        losses.append(_barlow_core_loss(c, lambd))

    if len(losses) == 0:
        # fallback if everything got skipped (extremely small b)
        return barlow_loss(z1, z2, bn, lambd)

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
        # subset_size: int | None = None,
    ):
        super().__init__()
        self.lambd = float(lambd)
        self.objective = str(objective).lower()
        self.num_subsets = int(num_subsets)
        # self.subset_size = subset_size if subset_size is None else int(subset_size)

        self.backbone = torchvision.models.resnet18(zero_init_residual=True)
        # CIFAR-style stem
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        sizes = [512] + list(map(int, projector.split("-")))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
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
                # subset_size=self.subset_size,
            )
        else:
            raise ValueError(f"Unknown objective: {self.objective}")


# import torch
# import torch.nn as nn
# import torch.distributed as dist
# import torchvision


# def _is_dist() -> bool:
#     return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


# def off_diagonal(x: torch.Tensor) -> torch.Tensor:
#     n, m = x.shape
#     assert n == m
#     return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# class BarlowTwins(nn.Module):
#     """
#     Reference-style Barlow Twins:
#     - resnet18 backbone
#     - projector defined by a string like "8192-8192-8192"
#     - BN(affine=False) for z normalization
#     - C computed as BN(z1)^T BN(z2), then all_reduce(C), then / global_batch
#     """
#     def __init__(self, projector: str = "512-512-512", lambd: float = 5e-3):
#         super().__init__()
#         self.lambd = lambd

#         self.backbone = torchvision.models.resnet18(zero_init_residual=True)
#         self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.backbone.fc = nn.Identity()
#         self.backbone.maxpool = nn.Identity()

#         sizes = [512] + list(map(int, projector.split("-")))
#         layers = []
#         for i in range(len(sizes) - 2):
#             layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
#             layers.append(nn.BatchNorm1d(sizes[i + 1]))
#             layers.append(nn.ReLU(inplace=True))
#         layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
#         self.projector = nn.Sequential(*layers)

#         self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

#     def forward(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
#         z1 = self.projector(self.backbone(y1))
#         z2 = self.projector(self.backbone(y2))
#         return self.barlow_loss(z1, z2)

#     def barlow_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
#         world = dist.get_world_size() if _is_dist() else 1
#         global_bs = z1.shape[0] * world

#         c = self.bn(z1).T @ self.bn(z2)
#         if _is_dist():
#             dist.all_reduce(c)
#         c = c / float(global_bs)

#         on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
#         off_diag_sum = off_diagonal(c).pow_(2).sum()
#         return on_diag + self.lambd * off_diag_sum
