import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _is_dist():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _gather(t: torch.Tensor) -> torch.Tensor:
    if not _is_dist():
        return t
    world = dist.get_world_size()
    rank = dist.get_rank()
    out = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(out, t.detach())  # remote negatives as constants
    out[rank] = t                     # keep local slice with grad
    return torch.cat(out, dim=0)


class SimCLR(nn.Module):
    def __init__(self, projector="2048-2048-128", temperature=0.2, backbone="resnet18"):
        super().__init__()
        self.tau = temperature

        if backbone == "resnet18":
            self.backbone = torchvision.models.resnet18(zero_init_residual=True)
            # feat_dim = self.backbone.fc.in_features
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()
            self.backbone.fc = nn.Identity()
        elif backbone == "resnet50":
            self.backbone = torchvision.models.resnet50(weights=None)
            # feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError("backbone must be resnet18 or resnet50")

        sizes = [512] + list(map(int, projector.split("-")))
        layers = []
        for i in range(len(sizes) - 2):
            layers += [nn.Linear(sizes[i], sizes[i + 1], bias=False),
                       nn.BatchNorm1d(sizes[i + 1]),
                       nn.ReLU(inplace=True)]
        layers += [nn.Linear(sizes[-2], sizes[-1], bias=False)]
        self.projector = nn.Sequential(*layers)

    def forward(self, x1, x2):
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))
        return self.info_nce_loss(z1, z2)

    def info_nce_loss(self, z1, z2):
        z1g = _gather(z1)
        z2g = _gather(z2)

        z = torch.cat([z1g, z2g], dim=0)  # [2N, D]
        N = z1g.size(0)

        z = F.normalize(z, dim=1)
        sim = (z @ z.T) / self.tau
        
     
        mask = torch.eye(2 * N, device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, torch.finfo(sim.dtype).min)
  
        pos = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)], dim=0).to(sim.device)
        return F.cross_entropy(sim, pos)
