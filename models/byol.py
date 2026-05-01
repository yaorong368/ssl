import copy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


def _parse_mlp_spec(spec: str) -> List[int]:
    parts = [int(x) for x in spec.split("-") if x.strip() != ""]
    if len(parts) < 1:
        raise ValueError(f"Invalid projector spec: {spec}")
    return parts


def _make_mlp(
    in_dim: int,
    dims: List[int],
    use_bn: bool = True,
    last_bn: bool = False
) -> nn.Sequential:
    layers = []
    prev = in_dim
    for i, d in enumerate(dims):
        is_last = (i == len(dims) - 1)

        layers.append(
            nn.Linear(
                prev,
                d,
                bias=False if (use_bn or (last_bn and is_last)) else True
            )
        )

        if use_bn and (not is_last):
            layers.append(nn.BatchNorm1d(d))
            layers.append(nn.ReLU(inplace=True))
        elif is_last and last_bn:
            layers.append(nn.BatchNorm1d(d, affine=False))

        prev = d

    return nn.Sequential(*layers)


class _BackboneWithHead(nn.Module):
    def __init__(self, backbone: str = "resnet18", is_cifar: bool = True):
        super().__init__()

        if backbone == "resnet18":
            net = tvm.resnet18(weights=None)
        elif backbone == "resnet50":
            net = tvm.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # --- CIFAR-10 MODIFICATION ---
        if is_cifar:
            net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        # -----------------------------

        feat_dim = net.fc.in_features
        net.fc = nn.Identity()

        self.backbone = net
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class BYOL(nn.Module):
    def __init__(
        self,
        projector: str = "2048-2048-256",
        backbone: str = "resnet18",
        m: float = 0.996,
        pred_hidden_dim: int = 1024,
    ):
        super().__init__()

        self.m = float(m)

        # online network
        self.backbone = _BackboneWithHead(backbone=backbone)
        proj_dims = _parse_mlp_spec(projector)
        z_dim = proj_dims[-1]

        self.projector = _make_mlp(
            self.backbone.feat_dim,
            proj_dims,
            use_bn=True,
            last_bn=True,
        )

        self.predictor = nn.Sequential(
            nn.Linear(z_dim, pred_hidden_dim, bias=False),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden_dim, z_dim, bias=True),
        )

        # target network (EMA)
        self.target_backbone = copy.deepcopy(self.backbone)
        self.target_projector = copy.deepcopy(self.projector)

        for p in self.target_backbone.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_moving_average(self, current_m: float):
        for p_o, p_t in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            p_t.data.mul_(current_m).add_(p_o.data, alpha=1.0 - current_m)

        for p_o, p_t in zip(self.projector.parameters(), self.target_projector.parameters()):
            p_t.data.mul_(current_m).add_(p_o.data, alpha=1.0 - current_m)

    def _loss(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return (2.0 - 2.0 * (p * z).sum(dim=1)).mean()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # online
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)

        z1 = self.projector(f1)
        z2 = self.projector(f2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # target (stop-grad)
        with torch.no_grad():
            t1 = self.target_backbone(x1)
            t2 = self.target_backbone(x2)

            z1_t = self.target_projector(t1)
            z2_t = self.target_projector(t2)

        loss = self._loss(p1, z2_t.detach()) + self._loss(p2, z1_t.detach())
        return loss