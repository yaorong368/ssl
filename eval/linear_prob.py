import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import SGD
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils.distributed import get_world_size, is_main_process


@torch.no_grad()
def accuracy(logits, targets, topk=(1,)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        res.append(correct[:k].reshape(-1).float().sum().mul_(100.0 / targets.size(0)))
    return res


def cosine_lr(base_lr, step, max_steps, warmup_steps=0):
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def extract_backbone_features(backbone, loader, device):
    backbone.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        f = backbone(x)
        feats.append(f.float().cpu())
        labels.append(y.cpu())
    return torch.cat(feats, 0), torch.cat(labels, 0)


def linear_probe_eval(
    backbone,
    train_loader,
    val_loader,
    num_classes: int,
    device,
    epochs: int = 10,
    lr: float = 0.1,
    weight_decay: float = 0.0,
    amp: bool = True,
):
    """
    Linear eval (frozen backbone).
    Returns: dict with top1, top5, best_top1
    """

    # Freeze backbone
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # Infer feat dim using one batch
    x0, _ = next(iter(train_loader))
    x0 = x0.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        feat_dim = backbone(x0).shape[1]

    head = nn.Linear(feat_dim, num_classes).to(device)

    # DDP-safe
    is_distributed = dist.is_available() and dist.is_initialized()
    if is_distributed and get_world_size() > 1:
        head = nn.parallel.DistributedDataParallel(head, device_ids=[device.index], broadcast_buffers=False)

    opt = SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scaler = GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss().to(device)

    steps_per_epoch = len(train_loader)
    max_steps = steps_per_epoch * epochs
    warmup_steps = int(0.05 * max_steps)
    global_step = 0

    best_top1 = 0.0

    for ep in range(epochs):
        head.train()
        if is_main_process():
            pbar = tqdm(total=steps_per_epoch, desc=f"linear probe ep {ep}/{epochs-1}")
        else:
            pbar = None

        for x, y in train_loader:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            lr_now = cosine_lr(lr, global_step, max_steps, warmup_steps)
            for pg in opt.param_groups:
                pg["lr"] = lr_now

            with torch.no_grad():
                f = backbone(x)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                logits = head(f)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(loss=float(loss.detach().cpu()), lr=lr_now)

            global_step += 1

        if pbar is not None:
            pbar.close()

        # Validation
        head.eval()
        top1_sum, top5_sum, n_sum = 0.0, 0.0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)

                f = backbone(x)
                logits = head(f)

                top1, top5 = accuracy(logits, y, topk=(1, 5))
                bs = y.size(0)

                top1_sum += float(top1) * bs
                top5_sum += float(top5) * bs
                n_sum += bs

        # DDP reduce
        if is_distributed and get_world_size() > 1:
            t = torch.tensor([top1_sum, top5_sum, n_sum], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            top1_sum, top5_sum, n_sum = float(t[0]), float(t[1]), float(t[2])

        top1 = top1_sum / max(1, n_sum)
        top5 = top5_sum / max(1, n_sum)

        if is_main_process():
            print(f"[Linear Eval] ep={ep} top1={top1:.2f} top5={top5:.2f}")

        best_top1 = max(best_top1, top1)

    return {"top1": top1, "top5": top5, "best_top1": best_top1}
