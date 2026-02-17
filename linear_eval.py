import os
import argparse
import math
import torch
import time
import torch.nn as nn
import torch.distributed as dist
from torch.optim import SGD
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.augment import build_eval_transform
from models.barlow import my_models
from utils.distributed import ddp_setup, get_world_size, is_main_process
from utils.checkpoint import load_checkpoint, save_checkpoint



def cosine_lr(base_lr, step, max_steps, warmup_steps=0):
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./runs/linear_eval")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--method", type=str, default='barlow', help="barlw; jdrx;")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "linear_eval.log")

    def _log(msg: str):
        if not is_main_process():
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    _log("===== START LINEAR EVAL =====")
    _log("Command: " + " ".join(os.sys.argv))
    _log(f"Args: {vars(args)}")

    torch.backends.cudnn.benchmark = True

    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    local_rank = 0
    if is_distributed:
        _, _, local_rank = ddp_setup("nccl")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    _log(f"Distributed={is_distributed} world_size={get_world_size()} local_rank={local_rank} device={device}")


    train_set = ImageFolder(os.path.join(args.data_dir, "train"), transform=build_eval_transform(args.img_size))
    val_set = ImageFolder(os.path.join(args.data_dir, "val"), transform=build_eval_transform(args.img_size))
    num_classes = len(train_set.classes)

    train_sampler = None
    if is_distributed and get_world_size() > 1:
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=False, persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None
    )

    # Load checkpoint FIRST (so we know projector / lambd used in training)
    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    assert ckpt is not None, "Checkpoint not found."

    # Build SSL model with the SAME projector/lambd as training
    proj = ckpt.get("args", {}).get("projector", "512-512")
    lambd = ckpt.get("args", {}).get("lambd", 5e-3)
    _log(f"Loaded ckpt: {args.ckpt}")
    _log(f"Using projector={proj}, lambd={lambd}")

    ssl = my_models(projector=proj, lambd=lambd, objective=args.method).to(device)
    # ssl = BarlowTwins(projector=proj, lambd=lambd).to(device)
    ssl.load_state_dict(ckpt["model"])
    ssl.eval()

    backbone = ssl.backbone
    for p_ in backbone.parameters():
        p_.requires_grad = False

    # ResNet50 backbone output dim is 2048
    head = nn.Linear(512, num_classes).to(device)

    if is_distributed and get_world_size() > 1:
        head = nn.parallel.DistributedDataParallel(head, device_ids=[local_rank], broadcast_buffers=False)

    opt = SGD(head.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    criterion = nn.CrossEntropyLoss().to(device)

    steps_per_epoch = len(train_loader)
    max_steps = steps_per_epoch * args.epochs
    warmup_steps = int(0.05 * max_steps)
    global_step = 0

    best_top1 = 0.0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        head.train()
        if is_main_process():
            pbar = tqdm(total=steps_per_epoch, desc=f"linear epoch {epoch}/{args.epochs-1}")
        else:
            pbar = None

        for (x, y) in train_loader:
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            lr_now = cosine_lr(args.lr, global_step, max_steps, warmup_steps)
            for pg in opt.param_groups:
                pg["lr"] = lr_now

            with torch.no_grad():
                f = backbone(x)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
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
            for (x, y) in val_loader:
                x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)

                f = backbone(x)
                logits = head(f)

                top1, top5 = accuracy(logits, y, topk=(1, 5))
                bs = y.size(0)

                top1_sum += float(top1) * bs
                top5_sum += float(top5) * bs
                n_sum += bs

        top1 = top1_sum / max(1, n_sum)
        top5 = top5_sum / max(1, n_sum)

        # DDP: average metrics across ranks
        if is_distributed and get_world_size() > 1:
            t = torch.tensor([top1_sum, top5_sum, n_sum], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            top1 = float(t[0].item() / max(1.0, t[2].item()))
            top5 = float(t[1].item() / max(1.0, t[2].item()))

        if is_main_process():
            _log(f"Epoch {epoch}: val top1={top1:.2f} | top5={top5:.2f}")


        if top1 > best_top1:
            best_top1 = top1
            state = {
                "epoch": epoch,
                "head": head.module.state_dict() if isinstance(head, nn.parallel.DistributedDataParallel) else head.state_dict(),
                "best_top1": best_top1,
                "args": vars(args),
            }
            save_checkpoint(os.path.join(args.out_dir, "best_linear.pt"), state)

    if is_main_process():
        _log(f"Best val top1 = {best_top1:.2f}")
        _log("===== LINEAR EVAL COMPLETE =====")



if __name__ == "__main__":
    main()
