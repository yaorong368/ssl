import os
import time
import math
import argparse

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from data.augment import build_train_transform, build_eval_transform
from data.dataloader import build_ssl_loader
from models.barlow import my_models
from optim.lars import LARS
from utils.distributed import ddp_setup, is_main_process, get_world_size
from utils.checkpoint import save_checkpoint, load_checkpoint

# Optional: linear eval hook (only runs on main process)
from eval.linear_prob import linear_probe_eval
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def adjust_learning_rate(args, optimizer, steps_per_epoch, step):
    max_steps = args.epochs * steps_per_epoch
    warmup_steps = 10 * steps_per_epoch
    base_lr = args.batch_size / 256.0  # args.batch_size is GLOBAL

    if step < warmup_steps:
        lr = base_lr * step / float(max(1, warmup_steps))
    else:
        s = step - warmup_steps
        ms = max_steps - warmup_steps
        q = 0.5 * (1.0 + math.cos(math.pi * s / float(max(1, ms))))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1.0 - q)

    optimizer.param_groups[0]["lr"] = lr * args.learning_rate_weights
    optimizer.param_groups[1]["lr"] = lr * args.learning_rate_biases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./runs/")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--method", type=str, default='barlow', help="barlw; jdrx;")

    # IMPORTANT: batch_size is GLOBAL (split across GPUs if DDP)
    p.add_argument("--batch_size", type=int, default=2048, help="GLOBAL batch size (split across GPUs)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--num_subsets", type=int, default=4, help="number of subset for jdrx")

    # Barlow-specific
    p.add_argument("--lambd", type=float, default=5e-3)
    p.add_argument("--projector", type=str, default="8192-8192-8192")
    p.add_argument("--learning_rate_weights", type=float, default=0.2)
    p.add_argument("--learning_rate_biases", type=float, default=0.0048)

    # misc
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume", type=str, default="")
    # p.add_argument("--save_every", type=int, default=10)  # no longer used
    # p.add_argument("--seed", type=int, default=1)

    # linear eval during training (monitoring)
    p.add_argument("--linear_eval_every", type=int, default=10)
    p.add_argument("--linear_eval_epochs", type=int, default=10)
    p.add_argument("--linear_eval_lr", type=float, default=0.1)
    p.add_argument("--linear_eval_batch_size", type=int, default=256)

    # logging
    p.add_argument("--log_name", type=str, default="train.log")

    args = p.parse_args()

    torch.backends.cudnn.benchmark = True

    # -------------------------
    # NEW: build a run-specific output dir name
    # methodname + batchsize + feature_dim + epochs
    # -------------------------
    method = args.method
    if method =='jdrx':
        feat_dim = int(args.projector.split("-")[-1])  # last projector dim
        run_name = f"{method}_bs{args.batch_size}_dim{feat_dim}_ep{args.epochs}_subsize{args.num_subsets}"
        args.out_dir = os.path.join(args.out_dir, run_name)
    else:
        feat_dim = int(args.projector.split("-")[-1])  # last projector dim
        run_name = f"{method}_bs{args.batch_size}_dim{feat_dim}_ep{args.epochs}"
        args.out_dir = os.path.join(args.out_dir, run_name)

    # Ensure output dir exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Simple file logger (main process only)
    log_path = os.path.join(args.out_dir, args.log_name)
    # Always overwrite this file each epoch
    ckpt_path_latest = os.path.join(args.out_dir, "ckpt_latest.pt")


    def _log(msg: str):
        if not is_main_process():
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    _log("===== START TRAINING =====")
    _log("Command: " + " ".join(os.sys.argv))
    _log(f"Args: {vars(args)}")

    # DDP setup (torchrun)
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    local_rank = 0
    if is_distributed:
        _, _, local_rank = ddp_setup("nccl")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    _log(f"Distributed={is_distributed} world_size={get_world_size()} local_rank={local_rank} device={device}")

    # GLOBAL batch -> per-GPU batch
    world_size = get_world_size()
    if args.batch_size % world_size != 0:
        raise ValueError(f"GLOBAL batch_size={args.batch_size} must be divisible by world_size={world_size}")
    per_gpu_batch = args.batch_size // world_size
    _log(f"Global batch={args.batch_size} per_gpu_batch={per_gpu_batch}")

    # SSL loader (train only)
    transform = build_train_transform(args.img_size)
    loader, sampler = build_ssl_loader(
        data_dir=args.data_dir,
        transform=transform,
        batch_size=per_gpu_batch,
        num_workers=args.num_workers,
        is_distributed=is_distributed,
    )
    _log(f"Steps per epoch (len(loader)) = {len(loader)}")

    # Model: resnet18 + projector string + BN/all_reduce loss
    model = my_models(projector=args.projector, lambd=args.lambd, objective=args.method, num_subsets=args.num_subsets).to(device)

    if is_distributed and world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False
        )
        _log("Wrapped model with SyncBN + DDP (broadcast_buffers=False)")
    else:
        _log("Running non-DDP (single process)")

    # LARS with two param groups: weights vs bias/BN (p.ndim==1)
    m_for_params = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    param_weights, param_biases = [], []
    for p_ in m_for_params.parameters():
        if p_.ndim == 1:
            param_biases.append(p_)
        else:
            param_weights.append(p_)
    parameters = [{"params": param_weights}, {"params": param_biases}]

    opt = LARS(
        parameters,
        lr=0.0,
        weight_decay=args.weight_decay,
        weight_decay_filter=True,
        lars_adaptation_filter=True,
    )

    scaler = GradScaler(enabled=args.amp)
    _log(f"AMP enabled = {bool(args.amp)}")

    start_epoch = 0
    global_step = 0

    # Resume
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location="cpu")
        if ckpt is not None:
            _log(f"Resuming from {args.resume}")
            if isinstance(model, nn.parallel.DistributedDataParallel):
                model.module.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            scaler.load_state_dict(ckpt.get("scaler", scaler.state_dict()))
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt.get("global_step", 0)
            _log(f"Resumed: start_epoch={start_epoch}, global_step={global_step}")
        else:
            _log(f"WARNING: resume path provided but checkpoint load failed: {args.resume}")

    steps_per_epoch = len(loader)

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()

        if is_main_process():
            pbar = tqdm(total=steps_per_epoch, desc=f"epoch {epoch}/{args.epochs-1}")
        else:
            pbar = None

        t0 = time.time()
        loss_sum = 0.0

        for it, ((x1, x2), _) in enumerate(loader):
            x1 = x1.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            x2 = x2.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)

            adjust_learning_rate(args, opt, steps_per_epoch, global_step)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                loss = model(x1, x2)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_val = float(loss.detach().cpu())
            loss_sum += loss_val

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    loss=loss_val,
                    lr_w=float(opt.param_groups[0]["lr"]),
                    lr_b=float(opt.param_groups[1]["lr"]),
                )

            global_step += 1

        if pbar is not None:
            pbar.close()

        dt = time.time() - t0
        avg_loss = loss_sum / max(1, steps_per_epoch)
        _log(f"Epoch {epoch} done in {dt:.1f}s | avg_loss={avg_loss:.6f} | global_step={global_step}")
        # Save checkpoint EVERY epoch (overwrite previous)
        if is_main_process():
            state = {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "args": vars(args),
            }
            save_checkpoint(ckpt_path_latest, state)
            _log(f"Saved checkpoint (overwrite): {ckpt_path_latest}")


        # # Linear evaluation hook every N epochs (monitoring only; runs on main rank)
        # do_linear = (args.linear_eval_every > 0) and ((epoch + 1) % args.linear_eval_every == 0)
        # if do_linear and is_main_process():
        #     _log(f"=== Running linear evaluation at epoch {epoch} ===")

        #     ssl_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        #     backbone = ssl_model.backbone

        #     # NOTE: your original code used build_eval_transform for BOTH train/val.
        #     # Keeping behavior unchanged here; if you want standard lincls-style,
        #     # swap train transform to RandomResizedCrop+HFlip.
        #     train_set = ImageFolder(os.path.join(args.data_dir, "train"), transform=build_eval_transform(args.img_size))
        #     val_set = ImageFolder(os.path.join(args.data_dir, "val"), transform=build_eval_transform(args.img_size))

        #     train_loader_lin = DataLoader(
        #         train_set,
        #         batch_size=args.linear_eval_batch_size,
        #         shuffle=True,
        #         num_workers=args.num_workers,
        #         pin_memory=True,
        #         drop_last=True,
        #         persistent_workers=(args.num_workers > 0),
        #         prefetch_factor=4 if args.num_workers > 0 else None,
        #     )
        #     val_loader_lin = DataLoader(
        #         val_set,
        #         batch_size=args.linear_eval_batch_size,
        #         shuffle=False,
        #         num_workers=args.num_workers,
        #         pin_memory=True,
        #         drop_last=False,
        #         persistent_workers=(args.num_workers > 0),
        #         prefetch_factor=4 if args.num_workers > 0 else None,
        #     )

        #     metrics = linear_probe_eval(
        #         backbone=backbone,
        #         train_loader=train_loader_lin,
        #         val_loader=val_loader_lin,
        #         num_classes=len(train_set.classes),
        #         device=device,
        #         epochs=args.linear_eval_epochs,
        #         lr=args.linear_eval_lr,
        #         weight_decay=0.0,
        #         amp=args.amp,
        #     )
        #     _log(f"=== Linear eval done: {metrics} ===")

    _log("===== TRAINING COMPLETE =====")



if __name__ == "__main__":
    main()
