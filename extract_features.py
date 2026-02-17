import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from data.augment import build_eval_transform
from data.dataloader import build_eval_loader
from models.barlow import BarlowTwins
from utils.checkpoint import load_checkpoint


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--out", type=str, default="./features.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = BarlowTwins().to(device)
    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    assert ckpt is not None, "Checkpoint not found."
    model.load_state_dict(ckpt["model"])
    model.eval()

    # use backbone only for features
    backbone = model.backbone
    backbone.eval()

    loader = build_eval_loader(args.data_dir, build_eval_transform(args.img_size), args.batch_size, args.num_workers, split=args.split)

    feats, labels = [], []
    for x, y in tqdm(loader):
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        f = backbone(x)  # [B, 2048]
        feats.append(f.cpu())
        labels.append(y.cpu())

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"feats": feats, "labels": labels}, args.out)
    print(f"Saved {feats.shape} features to {args.out}")


if __name__ == "__main__":
    main()
