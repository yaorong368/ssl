import os
import torch
from utils.distributed import is_main_process


def save_checkpoint(path: str, state: dict):
    if not is_main_process():
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location="cpu"):
    if not os.path.isfile(path):
        return None
    return torch.load(path, map_location=map_location)
