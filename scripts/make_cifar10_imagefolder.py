import os
import tarfile
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


CIFAR10_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _load_pickle_from_tar(tar: tarfile.TarFile, member_name: str) -> Dict:
    member = tar.getmember(member_name)
    f = tar.extractfile(member)
    assert f is not None, f"Failed to extract {member_name}"
    # CIFAR-10 pickles are latin1-encoded
    return pickle.load(f, encoding="latin1")


def _cifar_record_to_images(data: np.ndarray) -> np.ndarray:
    """
    data: [N, 3072] uint8
    returns: [N, 32, 32, 3] uint8
    """
    n = data.shape[0]
    x = data.reshape(n, 3, 32, 32).transpose(0, 2, 3, 1)
    return x


def _write_split(
    images: np.ndarray,
    labels: List[int],
    out_root: Path,
    split: str,
):
    for cls in CIFAR10_LABELS:
        (out_root / split / cls).mkdir(parents=True, exist_ok=True)

    counters = {i: 0 for i in range(10)}
    for idx in range(images.shape[0]):
        y = int(labels[idx])
        cls = CIFAR10_LABELS[y]
        counters[y] += 1
        fn = out_root / split / cls / f"{cls}_{counters[y]:05d}.png"
        Image.fromarray(images[idx]).save(fn)


def main(
    tar_gz_path: str = "./cifar-10-python.tar.gz",
    out_dir: str = "./cifar10_imagefolder",
):
    tar_gz_path = str(tar_gz_path)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_gz_path, "r:gz") as tar:
        # Figure out the base folder name inside tar (usually "cifar-10-batches-py")
        members = [m.name for m in tar.getmembers()]
        base = None
        for m in members:
            if m.endswith("batches.meta"):
                base = m.rsplit("/", 1)[0]
                break
        if base is None:
            raise RuntimeError("Could not find CIFAR-10 batches.meta in tarball.")

        # Train batches: data_batch_1..5
        train_imgs_list = []
        train_labels_list = []
        for k in range(1, 6):
            d = _load_pickle_from_tar(tar, f"{base}/data_batch_{k}")
            train_imgs_list.append(d["data"])
            train_labels_list.extend(d["labels"])

        train_data = np.concatenate(train_imgs_list, axis=0).astype(np.uint8)
        train_images = _cifar_record_to_images(train_data)

        # Test batch -> we will call it "val" to match your code
        test = _load_pickle_from_tar(tar, f"{base}/test_batch")
        val_data = test["data"].astype(np.uint8)
        val_images = _cifar_record_to_images(val_data)
        val_labels = test["labels"]

    print(f"Train images: {train_images.shape}, labels: {len(train_labels_list)}")
    print(f"Val images:   {val_images.shape}, labels: {len(val_labels)}")
    print(f"Writing to: {out_root.resolve()}")

    _write_split(train_images, train_labels_list, out_root, "train")
    _write_split(val_images, val_labels, out_root, "val")

    print("Done.")
    print("Example structure:")
    print(out_root / "train" / CIFAR10_LABELS[0])
    print(out_root / "val" / CIFAR10_LABELS[0])


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--tar", type=str, default="./cifar-10-python.tar.gz",
                   help="Path to cifar-10-python.tar.gz")
    p.add_argument("--out", type=str, default="./cifar10_imagefolder",
                   help="Output directory in ImageFolder format")
    args = p.parse_args()

    main(args.tar, args.out)
