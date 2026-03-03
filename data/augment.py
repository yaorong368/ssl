import random
from PIL import Image, ImageFilter, ImageOps

import torchvision.transforms as transforms


class GaussianBlur(object):
    """
    SimCLR-style Gaussian blur.
    For CIFAR-10 (32x32), blur is often disabled or very small probability.
    """
    def __init__(self, p: float = 0.0, sigma_min: float = 0.1, sigma_max: float = 2.0):
        self.p = float(p)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            sigma = random.uniform(self.sigma_min, self.sigma_max)
            return img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class Solarization(object):
    """
    Usually OFF for SimCLR baseline (kept here for compatibility).
    """
    def __init__(self, p: float = 0.0):
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img

# for cifar10 and cifar100---------------------

class Transform:
    """
    CIFAR-10 self-supervised baseline (SimCLR-style):
      - RandomResizedCrop(scale=(0.2, 1.0))
      - RandomHorizontalFlip(0.5)
      - ColorJitter(0.8,0.8,0.8,0.2) applied with p=0.8
      - RandomGrayscale(p=0.2)
      - GaussianBlur OFF by default for 32x32 (p=0.0)
      - Solarization OFF
      - Normalize (ImageNet stats, kept consistent with your training code)
    Returns (y1, y2).
    """
    def __init__(self, img_size: int = 32):
        self.img_size = int(img_size)

        # SimCLR baseline augmentation params
        cj = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        jitter_p = 0.8
        gray_p = 0.2

        # CIFAR-10: blur often disabled; if you want, set to 0.1
        blur_p_view1 = 0.0
        blur_p_view2 = 0.0

        # Crop: SimCLR-style
        crop = transforms.RandomResizedCrop(
            self.img_size,
            scale=(0.2, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )

        norm = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        self.transform = transforms.Compose([
            crop,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([cj], p=jitter_p),
            transforms.RandomGrayscale(p=gray_p),
            GaussianBlur(p=blur_p_view1, sigma_min=0.1, sigma_max=2.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            norm,
        ])

        self.transform_prime = transforms.Compose([
            crop,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([cj], p=jitter_p),
            transforms.RandomGrayscale(p=gray_p),
            GaussianBlur(p=blur_p_view2, sigma_min=0.1, sigma_max=2.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            norm,
        ])

    def __call__(self, x: Image.Image):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


def build_train_transform(img_size: int = 32):
    # Keeps your old API
    return Transform(img_size=img_size)


def build_eval_transform(img_size: int = 32):
    # Standard eval for CIFAR-10: no resize/crop, just tensor + normalize
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])



## for imagenet-------------

class Transform_imagenet:
    """
    ImageNet / 224x224 self-supervised augmentation (Barlow/DINO-style two-crop):
      View 1: blur p=1.0, solarize p=0.0
      View 2: blur p=0.1, solarize p=0.2
    Returns (y1, y2).
    """
    def __init__(self, img_size: int = 224):
        self.img_size = int(img_size)

        norm = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        # Note: your pasted code does NOT specify scale=..., so this uses torchvision defaults.
        crop = transforms.RandomResizedCrop(
            self.img_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        )

        jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
        )

        self.transform = transforms.Compose([
            crop,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0, sigma_min=0.1, sigma_max=2.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            norm,
        ])

        self.transform_prime = transforms.Compose([
            crop,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1, sigma_min=0.1, sigma_max=2.0),
            Solarization(p=0.2),
            transforms.ToTensor(),
            norm,
        ])

    def __call__(self, x: Image.Image):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


def build_train_transform_imagenet(img_size: int = 224):
    return Transform_imagenet(img_size=img_size)


def build_eval_transform_imagenet(img_size: int = 224):
    # Standard ImageNet eval: resize -> center crop -> normalize
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

# for stl10 ---------------------

class Transform_stl10:
    """
    STL-10 (96x96) Barlow Twins / DINO-style two-view augmentation:
      View 1: blur p=1.0, solarize p=0.0
      View 2: blur p=0.1, solarize p=0.2

    This is commonly used for Barlow-like cross-correlation objectives and
    works well for STL-10 resolution.
    """
    def __init__(self, img_size: int = 96):
        self.img_size = int(img_size)

        norm = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        crop = transforms.RandomResizedCrop(
            self.img_size,
            scale=(0.2, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )

        # DINO/Barlow-style jitter (a bit milder than SimCLR's 0.8/0.8/0.8/0.2)
        jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
        )

        self.transform = transforms.Compose([
            crop,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0, sigma_min=0.1, sigma_max=2.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            norm,
        ])

        self.transform_prime = transforms.Compose([
            crop,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1, sigma_min=0.1, sigma_max=2.0),
            Solarization(p=0.2),
            transforms.ToTensor(),
            norm,
        ])

    def __call__(self, x: Image.Image):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


def build_train_transform_stl10(img_size: int = 96):
    return Transform_stl10(img_size=img_size)


def build_eval_transform_stl10(img_size: int = 96):
    # Standard eval for STL-10: no augmentation, just tensor + normalize
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])