import os
import random
from glob import glob

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


def list_images(image_root):
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_paths = []
    for pattern in patterns:
        image_paths.extend(glob(os.path.join(image_root, "**", pattern), recursive=True))
    image_paths = sorted(set(image_paths))
    random.shuffle(image_paths)
    return image_paths


class LowLightDataset(data.Dataset):
    def __init__(self, lowlight_images_path, image_size=256, random_flip=True):
        self.image_paths = list_images(lowlight_images_path)
        self.image_size = image_size
        self.random_flip = random_flip
        if not self.image_paths:
            raise RuntimeError("No training images found under {}".format(lowlight_images_path))
        print("Total training examples:", len(self.image_paths))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.image_size and self.image_size > 0:
            image = image.resize((self.image_size, self.image_size), RESAMPLE)
        if self.random_flip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image = np.asarray(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    def __len__(self):
        return len(self.image_paths)
