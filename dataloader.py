import glob
import os
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


random.seed(1143)
IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def populate_train_list(lowlight_images_path):
    train_list = []
    for pattern in IMAGE_PATTERNS:
        train_list.extend(glob.glob(os.path.join(lowlight_images_path, pattern)))
    train_list = sorted(train_list)
    random.shuffle(train_list)
    return train_list


class lowlight_loader(data.Dataset):
    def __init__(self, lowlight_images_path, image_size=256):
        self.train_list = populate_train_list(lowlight_images_path)
        self.size = int(image_size) if image_size else None
        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]
        data_lowlight = Image.open(data_lowlight_path).convert("RGB")

        if self.size is not None:
            data_lowlight = data_lowlight.resize((self.size, self.size), Image.BICUBIC)

        data_lowlight = np.asarray(data_lowlight, dtype=np.float32) / 255.0
        data_lowlight = torch.from_numpy(data_lowlight).float()
        return data_lowlight.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)
