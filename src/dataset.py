import os
from pathlib import Path
import torch
from torchvision.io import ImageReadMode
import torchvision.io as io
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

class PlacasDataset(Dataset):

    def __init__(self, images_dir, masks_dir, img_size = (256, 256)):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size

        image_names = sorted(os.listdir(self.images_dir))
        mask_names = sorted(os.listdir(self.masks_dir))

        if len(image_names) != len(mask_names):
            raise ValueError(
                f"Número de imagens ({len(image_names)}) difere das máscaras ({len(mask_names)})."
            )

        self.image_paths = [
            str(self.images_dir / img) for img in image_names
        ]
        self.mask_paths = [str(self.masks_dir / msk) for msk in mask_names]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = io.read_image(img_path, mode=ImageReadMode.RGB)
        mask = io.read_image(mask_path, mode=ImageReadMode.GRAY)

        image = F.resize(image, self.img_size, InterpolationMode.BILINEAR)
        mask = F.resize(mask, self.img_size, InterpolationMode.NEAREST)

        image = image.to(torch.float32) / 255
        mask = (mask > 0).to(torch.float32)

        return image, mask
