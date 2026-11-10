import os
from pathlib import Path
import torch
from torchvision.io import ImageReadMode
import torchvision.io as io
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

def getImgMaskPairs(images_dir, mask_dir):

    image_dir_path = Path(images_dir)
    mask_dir_path = Path(mask_dir)

    image_names = sorted(os.listdir(image_dir_path))
    mask_names = sorted(os.listdir(mask_dir_path))

    if len(image_names) != len(mask_names):
        raise ValueError(
            f"QUANTIDADE DE IMAGENS É DIFERENTE DAS MÁSCARAS: {len(image_names)} IMAGENS E {len(mask_names)} MÁSCARAS."
        )

    image_paths = []
    mask_paths = []

    for image_name, mask_name in zip(image_names, mask_names):
        image_stem = Path(image_name)
        mask_stem = Path(mask_name)

        if image_stem != mask_stem:
            print(f"Par não coincide: Imagem {image_stem} e Máscara {mask_stem}")

    full_image_path = str(image_dir_path / image_name)
    full_mask_path = str(mask_dir_path / mask_name)

    image_paths.append(full_image_path)
    mask_paths.append(full_mask_path)

    return image_paths, mask_paths

if __name__ == "__main__":
    train_image_dir = "data/train/images"
    train_mask_dir = "data/train/masks"

    image_paths, mask_paths = getImgMaskPairs(train_image_dir, train_mask_dir)
    print(f"Total de pares: {len(train_image_dir)}")

def load_image_mask(image_path, mask_path):
    image_tensor = io.read_image(image_path, mode=ImageReadMode.RGB)
    mask_tensor = io.read_image(mask_path, mode=ImageReadMode.GRAYSCALE)

    binary_mask = (mask_tensor > 0).to(torch.float32)

    return image_tensor, binary_mask

if __name__ == "__main__":
    img_exaple_path = "data/train/images/7-9_AM_P1C1_Serial.jpg"
    msk_example_path = "data/train/masks/7-9_AM_P1C1_Serial.jpg"

    img_tensor, msk_tensor = load_image_mask(img_exaple_path, msk_example_path)

    print(f"Máscara = {msk_tensor.shape}, {torch.unique(msk_tensor)}")
    print(f"Imagem = {img_tensor.shape}")




    
