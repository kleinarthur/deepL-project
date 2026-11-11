from torch.utils.data import DataLoader
from dataset import PlacasDataset
import torch

def get_dataloaders(
    train_img_dir,
    train_mask_dir,
    val_img_dir=None,
    val_mask_dir=None,
    batch_size=4,
    img_size=(256, 256),
    num_workers=2,
):
    train_dataset = PlacasDataset(
        images_dir=train_img_dir,
        masks_dir=train_mask_dir,
        img_size=img_size,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader

if __name__ == "__main__":
    train_image_dir = "data/train/images"
    train_mask_dir = "data/train/masks"

    train_loader = get_dataloaders(
        train_img_dir=train_image_dir,
        train_mask_dir=train_mask_dir,
        batch_size=4,
        img_size=(256, 256),
        num_workers=2,
    )

    print(f"Total de lotes (batches) no treino: {len(train_loader)}")

    images_batch, masks_batch = next(iter(train_loader))


    print(
        f"Lote de Imagens  : {images_batch.shape} -> [Batch Size, Canais, Altura, Largura]"
    )
    print(
        f"Lote de Máscaras : {masks_batch.shape}  -> [Batch Size, Canais, Altura, Largura]"
    )
    
    