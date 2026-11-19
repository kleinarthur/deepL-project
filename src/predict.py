import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataset import PlacasDataset
from model import UNet


def calculate_metrics(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    total = pred_bin.sum() + target.sum()
    dice = (2.0 * intersection + 1e-6) / (total + 1e-6)
    iou = (intersection + 1e-6) / (
        pred_bin.sum() + target.sum() - intersection + 1e-6
    )
    return dice.item(), iou.item()


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(
        torch.load("data/results/unet_petri.pth", map_location=device)
    )
    model.eval()

    dataset = PlacasDataset(
        images_dir="data/val/images", masks_dir="data/val/masks"
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    output_dir = "data/results/predicoes"
    os.makedirs(output_dir, exist_ok=True)

    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for idx, (image, mask) in enumerate(dataloader):
            image = image.to(device)
            mask = mask.to(device)

            pred = model(image)

            dice, iou = calculate_metrics(pred, mask)
            total_dice += dice
            total_iou += iou
            print(
                f"Imagem [{idx + 1}/{len(dataloader)}] - Dice: {dice:.4f} | IoU: {iou:.4f}"
            )

            img_plot = image[0].cpu().permute(1, 2, 0).numpy()
            gt_plot = mask[0, 0].cpu().numpy()
            pred_plot = (pred[0, 0] > 0.5).cpu().float().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_plot)
            axes[0].set_title("Imagem Original")
            axes[0].axis("off")

            axes[1].imshow(gt_plot, cmap="gray")
            axes[1].set_title("Mascara Real")
            axes[1].axis("off")

            axes[2].imshow(pred_plot, cmap="gray")
            axes[2].set_title(f"Predicao UNet (Dice: {dice:.2f})")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(f"{output_dir}/resultado_{idx + 1}.png")
            plt.close(fig) 

    num_images = len(dataloader)
    print("-" * 50)
    print(
        f"Media Geral ({num_images} imagens) - Dice: {total_dice / num_images:.4f} | IoU: {total_iou / num_images:.4f}"
    )
    print(f"Todas as predicoes foram salvas em '{output_dir}/'.")


if __name__ == "__main__":
    evaluate()