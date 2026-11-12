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

    # Usa a pasta de VALIDAÇÃO
    dataset = PlacasDataset(
        images_dir="data/val/images", masks_dir="data/val/masks"
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    image, mask = next(iter(dataloader))
    image = image.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        pred = model(image)

    dice, iou = calculate_metrics(pred, mask)
    print(f"Dice Score: {dice:.4f} | IoU: {iou:.4f}")

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
    plt.savefig("data/results/resultado_inferencia.png")
    print("Resultado salvo em 'data/results/resultado_inferencia.png'.")


if __name__ == "__main__":
    evaluate()