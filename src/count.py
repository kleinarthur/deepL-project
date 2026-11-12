import matplotlib.pyplot as plt
from scipy.ndimage import label
import torch
from torch.utils.data import DataLoader

from dataset import PlacasDataset
from model import UNet


def count_colonies():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(
        torch.load("data/results/unet_petri.pth", map_location=device)
    )
    model.eval()

    dataset = PlacasDataset(
        images_dir="data/val/images", masks_dir="data/val/masks"
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    image, mask = next(iter(dataloader))
    image = image.to(device)

    with torch.no_grad():
        pred = model(image)

    pred_bin = (pred[0, 0] > 0.5).cpu().numpy().astype(int)

    labeled_mask, num_features = label(pred_bin)
    print(f"Total de colonias detectadas: {num_features}")

    img_plot = image[0].cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img_plot)
    axes[0].set_title("Imagem Original")
    axes[0].axis("off")

    axes[1].imshow(labeled_mask, cmap="nipy_spectral")
    axes[1].set_title(f"Colonias Identificadas (Total: {num_features})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("data/results/resultado_contagem.png")
    print("Resultado salvo em 'data/results/resultado_contagem.png'.")


if __name__ == "__main__":
    count_colonies()