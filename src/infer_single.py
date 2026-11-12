import sys
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import label
import torch
import torchvision.transforms as T

from model import UNet


def predict_single(image_path, model_path="data/results/unet_petri.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    raw_img = Image.open(image_path).convert("RGB")
    image_tensor = transform(raw_img).unsqueeze(0).to(device)

    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        pred = model(image_tensor)

    pred_bin = (pred[0, 0] > 0.5).cpu().numpy().astype(int)
    labeled_mask, num_features = label(pred_bin)

    print(f"Imagem processada: {image_path}")
    print(f"Total de colonias encontradas: {num_features}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(raw_img.resize((256, 256)))
    axes[0].set_title("Imagem de Entrada")
    axes[0].axis("off")

    axes[1].imshow(pred_bin, cmap="gray")
    axes[1].set_title("Mascara Binarizada")
    axes[1].axis("off")

    axes[2].imshow(labeled_mask, cmap="nipy_spectral")
    axes[2].set_title(f"Contagem Final ({num_features} colonias)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("data/results/resultado_inferencia_unica.png")
    print("Relatorio visual salvo em 'data/results/resultado_inferencia_unica.png'.")


if __name__ == "__main__":
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/val/images/imagem_exemplo.png"
    )
    predict_single(path)