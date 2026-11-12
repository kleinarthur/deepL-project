import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import PlacasDataset
from losses import BCEDiceLoss
from model import UNet


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    dataset = PlacasDataset(
        images_dir="data/train/images", masks_dir="data/train/masks"
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    epochs = 20

    for epoch in range(epochs):
        loss = train_one_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoca [{epoch + 1}/{epochs}] - Loss: {loss:.4f}")

    torch.save(model.state_dict(), "data/results/unet_petri.pth")
    print("Modelo salvo em 'data/results/unet_petri.pth'.")


if __name__ == "__main__":
    train()