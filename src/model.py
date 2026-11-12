import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        #ENCODER

        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)

        self.bottleneck = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #DECODER

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(
            256, 128
        ) 
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(
            128, 64
        ) 
       
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        
        skip1 = self.enc1(x)
        x = self.pool(skip1)

        skip2 = self.enc2(x)
        x = self.pool(skip2)

        skip3 = self.enc3(x)
        x = self.pool(skip3)

        x = self.bottleneck(x)

        x = self.upconv3(x)
        x = torch.cat((skip3, x), dim=1)
        x = self.dec3(x)

        x = self.upconv2(x)
        x = torch.cat((skip2, x), dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = torch.cat((skip1, x), dim=1)
        x = self.dec1(x)

        x = self.final_conv(x)
        x = self.sigmoid(x)

        return x

if __name__ == "__main__":

    y, x = torch.meshgrid(
        torch.arange(256), torch.arange(256), indexing="ij"
    )

    dist_centro = torch.sqrt((x - 128) ** 2 + (y - 128) ** 2)
    placa_mask = dist_centro < 110

    colonia1 = torch.sqrt((x - 90) ** 2 + (y - 100) ** 2) < 18
    colonia2 = torch.sqrt((x - 160) ** 2 + (y - 140) ** 2) < 22
    colonia3 = torch.sqrt((x - 120) ** 2 + (y - 170) ** 2) < 12

    x_dummy = torch.zeros((1, 3, 256, 256))

    x_dummy[0, 0, placa_mask] = 0.4
    x_dummy[0, 1, placa_mask] = 0.1
    x_dummy[0, 2, placa_mask] = 0.1

    for col in [colonia1, colonia2, colonia3]:
        x_dummy[0, 0, col] = 0.95
        x_dummy[0, 1, col] = 0.90
        x_dummy[0, 2, col] = 0.40

    model = UNet(in_channels=3, out_channels=1)
    model.eval()

    with torch.no_grad():
        output = model(x_dummy)

    img_plot = x_dummy[0].permute(1, 2, 0).numpy()
    mask_plot = output[0, 0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img_plot)
    axes[0].set_title("Entrada (Placa com Círculos Sintéticos)")
    axes[0].axis("off")

    axes[1].imshow(mask_plot, cmap="gray")
    axes[1].set_title("Saída da U-Net (Não Treinada)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("resultado_teste.png")
    print("Imagem gerada e salva como 'resultado_teste.png'.")