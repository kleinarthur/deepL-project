import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d( #lupa
                in_channels,
                out_channels,
                kernel_size=3, #janela 3x3
                stride=1, #pixel a pixel
                padding=1, #cria moldura pro centro chegar à última camada (se não fica 254**2)
                bias=False, #peso ao output 
            ),
            nn.BatchNorm2d(out_channels), # 
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

        x = self.final_conv(x) # 32 -> 1    
        x = self.sigmoid(x) 

        return x
