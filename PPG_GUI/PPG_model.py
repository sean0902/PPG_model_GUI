import torch
import torch.nn as nn

class DilatedDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DilatedDenoisingAutoencoder, self).__init__()

        # Encoder (使用 Dilated Conv)
        self.encoder1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
#             nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
#             nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
#             nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.encoder5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=16, dilation=16),
            nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Decoder (使用 Transposed Conv)
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
#             nn.ReLU()
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, kernel_size=2, stride=2),  # Skip connection
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
#             nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose1d(128, 32, kernel_size=2, stride=2),  # Skip connection
            nn.BatchNorm1d(32),
#             nn.LeakyReLU(0.2)
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose1d(64, 16, kernel_size=2, stride=2),  # Skip connection
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2)
#             nn.ReLU()
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose1d(32, 1, kernel_size=2, stride=2),  # Final Output
#             nn.Tanh()  # Output in range [-1,1]
            nn.LeakyReLU(0.2)
#             nn.ReLU()
        )

    def forward(self, x):
        x = x.squeeze(2)  # 變為 (batch, 1, 512) 以適應 Conv1d

        # Encoding
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)
        encoded = self.encoder5(skip4)

        # Decoding with skip connections
        decoded5 = self.decoder5(encoded)
        decoded5 = torch.cat((decoded5, skip4), dim=1)

        decoded4 = self.decoder4(decoded5)
        decoded4 = torch.cat((decoded4, skip3), dim=1)

        decoded3 = self.decoder3(decoded4)
        decoded3 = torch.cat((decoded3, skip2), dim=1)

        decoded2 = self.decoder2(decoded3)
        decoded2 = torch.cat((decoded2, skip1), dim=1)

        decoded1 = self.decoder1(decoded2)  # 最終輸出
        decoded1 = decoded1.unsqueeze(2)  # 變回 (batch, 1, 1, 512)

        return decoded1