import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Residual Block
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        return x + self.block(x)


# ============================================================
# KLA Image Restoration Network
# ============================================================

class KLAImageRestorationNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=48,
        num_blocks=6
    ):

        super().__init__()

        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        self.head = nn.Conv2d(
            in_channels,
            features,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # Residual feature processing
        # ----------------------------------------------------

        blocks = []

        for _ in range(num_blocks):

            blocks.append(
                ResidualBlock(features)
            )

        self.body = nn.Sequential(*blocks)

        self.body_conv = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # 2x Super Resolution
        # PixelShuffle converts:
        #
        # [B, 48*4, H, W]
        #
        # into:
        #
        # [B, 48, 2H, 2W]
        # ----------------------------------------------------

        self.upsample = nn.Sequential(

            nn.Conv2d(
                features,
                features * 4,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(2),

            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # Final reconstruction
        # ----------------------------------------------------

        self.tail = nn.Conv2d(
            features,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        # Feature extraction
        x1 = self.head(x)

        # Residual body
        body = self.body(x1)

        body = self.body_conv(body)

        # Global residual connection
        features = x1 + body

        # 2x upsampling
        features = self.upsample(features)

        # Final image
        output = self.tail(features)

        return torch.sigmoid(output)


# ============================================================
# Test the model
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # Create model
    model = KLAImageRestorationNet()

    model = model.to(device)

    # Test input
    test_input = torch.randn(
        1,
        1,
        128,
        128,
        device=device
    )

    print()
    print("Input shape:")
    print(test_input.shape)

    # Forward pass
    with torch.no_grad():

        output = model(test_input)

    print()
    print("Output shape:")
    print(output.shape)

    # Number of parameters
    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        "Parameters:",
        f"{parameters:,}"
    )

    print()
    print("MODEL TEST SUCCESSFUL")