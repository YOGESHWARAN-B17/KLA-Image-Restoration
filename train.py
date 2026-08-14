import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
from torch.utils.data import DataLoader, random_split

from dataset import KLADataset
from model import KLAImageRestorationNet


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 1
EPOCHS = 50
LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(pred, target):

    mse = F.mse_loss(pred, target)

    if mse.item() == 0:
        return 100.0

    psnr = 10 * torch.log10(
        1.0 / mse
    )

    return psnr.item()


# ============================================================
# Gradient loss
# ============================================================

def gradient_loss(pred, target):

    pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_y = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_x = F.l1_loss(
        pred_x,
        target_x
    )

    loss_y = F.l1_loss(
        pred_y,
        target_y
    )

    return loss_x + loss_y


# ============================================================
# Combined restoration loss
# ============================================================

def restoration_loss(pred, target):

    # Pixel accuracy
    l1 = F.l1_loss(
        pred,
        target
    )

    # Edge/detail preservation
    grad = gradient_loss(
        pred,
        target
    )

    # Total loss
    loss = l1 + 0.1 * grad

    return loss, l1, grad


# ============================================================
# Dataset
# ============================================================

print("=" * 60)
print("KLA IMAGE RESTORATION TRAINING")
print("=" * 60)

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


dataset = KLADataset()

print()
print("Total dataset samples:", len(dataset))


# ============================================================
# Train / validation split
# ============================================================

if len(dataset) < 2:

    raise RuntimeError(
        "Need at least 2 image pairs for training."
    )


validation_size = max(
    1,
    int(len(dataset) * 0.2)
)

training_size = len(dataset) - validation_size


train_dataset, val_dataset = random_split(
    dataset,
    [training_size, validation_size],
    generator=torch.Generator().manual_seed(42)
)


print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)


# ============================================================
# Data loaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# Model
# ============================================================

model = KLAImageRestorationNet()

model = model.to(DEVICE)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# Mixed precision
# ============================================================

use_amp = torch.cuda.is_available()

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# Training
# ============================================================

best_val_loss = float("inf")


for epoch in range(1, EPOCHS + 1):

    model.train()

    total_train_loss = 0.0

    for degraded, target in train_loader:

        degraded = degraded.to(
            DEVICE,
            non_blocking=True
        )

        target = target.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=use_amp
        ):

            prediction = model(
                degraded
            )

            loss, l1, grad = restoration_loss(
                prediction,
                target
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        total_train_loss += loss.item()


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    total_val_loss = 0.0
    total_psnr = 0.0

    with torch.no_grad():

        for degraded, target in val_loader:

            degraded = degraded.to(
                DEVICE,
                non_blocking=True
            )

            target = target.to(
                DEVICE,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp
            ):

                prediction = model(
                    degraded
                )

                loss, _, _ = restoration_loss(
                    prediction,
                    target
                )

            total_val_loss += loss.item()

            total_psnr += calculate_psnr(
                prediction,
                target
            )


    avg_train_loss = (
        total_train_loss /
        len(train_loader)
    )

    avg_val_loss = (
        total_val_loss /
        len(val_loader)
    )

    avg_psnr = (
        total_psnr /
        len(val_loader)
    )


    # ========================================================
    # Print results
    # ========================================================

    print(
        f"Epoch [{epoch:03d}/{EPOCHS}] "
        f"Train Loss: {avg_train_loss:.6f} "
        f"Val Loss: {avg_val_loss:.6f} "
        f"PSNR: {avg_psnr:.2f} dB"
    )


    # ========================================================
    # Save latest model
    # ========================================================

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "psnr": avg_psnr
        },
        os.path.join(
            CHECKPOINT_DIR,
            "latest.pt"
        )
    )


    # ========================================================
    # Save best model
    # ========================================================

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        torch.save(
            model.state_dict(),
            os.path.join(
                CHECKPOINT_DIR,
                "best_model.pt"
            )
        )

        print(
            "  ★ Best model saved!"
        )


print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "Best model:",
    "checkpoints/best_model.pt"
)