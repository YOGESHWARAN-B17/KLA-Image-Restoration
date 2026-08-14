import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import KLAImageRestorationNet


# ============================================================
# Configuration
# ============================================================

TEST_DIR = "data/test/degraded"
OUTPUT_DIR = "outputs/test_restored"
CHECKPOINT = "checkpoints/best_model.pt"

# Create output folder if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("KLA IMAGE RESTORATION - TEST SET INFERENCE")
print("=" * 60)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Check folders and model
# ============================================================

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Test folder not found: {TEST_DIR}"
    )

if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(
        f"Model checkpoint not found: {CHECKPOINT}"
    )


# ============================================================
# Load trained model
# ============================================================

print()
print("Loading trained model...")

model = KLAImageRestorationNet()

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

print("Model loaded:", CHECKPOINT)


# ============================================================
# Find test files
# ============================================================

test_files = sorted(
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith(".npy")
)

print()
print("Test images found:", len(test_files))


if len(test_files) == 0:
    raise RuntimeError(
        "No .npy test images found in " + TEST_DIR
    )


# ============================================================
# Inference
# ============================================================

total_time = 0.0
processed = 0

print()
print("=" * 60)
print("STARTING TEST INFERENCE")
print("=" * 60)


with torch.no_grad():

    for index, filename in enumerate(test_files, 1):

        print()
        print(
            f"[{index}/{len(test_files)}] "
            f"Processing: {filename}"
        )

        # ----------------------------------------------------
        # Load numpy image
        # ----------------------------------------------------

        input_path = os.path.join(
            TEST_DIR,
            filename
        )

        image = np.load(input_path)

        print("    Input shape:", image.shape)

        # ----------------------------------------------------
        # Convert numpy image to tensor
        # ----------------------------------------------------

        image_tensor = torch.from_numpy(
            image
        ).float()

        # If image is H x W
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)

        # If image is C x H x W
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(DEVICE)

        # ----------------------------------------------------
        # Run model
        # ----------------------------------------------------

        start_time = time.time()

        restored = model(image_tensor)

        # Synchronize CUDA before measuring time
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        inference_time = (
            time.time() - start_time
        ) * 1000

        total_time += inference_time

        # ----------------------------------------------------
        # Convert output to numpy
        # ----------------------------------------------------

        restored = (
            restored
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Clip output to valid image range
        # ----------------------------------------------------

        restored = np.clip(
            restored,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Create output filename
        # ----------------------------------------------------

        image_id = os.path.splitext(
            filename
        )[0]

        output_file = os.path.join(
            OUTPUT_DIR,
            image_id + "_restored.png"
        )

        # ----------------------------------------------------
        # Save restored image
        # ----------------------------------------------------

        plt.imsave(
            output_file,
            restored,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        processed += 1

        print(
            f"    Saved: {output_file}"
        )

        print(
            f"    Inference time: "
            f"{inference_time:.2f} ms"
        )


# ============================================================
# Results
# ============================================================

average_time = (
    total_time / processed
    if processed > 0
    else 0
)


print()
print("=" * 60)
print("TEST INFERENCE COMPLETE")
print("=" * 60)

print(
    "Test images found:",
    len(test_files)
)

print(
    "Test images processed:",
    processed
)

print(
    f"Average inference time: "
    f"{average_time:.2f} ms/image"
)

print(
    "Output folder:",
    OUTPUT_DIR
)

print("=" * 60)