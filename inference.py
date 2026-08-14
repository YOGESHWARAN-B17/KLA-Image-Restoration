import os
import time
import numpy as np
import torch
from PIL import Image

from model import KLAImageRestorationNet


# ============================================================
# Settings
# ============================================================

MODEL_PATH = "checkpoints/best_model.pt"

INPUT_DIR = "data/train/degraded"

OUTPUT_DIR = "outputs/restored"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Create output folder
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Load model
# ============================================================

print("=" * 60)
print("KLA IMAGE RESTORATION INFERENCE")
print("=" * 60)

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


model = KLAImageRestorationNet()

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    state_dict
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully!")


# ============================================================
# Find images
# ============================================================

files = sorted(
    f
    for f in os.listdir(INPUT_DIR)
    if f.endswith("_degraded.npy")
)

print(
    "Images found:",
    len(files)
)


# ============================================================
# Inference
# ============================================================

with torch.no_grad():

    for filename in files:

        print()
        print("Processing:", filename)

        path = os.path.join(
            INPUT_DIR,
            filename
        )

        # Load numpy image
        image = np.load(
            path
        ).astype(np.float32)

        # Convert to tensor
        tensor = torch.from_numpy(
            image
        ).unsqueeze(0).unsqueeze(0)

        tensor = tensor.to(
            DEVICE
        )

        # Measure inference time
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        # Run model
        restored = model(
            tensor
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        # Remove batch/channel
        restored = restored.squeeze().cpu().numpy()

        # Ensure valid image range
        restored = np.clip(
            restored,
            0.0,
            1.0
        )

        # Convert to 8-bit
        restored_uint8 = (
            restored * 255
        ).round().astype(
            np.uint8
        )

        # Output filename
        output_name = filename.replace(
            "_degraded.npy",
            "_restored.png"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            output_name
        )

        # Save PNG
        Image.fromarray(
            restored_uint8
        ).save(
            output_path
        )

        print(
            "Saved:",
            output_path
        )

        print(
            f"Inference time: {elapsed * 1000:.2f} ms"
        )


print()
print("=" * 60)
print("INFERENCE COMPLETE")
print("=" * 60)

print(
    "Output folder:",
    OUTPUT_DIR
)