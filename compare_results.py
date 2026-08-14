import os
import numpy as np
from PIL import Image, ImageDraw


GT_DIR = "data/train/ground_truth"
DEG_DIR = "data/train/degraded"
REST_DIR = "outputs/restored"

OUT_DIR = "outputs/comparisons"

os.makedirs(OUT_DIR, exist_ok=True)


files = sorted(
    f for f in os.listdir(DEG_DIR)
    if f.endswith("_degraded.npy")
)


for filename in files:

    image_id = filename.replace(
        "_degraded.npy", ""
    )

    deg_path = os.path.join(
        DEG_DIR,
        filename
    )

    gt_path = os.path.join(
        GT_DIR,
        image_id + "_gt.npy"
    )

    rest_path = os.path.join(
        REST_DIR,
        image_id + "_restored.png"
    )

    if not os.path.exists(gt_path):
        continue

    if not os.path.exists(rest_path):
        continue

    degraded = np.load(
        deg_path
    ).astype(np.float32)

    ground_truth = np.load(
        gt_path
    ).astype(np.float32)

    restored = np.array(
        Image.open(rest_path)
    ).astype(np.float32) / 255.0

    # Convert to 8-bit
    degraded = np.clip(
        degraded, 0, 1
    )

    ground_truth = np.clip(
        ground_truth, 0, 1
    )

    degraded = (
        degraded * 255
    ).astype(np.uint8)

    ground_truth = (
        ground_truth * 255
    ).astype(np.uint8)

    restored = (
        restored * 255
    ).astype(np.uint8)

    # Resize degraded to GT size
    degraded_img = Image.fromarray(
        degraded
    ).resize(
        Image.fromarray(ground_truth).size
    )

    restored_img = Image.fromarray(
        restored
    )

    gt_img = Image.fromarray(
        ground_truth
    )

    # Make comparison canvas
    width = gt_img.width
    height = gt_img.height

    canvas = Image.new(
        "L",
        (width * 3, height + 40),
        255
    )

    canvas.paste(
        degraded_img,
        (0, 40)
    )

    canvas.paste(
        restored_img,
        (width, 40)
    )

    canvas.paste(
        gt_img,
        (width * 2, 40)
    )

    draw = ImageDraw.Draw(canvas)

    draw.text(
        (width // 2 - 35, 10),
        "Degraded"
    )

    draw.text(
        (width + width // 2 - 35, 10),
        "Restored"
    )

    draw.text(
        (width * 2 + width // 2 - 55, 10),
        "Ground Truth"
    )

    output_path = os.path.join(
        OUT_DIR,
        image_id + "_comparison.png"
    )

    canvas.save(
        output_path
    )

    print(
        "Saved:",
        output_path
    )


print()
print("Comparison generation complete!")
print(
    "Open:",
    OUT_DIR
)