import os
import numpy as np
import matplotlib.pyplot as plt

degraded_dir = "data/train/degraded"
gt_dir = "data/train/ground_truth"

output_dir = "outputs/preview"
os.makedirs(output_dir, exist_ok=True)

degraded_files = sorted(
    f for f in os.listdir(degraded_dir)
    if f.endswith("_degraded.npy")
)

print("Degraded images found:", len(degraded_files))

for degraded_file in degraded_files:

    image_id = degraded_file.replace("_degraded.npy", "")

    gt_file = image_id + "_gt.npy"

    degraded_path = os.path.join(
        degraded_dir,
        degraded_file
    )

    gt_path = os.path.join(
        gt_dir,
        gt_file
    )

    print("\nChecking:", image_id)

    if not os.path.exists(gt_path):
        print("  Ground truth NOT found:", gt_file)
        continue

    degraded = np.load(degraded_path)
    ground_truth = np.load(gt_path)

    print("  Degraded shape:", degraded.shape)
    print("  Ground truth shape:", ground_truth.shape)

    print(
        "  Degraded range:",
        float(degraded.min()),
        "to",
        float(degraded.max())
    )

    print(
        "  Ground truth range:",
        float(ground_truth.min()),
        "to",
        float(ground_truth.max())
    )

    # Display only: clip for visualization.
    # Original .npy data is NOT modified.
    degraded_display = np.clip(degraded, 0, 1)
    gt_display = np.clip(ground_truth, 0, 1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(degraded_display, cmap="gray")
    plt.title("Degraded")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(gt_display, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.tight_layout()

    output_file = os.path.join(
        output_dir,
        image_id + "_comparison.png"
    )

    plt.savefig(output_file, dpi=150)
    plt.close()

    print("  Saved:", output_file)

print("\nFinished!")