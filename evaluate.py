import os
import time
import argparse
import numpy as np
import torch
from PIL import Image

from model import KLAImageRestorationNet


# ============================================================
# Utility functions
# ============================================================

def load_npy(path):
    image = np.load(path).astype(np.float32)

    if image.ndim == 2:
        image = image[np.newaxis, :, :]

    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.transpose(image, (2, 0, 1))

    return image


def save_restored_image(image, path):
    """
    Save restored model output as an 8-bit grayscale PNG.
    """

    image = np.squeeze(image)

    image = np.clip(image, 0.0, 1.0)

    image_uint8 = (
        image * 255.0
    ).round().astype(np.uint8)

    Image.fromarray(
        image_uint8,
        mode="L"
    ).save(path)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="KLA Image Restoration Evaluation"
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing degraded .npy test images"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where restored images will be saved"
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pt",
        help="Path to trained model checkpoint"
    )

    parser.add_argument(
        "--ground_truth_dir",
        default=None,
        help="Optional directory containing ground-truth .npy files"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("KLA IMAGE RESTORATION - EVALUATION")
    print("=" * 70)

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(
            f"Input directory not found: {args.input_dir}"
        )

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading model...")

    model = KLAImageRestorationNet()

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    # Support either a direct state_dict or a checkpoint
    # containing a "model_state_dict" key.
    if isinstance(checkpoint, dict) and \
            "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model = model.to(device)
    model.eval()

    print(
        "Checkpoint:",
        args.checkpoint
    )

    # --------------------------------------------------------
    # Find test images
    # --------------------------------------------------------

    test_files = sorted(
        f
        for f in os.listdir(args.input_dir)
        if f.endswith("_degraded.npy")
    )

    if len(test_files) == 0:
        raise RuntimeError(
            "No *_degraded.npy files found in "
            + args.input_dir
        )

    print()
    print(
        "Input directory:",
        args.input_dir
    )

    print(
        "Output directory:",
        args.output_dir
    )

    print(
        "Test images:",
        len(test_files)
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    total_time = 0.0

    print()
    print("Starting inference...")
    print()

    with torch.no_grad():

        for index, filename in enumerate(
            test_files,
            start=1
        ):

            input_path = os.path.join(
                args.input_dir,
                filename
            )

            image = load_npy(
                input_path
            )

            image_tensor = torch.from_numpy(
                image
            ).float()

            # [C, H, W] -> [1, C, H, W]
            image_tensor = image_tensor.unsqueeze(0)

            image_tensor = image_tensor.to(
                device
            )

            # --------------------------------------------
            # GPU timing
            # --------------------------------------------

            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            restored = model(
                image_tensor
            )

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter() - start_time
            )

            total_time += elapsed

            # --------------------------------------------
            # Convert output
            # --------------------------------------------

            restored = (
                restored
                .detach()
                .cpu()
                .numpy()
            )

            image_id = filename.replace(
                "_degraded.npy",
                ""
            )

            output_filename = (
                image_id
                + "_restored.png"
            )

            output_path = os.path.join(
                args.output_dir,
                output_filename
            )

            save_restored_image(
                restored,
                output_path
            )

            print(
                f"[{index}/{len(test_files)}] "
                f"{filename} -> "
                f"{output_filename} "
                f"({elapsed * 1000:.2f} ms)"
            )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    average_time = (
        total_time / len(test_files)
    )

    images_per_second = (
        1.0 / average_time
        if average_time > 0
        else 0.0
    )

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        "Images processed:",
        len(test_files)
    )

    print(
        f"Total inference time: "
        f"{total_time:.4f} seconds"
    )

    print(
        f"Average inference time: "
        f"{average_time * 1000:.2f} ms/image"
    )

    print(
        f"Throughput: "
        f"{images_per_second:.2f} images/second"
    )

    print(
        "Restored outputs:",
        args.output_dir
    )

    print("=" * 70)


if __name__ == "__main__":
    main()