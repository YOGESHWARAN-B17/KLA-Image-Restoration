import os
import numpy as np
import torch
from torch.utils.data import Dataset


class KLADataset(Dataset):

    def __init__(
        self,
        degraded_dir="data/train/degraded",
        gt_dir="data/train/ground_truth"
    ):

        self.degraded_dir = degraded_dir
        self.gt_dir = gt_dir

        degraded_files = sorted(
            f for f in os.listdir(degraded_dir)
            if f.endswith("_degraded.npy")
        )

        self.pairs = []

        for degraded_file in degraded_files:

            image_id = degraded_file.replace(
                "_degraded.npy",
                ""
            )

            gt_file = image_id + "_gt.npy"

            degraded_path = os.path.join(
                degraded_dir,
                degraded_file
            )

            gt_path = os.path.join(
                gt_dir,
                gt_file
            )

            if os.path.exists(gt_path):

                self.pairs.append(
                    (degraded_path, gt_path)
                )

        print("Dataset pairs:", len(self.pairs))


    def __len__(self):

        return len(self.pairs)


    def __getitem__(self, index):

        degraded_path, gt_path = self.pairs[index]

        degraded = np.load(
            degraded_path
        ).astype(np.float32)

        ground_truth = np.load(
            gt_path
        ).astype(np.float32)

        degraded = torch.from_numpy(
            degraded
        ).unsqueeze(0)

        ground_truth = torch.from_numpy(
            ground_truth
        ).unsqueeze(0)

        return degraded, ground_truth


if __name__ == "__main__":

    dataset = KLADataset()

    print()
    print("Number of pairs:", len(dataset))

    if len(dataset) > 0:

        degraded, ground_truth = dataset[0]

        print()
        print("First sample:")
        print("Degraded shape:", degraded.shape)
        print("Ground truth shape:", ground_truth.shape)

        print(
            "Degraded min:",
            degraded.min().item()
        )

        print(
            "Degraded max:",
            degraded.max().item()
        )

        print(
            "Ground truth min:",
            ground_truth.min().item()
        )

        print(
            "Ground truth max:",
            ground_truth.max().item()
        )