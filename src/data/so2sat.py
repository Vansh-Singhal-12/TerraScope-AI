"""So2Sat LCZ42 HDF5 dataset access."""

import h5py
import numpy as np
from torch.utils.data import Dataset

from .preprocessing import preprocess_sentinel2_rgb


class So2SatDataset(Dataset):
    """Read So2Sat Sentinel-2 RGB patches and 17-class one-hot labels.

    Opens the HDF5 file lazily in each process to avoid sharing an HDF5
    handle across DataLoader workers. For this project, num_workers=0
    has been the most reliable Colab configuration.
    """

    def __init__(self, h5_path: str, indices, reflectance_max: float = 2.8):
        self.h5_path = h5_path
        self.indices = np.asarray(indices, dtype=np.int64)
        self.reflectance_max = reflectance_max
        self.file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")

        source_index = int(self.indices[index])

        # Stored channel order begins B2, B3, B4. Reverse the first
        # three channels to obtain B4, B3, B2 = RGB.
        raw = self.file["sen2"][source_index, :, :, :3][..., ::-1].copy()
        image = preprocess_sentinel2_rgb(
            raw,
            reflectance_max=self.reflectance_max,
        )

        label = int(np.argmax(self.file["label"][source_index]))
        return image, label

    def __del__(self):
        if self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass


def load_so2sat_split_indices(split_path: str):
    """Load the saved train_idx and val_idx arrays from the project NPZ."""
    split = np.load(split_path)
    return split["train_idx"], split["val_idx"]
