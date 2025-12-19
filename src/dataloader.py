import numpy as np
import torch
from torch.utils.data import Dataset

class MemmapDataset(Dataset):
    def __init__(self, X_path, y_path, shape_X, shape_y):
        self.X = np.memmap(X_path, dtype=np.float32, mode="r", shape=shape_X)
        self.y = np.memmap(y_path, dtype=np.float32, mode="r", shape=shape_y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx].copy()),
            torch.from_numpy(self.y[idx].copy())
        )