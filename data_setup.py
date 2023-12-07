from torch.utils.data import DataLoader
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
from typing import Tuple


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        data = torch.FloatTensor(self.data)
        targets = torch.LongTensor(self.targets)
        return data[idx], targets[idx]


def get_datasets(
        n_samples: int = 1000,
        lab_size: int = 100,
        unlab_size: int = 700,
        n_features: int = 10,
        n_classes: int = 2,
        random_state: int = 42,
        shuffle: bool = True
) -> Tuple[MyDataset, MyDataset, MyDataset]:
    """" Create datasets splitted in training labeled, training unlabeled, testing.
    """
    dataset = make_classification(
        n_samples = n_samples,
        n_features = n_features,
        n_classes = n_classes,
        n_informative = n_classes,
        random_state = random_state,
        shuffle = shuffle
    )
    lab_data, unlab_data, test_data = np.split(
        ary = dataset[0],
        indices_or_sections = [lab_size, (lab_size + unlab_size)]
    )
    lab_targets, unlab_targets, test_targets = np.split(
        ary = dataset[1],
        indices_or_sections = [lab_size, (lab_size + unlab_size)]
    )
    lab_dataset = MyDataset(data = lab_data, targets = lab_targets)
    unlab_dataset = MyDataset(data = unlab_data, targets = unlab_targets)
    test_dataset = MyDataset(data = test_data, targets = test_targets)
    return lab_dataset, unlab_dataset, test_dataset