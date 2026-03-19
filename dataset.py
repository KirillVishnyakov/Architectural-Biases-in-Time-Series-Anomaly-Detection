import pandas as pd
import torch
import torch.utils.data as data
import numpy as np
from sklearn.preprocessing import RobustScaler


# Stride of 1 is implicitly implied. Window size can be set.
class LSTM_Dataset(data.Dataset):
    def __init__(self, device, window_size, start = 0, end = 90000, train = True):
        self.train = train
        self.window_size = window_size
        self.device = device
        if self.train:
            self.dataset = pd.read_csv("/kaggle/input/datasets/kirillvishnyakov/cats-dataset/data.csv", skiprows=range(1, start+1), nrows=end - start).drop(["y", "category", "timestamp"], axis = 1)
        else:
            self.dataset = pd.read_csv("/kaggle/input/datasets/kirillvishnyakov/cats-dataset/data.csv", skiprows=range(1, start+1), nrows=end - start).drop(["timestamp"], axis = 1)
            self.labels = self.dataset["y"].values
            self.categories = self.dataset["category"].values
            self.total_anomalies = np.sum(self.labels == 1)
            self.dataset = self.dataset.drop(["y", "category"], axis = 1)

        self.normalized_dataset = torch.from_numpy(RobustScaler().fit_transform(self.dataset))

    def __len__(self):
        return len(self.normalized_dataset) - self.window_size

    def __getitem__(self, idx):
        X = self.normalized_dataset[idx : idx + self.window_size]
        y = self.normalized_dataset[idx + self.window_size]

        if self.train:
            return X.to(self.device), y.to(self.device)
        else: 
            label = self.labels[idx + self.window_size]
            category = self.categories[idx + self.window_size]
            return X.to(self.device), y.to(self.device), label, category


