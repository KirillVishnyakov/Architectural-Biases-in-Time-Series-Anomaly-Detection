import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.utils.data as data
from sklearn.preprocessing import RobustScaler



class myDataset(data.Dataset):
    def __init__(self, device, window_size, train = True):
        self.dataset = pd.read_csv("/kaggle/input/datasets/kirillvishnyakov/cats-dataset/data.csv").drop(["y", "category", "timestamp"], axis = 1)
        self.normalized_dataset = RobustScaler().fit_transform(self.dataset)

        start, end = (0, 90000) if train else (90000, 100000)
        self.clean_data =  torch.from_numpy(self.normalized_dataset[start: end])
        
        self.X, self.y = torch.zeros((end - start - window_size, window_size, 17)), torch.zeros((end - start - window_size, 17))
        for i in range(len(self.clean_data) - window_size):
            window = self.clean_data[i: i+window_size]
            target = self.clean_data[i+window_size]

            self.X[i] = window
            self.y[i] = target
        self.X = self.X.to(device)
        self.y = self.y.to(device)
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
