import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import sys
import warnings
from itertools import product
import utils.config as config
from models.lstm_forecaster import lstm_forecaster
from models.lstm_ae import lstm_ae
from models.transformer_encoder_forecaster import patch_transformer
from utils.evaluation_utils import score_custom_N2RE
from utils.evaluation_utils import fit_custom_N2RE
from utils.evaluation_utils import compute_residuals
from dataset import forecasting_Dataset
import os
from train import transform_data
from torch.utils.data import DataLoader

if __name__ == "__main__":
    
    """    
    config.init("data.csv", "/checkpoint_dir")
    model = lstm_forecaster(hidden_size = 128, horizon = 4, num_layers = 1)
    train_dataset = forecasting_Dataset(
        'cpu', window_size=100, horizon=4, start=0, end=1000)
    val_dataset = forecasting_Dataset(
        'cpu', window_size=100, horizon=4, scaler = train_dataset.scaler, 
        start=1000, end=2500, train = False)
    weights, val_loss, train_loss = fit(
         'cpu', model, "exp", train_dataset, val_dataset,
        lr=1e-3, batch_size=32, num_epochs=10
    )"""
    
    x = torch.randn(2, 32, 100)
    x_transformed = transform_data('cpu', x, "train")
    print(x_transformed)
