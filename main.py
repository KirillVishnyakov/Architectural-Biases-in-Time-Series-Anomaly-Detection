import torch
import os
import numpy as np
import json
import gc

from app.utils.config import Config
from app.utils.evaluation_utils import evaluation_metrics_helper
from app.models.transformer_encoder_forecaster import patch_transformer
from app.data.dataset import forecasting_Dataset
from app.scoring.residual_computation import compute_residuals
from app.scoring.knn_scorer import fit_custom_N2RE
from app.scoring.knn_scorer import score_custom_N2RE
from app.scoring.knn_scorer import KNNResidualScorer


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(
        "c:\\Architectural-Biases-in-Time-Series-Anomaly-Detection", 
        "saved_model_weights",
        "training", 
        "training_results",
        "data.csv", 
        "training_checkpoints"
    )
    transformer_model = patch_transformer(
        lookback_window = 256, 
        forecast_horizon = 4,
        d_model = 256, 
        nhead = 8, 
        dropout = 0.0, 
        num_features = 17, 
        num_blocks = 1
    )
    """
    transformer_model.load_state_dict(
        torch.load(
            os.path.join(
                cfg.weights, 'transformer_forecaster_weights.pt'),
                map_location = device
        )
    )
    """
    
    train_dataset = forecasting_Dataset(
        device, cfg, 256, 4, start = 0, end = 256*4)

    val_dataset = forecasting_Dataset(
        device, cfg,  256, 4, scaler = train_dataset.scaler, 
        start = 256*4, end = 256*8, train = False)
    
    train_residuals, _, _ = compute_residuals(device, transformer_model, train_dataset, batch_size = 1)
    val_residuals, labels, cats = compute_residuals(device, transformer_model, val_dataset, batch_size = 1, train = False)
    print(train_residuals.shape)
    print()
    print(val_residuals.shape)
    print(labels.shape)
    print(cats.shape)

    knn_module = KNNResidualScorer(device, 0.95, 4)
    knn_module = knn_module.fit(train_residuals)
    scores = knn_module.score(val_residuals)
    print(scores.shape)
