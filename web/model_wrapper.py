import torch
from app.utils.config import Config
from app.models.transformer_encoder_forecaster import patch_transformer
from app.data.dataset import forecasting_Dataset
from app.scoring.residual_computation import compute_residuals
from app.scoring.knn_scorer import KNNResidualScorer