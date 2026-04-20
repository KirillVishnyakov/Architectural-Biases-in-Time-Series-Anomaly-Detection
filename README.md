# Mapping Architectural Biases in Time Series Anomaly Detection

This project compares how different deep learning architectures behave in multivariate time series anomaly detection. An LSTM forecaster, LSTM autoencoder, and PatchTST-inspired transformer
encoder are evaluated using a residual-based pipeline with k-nearest neighbors anomaly scoring. While all models achieve similar overall F1 scores, they exhibit different detection patterns
across anomaly types. The reconstruction-based model perform worse on complex anomalies, while the forecasting-based models are more consistent. Between forecasting models, the transformer and LSTM
achieve similar overall performance, but show different recall patterns across anomaly types. These differences indicate that architectural choices introduce biases in anomaly detection.

<img width="1190" height="295" alt="pa_recall" src="https://github.com/user-attachments/assets/d243c978-6029-4e8b-b890-d7d52bb0d1f2" />

## Structure

```
Architectural-Biases-in-Time-Series-Anomaly-Detection/
├── images/                # images used in report
├── models/                # the different models 
├── notebooks/             # exploratory notebooks
│   ├── tuning/
│   ├── evaluation/
│   └── scoring_residuals.ipynb
├── saved_model_weights/   # model weights
├── saved_model_scores/    # model anomaly scores
├── utils/                 # utility files
├── dataset.py             # configures dataset for models
├── train.py               # training loop
├── README.md
└── report.ipynb           # project report
```


# Citations
I use RevIN (Kim et al., 2021) to mitigate distribution shift via reversible normalization.
RevIN.py is taken from (https://github.com/ts-kim/RevIN/tree/master)
