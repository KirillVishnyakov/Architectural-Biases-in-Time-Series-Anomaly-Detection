# Architectural-Biases-in-Time-Series-Anomaly-Detection

Comparing the anomaly detection capabilities of different neural architectures 
(short-window LSTM, long-window LSTM, dual-window LSTM, LSTM autoencoder, and transformer) on the 
CATS (https://www.kaggle.com/datasets/patrickfleith/controlled-anomalies-time-series-dataset/data) 
dataset, with a focus on per-anomaly-class detection performance.


## Structure

\```
notebooks/
├── tuning/
│   ├── tuning_LSTM_short.ipynb
│   ├── tuning_LSTM_long.ipynb
│   ├── tuning_LSTM_AE.ipynb
│   └── tuning_transformer.ipynb
├── evaluation/
│   ├── evaluating_LSTM_short.ipynb
│   ├── evaluating_LSTM_long.ipynb
│   ├── evaluating_LSTM_AE.ipynb
│   ├── evaluating_transformer.ipynb
│   └── comparison.ipynb
└── results/
    ├── LSTM_short_results.csv
    ├── LSTM_long_results.csv
    ├── LSTM_AE_results.csv
    └── transformer_results.csv

\```