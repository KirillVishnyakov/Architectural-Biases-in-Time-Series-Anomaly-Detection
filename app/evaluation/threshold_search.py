import numpy as np
from app.utils.evaluation_utils import evaluation_metrics_helper

def search_thresholds(scores, labels, cats):
    results = []
    for anomaly_percentile in [3.0, 3.8, 4.6, 5.5]:
        threshold_percentile = 100 - anomaly_percentile
        for window in np.linspace(150, 600, 7).astype(int):
            F1, precision, cat_dict =  \
                evaluation_metrics_helper(
                    scores, 
                    labels, 
                    cats, 
                    window, 
                    threshold_percentile
                )
            results.append({
                "F1": round(F1, 3),
                "precision": round(precision, 3),
                "threshold_percentile": threshold_percentile,
                "smoothing_window": window,
                "recalls_per_anomaly_cat": cat_dict,
            })

    results.sort(key=lambda x: x['F1'], reverse=True)
    top_results = results[:5]
    return top_results