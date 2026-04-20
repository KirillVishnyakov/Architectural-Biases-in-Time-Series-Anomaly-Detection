import numpy as np
def evaluation_metrics_helper(scores, labels, cats, window):
    """ computes evaluation metrics for anomaly detection using smoothed residual scores
    
    Args
    ---------
    scores : np.ndarray (N,)
        raw anomaly scores
    labels : np.ndarray (N,)
        binary ground truth labels
    cats : np.ndarray (N,)
        categorical anomaly labels
    window : int
        smoothing window size for moving average over scores

    Returns
    ---------
    tuple[float, float, dict]:
        F1 score, precision, and dictionary of per-category metrics.
    Example
    ---------
    >>> scores = np.random.rand(1000)
    >>> labels = np.zeros(1000)
    >>> labels[900:] = 1
    >>> cats = np.zeros(1000)
    >>> cats[900:950] = 1
    >>> cats[950:] = 2
    >>> F1, precision, cat_metrics = \
    ...     evaluation_metrics_helper(scores, labels, cats, window=50)
    >>> F1
    """
    window = int(window)
    kernel = np.ones(window) / window # [len(window),]
    current_score = np.convolve(scores, kernel, mode='same')
    cat_dict = {}
    normal_mask = cats == 0
    threshold = np.percentile(current_score[normal_mask], 100 - 1.5)
    FPR = np.sum(current_score[normal_mask] > threshold) / np.sum(normal_mask)
    cat_dict["FPR (0.0)"] = round(float(FPR), 2)

    for cat in np.unique(cats):
        if cat == 0.0:
            continue
        cat_mask = cats == cat
        recall = np.sum(current_score[cat_mask] > threshold) / np.sum(cat_mask)
        cat_dict[f"r({int(cat)})"] = round(float(recall), 2)

    TP = np.sum((current_score > threshold) & (labels == 1))
    FP = np.sum((current_score > threshold) & (labels == 0))
    recall = TP / np.sum(labels == 1)
    precision = TP / (TP + FP)
    F1 = 2 * (precision * recall) / (precision + recall)
    return F1, precision, cat_dict