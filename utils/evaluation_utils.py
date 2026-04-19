import numpy as np
def evaluation_metrics_helper(scores, labels, cats, window):
    window = int(window)
    current_score = np.convolve(scores, np.ones(window) / window, mode='same')
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