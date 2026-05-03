import numpy as np
import json
from app.scoring.residual_computation import compute_residuals
from app.evaluation.threshold_search import search_thresholds

def evaluate_model(device, model, train_dataset, val_dataset, name, cfg, scorers):
    print(f"--- evaluating model {name} --- ")
    path = cfg.training_results_dir / name
    path.mkdir(parents=True, exist_ok=True)

    train_residuals, _, _ = compute_residuals(device, model, train_dataset, batch_size = 512, train = False)
    val_residuals, labels, cats = compute_residuals(device, model, val_dataset, batch_size = 512, train = False)
    results = []
    for scorer in scorers:
        scorer = scorer.fit(train_residuals)
        scores = scorer.score(val_residuals).cpu().numpy()
        np.savez_compressed(
            path / f"{scorer.name()}_residuals_scores_labels_categories.npz",
            residuals = val_residuals.cpu().numpy(),
            scores = scores,
            labels = labels.cpu().numpy(),
            categories = cats.cpu().numpy()
        )
        top_results = search_thresholds(scores, labels.cpu().numpy(), cats.cpu().numpy())
        for r in top_results:
            r["name"] = scorer.name()
        results.extend(top_results)

    results.sort(key=lambda x: x['F1'], reverse=True)
    with open(path / 'results.json', 'w') as f:
        json.dump(
            results[:3], 
            f, 
            indent = 4, 
            default = lambda x: x.item() if hasattr(x, "item") else str(x)
        )
