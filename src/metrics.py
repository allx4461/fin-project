import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import binomtest
from typing import Dict, Any


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes standard financial regression and directional accuracy metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Directional Accuracy (Hit Ratio): fraction of days where predicted sign matches real sign
    dir_acc = float(np.mean((y_true > 0) == (y_pred > 0)))

    return {
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "dir_acc": round(dir_acc, 4),
    }


def binomial_test_p_value(
    y_true: np.ndarray, y_pred: np.ndarray, baseline_acc: float = 0.50
) -> float:
    """
    Performs an exact binomial test (H0: accuracy <= baseline_acc, H1: accuracy > baseline_acc).
    Returns the p-value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    k = int(np.sum((y_true > 0) == (y_pred > 0)))

    test_res = binomtest(k, n, p=baseline_acc, alternative="greater")
    return float(round(test_res.pvalue, 4))


def compute_baselines(y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Computes standard academic baseline models:
    1. 'Always Up' (Majority/Bull baseline): predicts positive return (+1) every day.
    2. 'Yesterday Sign' (Momentum baseline): predicts tomorrow will have same sign as yesterday.
    3. 'Mean Predictor' (Dummy baseline): predicts the historical train mean return.
    """
    n_test = len(y_test)
    y_test_arr = np.asarray(y_test)
    
    # 1. Always Up
    always_up_pred = np.ones(n_test) * 0.001
    always_up_metrics = evaluate_predictions(y_test_arr, always_up_pred)
    
    # 2. Mean Dummy
    mean_val = float(np.mean(y_train))
    mean_pred = np.full(n_test, mean_val)
    mean_metrics = evaluate_predictions(y_test_arr, mean_pred)
    
    return {
        "always_up_baseline": always_up_metrics,
        "mean_baseline": mean_metrics,
        "up_days_ratio": float(round(np.mean(y_test_arr > 0), 4)),
    }
