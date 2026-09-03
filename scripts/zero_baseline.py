from src.strategy import backtest
from src.models import time_split, evaluate_predictions
from src.features import FEATURE_SETS
from src.config import TICKER, DATA_PROCESSED_FEATURES, RESULTS_DIR, RANDOM_SEED
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


def train_model(df: pd.DataFrame):
    _, val_df, test_df = time_split(df)
    x_test = test_df['Date']
    x_val = val_df['Date']

    val_pred = np.zeros(len(x_test))
    test_pred = np.zeros(len(x_val))
    return val_pred, test_pred


def validate(df, val_pred, test_pred):
    _, val_df, test_df = time_split(df)
    y_val = val_df['target_return']
    y_test = test_df['target_return']
    val_metrics = evaluate_predictions(y_val, val_pred)
    test_metrics = evaluate_predictions(y_test, test_pred)
    _, strategy_metrics = backtest(y_val, val_pred, val_df['Date'])
    return val_metrics, test_metrics, strategy_metrics


def run_grid_search(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    val_metrics, test_metrics, strategy_metrics = validate(
        df, train_model(df)[0], train_model(df)[1])
    results.append({
        'feature_set': 'no_feature',
        'val_dir_acc': round(val_metrics['dir_acc'], 4),
        'val_r2': round(val_metrics['r2'], 4),
        'val_mae': round(val_metrics['mae'], 4),
        'test_dir_acc': round(test_metrics['dir_acc'], 4),
        'test_r2': round(test_metrics['r2'], 4),
        'test_mae': round(test_metrics['mae'], 4),
        'total_return': round(strategy_metrics['total_return'], 4),
        'sharpe': round(strategy_metrics['sharpe'], 4),
        'max_drawdown': round(strategy_metrics['max_drawdown'], 4),
        'win_rate': round(strategy_metrics['win_rate'], 4),
    })
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(
        by='val_dir_acc', ascending=False).reset_index(drop=True)
    return res_df


def start():
    data_path = DATA_PROCESSED_FEATURES
    output_path = RESULTS_DIR / "metrics_zero_baseline.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    metrics_df = run_grid_search(df)

    metrics_df.to_csv(output_path, index=False)
    print(f"\n```````````````baseline ({TICKER})```````````````\n ")
    print(metrics_df.head(10).to_string())
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    start()
