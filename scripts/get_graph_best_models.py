from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import TICKER, DATA_PROCESSED_FEATURES, FIGURES_DIR
from src.features import FEATURE_SETS
from src.models import time_split
import scripts.train_linear as train_linear
import scripts.train_forest as train_forest
import scripts.train_catboost as train_catboost


def get_graph(output_path=None):
    if output_path is None:
        output_path = FIGURES_DIR / "price_vs_prediction.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PROCESSED_FEATURES)

    train_df, val_df, test_df = time_split(df)
    date_col = 'Date' if 'Date' in df.columns else 'trading_day'
    dates_val = pd.to_datetime(val_df[date_col].values)
    dates_test = pd.to_datetime(test_df[date_col].values)
    y_val_true = val_df['target_return'].values
    y_test_true = test_df['target_return'].values

    def build_model_df(val_pred, test_pred, model_name):  
        v_df = pd.DataFrame({'date': dates_val, 'y_true': y_val_true, 'y_pred': val_pred, 'split': 'val', 'model': model_name})
        t_df = pd.DataFrame({'date': dates_test, 'y_true': y_test_true, 'y_pred': test_pred, 'split': 'test', 'model': model_name})
        return pd.concat([v_df, t_df], ignore_index=True)

    val_pred_linear, test_pred_linear = train_linear.train_model(df, FEATURE_SETS['price_roberta'])
    pred_linear = build_model_df(val_pred_linear, test_pred_linear, 'linear')

    val_pred_forest, test_pred_forest = train_forest.train_model(df, FEATURE_SETS['all_features'])
    pred_forest = build_model_df(val_pred_forest, test_pred_forest, 'forest')

    val_pred_catboost, test_pred_catboost = train_catboost.train_model(df, 0.05, 8, 1.0, FEATURE_SETS['all_features'])
    pred_catboost = build_model_df(val_pred_catboost, test_pred_catboost, 'catboost')

    all_preds = pd.concat([pred_linear, pred_forest, pred_catboost], ignore_index=True)
    all_preds['date'] = pd.to_datetime(all_preds['date'])
    all_preds = all_preds.sort_values('date')

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for ax, model_name in zip(axes, ['linear', 'forest', 'catboost']):
        model_data = all_preds[all_preds['model'] == model_name]
        val_data = model_data[model_data['split'] == 'val']
        test_data = model_data[model_data['split'] == 'test']

        ax.plot(model_data['date'], model_data['y_true'], label='real', color='black', linewidth=1)
        ax.plot(val_data['date'], val_data['y_pred'], label='val', color='steelblue', alpha=0.7)
        ax.plot(test_data['date'], test_data['y_pred'], label='test', color='crimson', alpha=0.7)

        if not test_data.empty:
            ax.axvline(test_data['date'].min(), color='gray', linestyle='--', alpha=0.5)

        ax.set_title(f"{TICKER} — {model_name}")
        ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


if __name__ == '__main__':
    get_graph()