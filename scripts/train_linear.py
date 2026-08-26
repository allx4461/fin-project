from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.features import PRICE_FEATURES, FINBERT_FEATURES, ROBERTA_FEATURES, VADER_FEATURES, CALENDAR_FEATURES
from src.models import time_split, evaluate_predictions

FEATURE_SETS = {
    'price_only': PRICE_FEATURES + CALENDAR_FEATURES,
    'price_finbert': PRICE_FEATURES + CALENDAR_FEATURES + FINBERT_FEATURES + ['has_news'],
    'price_roberta': PRICE_FEATURES + CALENDAR_FEATURES + ROBERTA_FEATURES + ['has_news'],
    'price_vader': PRICE_FEATURES + CALENDAR_FEATURES + VADER_FEATURES + ['has_news'],
    'all_features': PRICE_FEATURES + CALENDAR_FEATURES + FINBERT_FEATURES + ROBERTA_FEATURES + VADER_FEATURES + ['has_news']
}

def run_grid_search(df: pd.DataFrame) -> pd.DataFrame:
    train_df, val_df, test_df = time_split(df)
    results = []
    for feat_name, features in FEATURE_SETS.items():
        X_train, y_train = train_df[features], train_df['target_return']
        X_val, y_val = val_df[features], val_df['target_return']
        X_test, y_test = test_df[features], test_df['target_return']    
        model = LinearRegression()
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_metrics = evaluate_predictions(y_val, val_pred)
        test_metrics = evaluate_predictions(y_test, test_pred)

        results.append({
            'feature_set': feat_name,
            'val_dir_acc': round(val_metrics['dir_acc'], 4),
            'val_r2': round(val_metrics['r2'], 4),
            'val_mae': round(val_metrics['mae'], 4),
            'test_dir_acc': round(test_metrics['dir_acc'], 4),
            'test_r2': round(test_metrics['r2'], 4),
            'test_mae': round(test_metrics['mae'], 4),
        })

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(
        by='val_dir_acc', ascending=False).reset_index(drop=True)
    return res_df

def start():
    data_path = PROJECT_ROOT / "data" / "processed" / "nvda_features.csv"
    output_path = PROJECT_ROOT / "results" / "metrics_linear.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    metrics_df = run_grid_search(df)

    metrics_df.to_csv(output_path, index=False)
    print("\n```````````````best 10 linear```````````````\n ")
    print(metrics_df.head(10).to_string())
    #print(f"\n results in {output_path}")

if __name__ == '__main__':
    start()
