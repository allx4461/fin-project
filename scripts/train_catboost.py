from pathlib import Path
import sys
import pandas as pd
from catboost import CatBoostRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import TICKER, DATA_PROCESSED_FEATURES, RESULTS_DIR, RANDOM_SEED
from src.features import FEATURE_SETS
from src.models import time_split, evaluate_predictions
from src.strategy import backtest



def train_model(df: pd.DataFrame, lrt, dth, l2, features):
    train_df, val_df, test_df = time_split(df)
    x_train = train_df[features] 
    x_val = val_df[features]
    x_test = test_df[features]
    y_train = train_df['target_return']
    y_val = val_df['target_return']

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=lrt,
        depth=dth,
        l2_leaf_reg=l2,
        random_seed=RANDOM_SEED,
        use_best_model=False,
        verbose=False
    )
    model.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=False)
    
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    return val_pred, test_pred


def validate(df, val_pred, test_pred):
    train_df, val_df, test_df = time_split(df)
    y_val = val_df['target_return']
    y_test = test_df['target_return']
    val_metrics = evaluate_predictions(y_val, val_pred)
    test_metrics = evaluate_predictions(y_test, test_pred)
    strategy_metrics=backtest(y_val,val_pred,df['Date'])
    return val_metrics, test_metrics, strategy_metrics


def run_grid_search(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    depths = [2, 3, 4, 6, 8]
    learning_rates = [0.01, 0.03, 0.05]
    l2_regs = [1., 3., 5., 10.]
    for feat_name, features in FEATURE_SETS.items():
        for dth in depths:
            for lrt in learning_rates:
                for l2 in l2_regs:
                    val_pred, test_pred = train_model(df, lrt, dth, l2, features)
                    val_metrics, test_metrics, strategy_metrics = validate(df, val_pred, test_pred)
                    results.append({
                        'feature_set': feat_name,
                        'depth': dth,
                        'learning_rate': lrt,
                        'l2_reg': l2,
                        'val_dir_acc': round(val_metrics['dir_acc'], 4),
                        'val_r2': round(val_metrics['r2'], 4),
                        'val_mae': round(val_metrics['mae'], 4),
                        'test_dir_acc': round(test_metrics['dir_acc'], 4),
                        'test_r2': round(test_metrics['r2'], 4),
                        'test_mae': round(test_metrics['mae'], 4),
                    })
                    results.extend(strategy_metrics)

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by='val_dir_acc', ascending=False).reset_index(drop=True)
    return res_df


def start():
    data_path = DATA_PROCESSED_FEATURES
    output_path = RESULTS_DIR / "metrics_catboost.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    metrics_df = run_grid_search(df)
    

    metrics_df.to_csv(output_path, index=False)
    print(f"\n```````````````best 10 catboost ({TICKER})```````````````\n ")
    print(metrics_df.head(10).to_string())
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    start()
