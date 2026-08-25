import pandas as pd
from pathlib import Path
import sys
import numpy as np
from catboost import CatBoostRegressor
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.models import time_split,evaluate_predictions

from src.features import PRICE_FEATURES, FINBERT_FEATURES, ROBERTA_FEATURES, VADER_FEATURES, CALENDAR_FEATURES

def train_catboosts(df: pd.DataFrame):
    train_df,val_df,test_df=time_split(df)
    for feature_list_tuple in [(PRICE_FEATURES,CALENDAR_FEATURES),(PRICE_FEATURES,CALENDAR_FEATURES,FINBERT_FEATURES),(PRICE_FEATURES,CALENDAR_FEATURES,ROBERTA_FEATURES),(PRICE_FEATURES,CALENDAR_FEATURES,VADER_FEATURES),(PRICE_FEATURES,CALENDAR_FEATURES,FINBERT_FEATURES,ROBERTA_FEATURES,VADER_FEATURES)]:
        model = CatBoostRegressor(
            iterations=500,          # максимальное число деревьев (эпох)
            learning_rate=0.03,      # скорость обучения (шаг градиентного спуска)
            depth=4,                 # глубина деревьев (для финансов 4–6)
            # L2-регуляризация (защита от переобучения на шуме)
            l2_leaf_reg=5.0,
            random_seed=42,          # фиксация случайности для воспроизводимости
            # выводить лог каждые 100 деревьев (или False для тишины)
            verbose=False
        )  # бтв более подробное пояснение за параметры накарякала в вольте
        all_required=[]
        for feature_list in feature_list_tuple: all_required.extend(feature_list)
        x_train=train_df[all_required]
        y_train=train_df['target_return']
        x_val=val_df[all_required]
        y_val=val_df['target_return']
        x_test=test_df[all_required]
        y_test=test_df['target_return']
        model.fit(x_train, y_train, eval_set=(x_val, y_val),
                        verbose=False)  # fit это тренировка
        y_val_pred = model.predict(x_val)
        y_test_pred = model.predict(x_test)
        importances = model.get_feature_importance()
        print(f'\n\n\n\n done! catboost fitted on {all_required} features \n\n and predicted {y_test_pred}, {y_val_pred} \n\n found these importances {importances}\n\n')
        print(f'\n\npredictions evaluated as \n\n{evaluate_predictions(y_val,y_val_pred)}\n\n\n\n')
if __name__=='__main__':
    df=pd.read_csv(PROJECT_ROOT / "data" / "processed" / "nvda_features.csv")
    train_catboosts(df)
