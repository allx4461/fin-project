import pandas as pd
from pathlib import Path
import sys
import numpy as np
from catboost import CatBoostRegressor
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.models import time_split,evaluate_predictions

from src.features import PRICE_FEATURES, FINBERT_FEATURES, ROBERTA_FEATURES, VADER_FEATURES, CALENDAR_FEATURES

def train_catboosts(df: pd.DataFrame,dth=4,lrt=0.03,l2=5.):
    train_df,val_df,test_df=time_split(df)
    for feature_list_tuple in [(PRICE_FEATURES,CALENDAR_FEATURES),(PRICE_FEATURES,CALENDAR_FEATURES,FINBERT_FEATURES),(PRICE_FEATURES,CALENDAR_FEATURES,ROBERTA_FEATURES),(PRICE_FEATURES,CALENDAR_FEATURES,VADER_FEATURES),(PRICE_FEATURES,CALENDAR_FEATURES,FINBERT_FEATURES,ROBERTA_FEATURES,VADER_FEATURES)]:
        model = CatBoostRegressor(
            iterations=500,          # максимальное число деревьев (эпох)
            learning_rate=lrt,      # скорость обучения (шаг градиентного спуска)
            depth=dth,                 # глубина деревьев (для финансов 4–6)
            # L2-регуляризация (защита от переобучения на шуме)
            l2_leaf_reg=l2,
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
        feature_imp = pd.Series(importances, index=all_required).sort_values(ascending=False)
        print(f'\nmodel used {feature_list} \nsetted depth={dth} learnrate={lrt} l2_reg={l2}\ngot r2 accuracy {evaluate_predictions(y_val,y_val_pred)['r2']}\ntop importances {feature_imp}\n')
if __name__=='__main__':
    df=pd.read_csv(PROJECT_ROOT / "data" / "processed" / "nvda_features.csv")
    for dth in [3,4,6,8]:
        for lrt in [0.01,0.03,0.05,0.1,0.3]:
            for l2 in [1.,3.,5.,10.]: 
                train_catboosts(df,dth,lrt,l2)
