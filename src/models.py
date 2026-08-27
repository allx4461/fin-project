import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from src.config import TRAIN_SPLIT, VAL_SPLIT
def time_split(df:pd.DataFrame,train:float=TRAIN_SPLIT,validation:float=VAL_SPLIT):
    n=len(df)
    train_end=int(n*train)
    validation_end=int(n*(train+validation))
    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:validation_end]
    test_df  = df.iloc[validation_end:]
    return train_df, val_df, test_df

def evaluate_predictions(y_true,y_pred)->dict:
    r2=r2_score(y_true,y_pred)
    mae=mean_absolute_error(y_true,y_pred)
    hit_ratio=np.mean((y_true>0)==(y_pred>0))
    return {'r2':r2,'mae':mae,'dir_acc':hit_ratio}