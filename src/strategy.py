import numpy as np
import pandas as pd
from math import sqrt

def backtest(y_true, y_pred, dates):
    position = np.where(y_pred > 0, 1.0, 0.0)
    strategy_return = position * y_true
    benchmark_return = y_true  # просто купи и держи
    cumulative_strategy = np.cumprod(1 + strategy_return) - 1
    cumulative_benchmark = np.cumprod(1 + benchmark_return) - 1
    sharpe = np.mean(strategy_return) / np.std(strategy_return) * sqrt(252)
    win_rate=np.mean(strategy_return > 0)
    equity = np.cumprod(1 + strategy_return)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = np.min(drawdown)

    df=pd.DataFrame({
        'date':dates,
        'position': position,
        'strategy_return': strategy_return,
        'cumulative_strategy': cumulative_strategy,
        'cumulative_benchmark':cumulative_benchmark
    })
    metrics={'total_return': cumulative_strategy.iloc(-1), 'sharpe': sharpe, 'max_drawdown': max_drawdown, 'win_rate': win_rate}
    return df,metrics