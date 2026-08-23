import pandas as pd
from pathlib import Path
import datetime
from tqdm import tqdm
import pandas_market_calendars as mcal
import numpy as np
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nasdaq_exteral_data.csv"
chunksize = 100_000
with open(DATA_PATH, 'rb') as f:
    row_count = sum(1 for _ in f) - 1
total_chunks = row_count // chunksize + 1


def find_only_nvda(DATA_PATH) -> pd.DataFrame:
    chunks = pd.read_csv(DATA_PATH, chunksize=100_000,
                         usecols=['Stock_symbol', 'Date','Article_title'])
    nvda_news_list = []
    for chunk in tqdm(chunks, desc="Filtering NVDA", total=total_chunks):
        nvda_chunk = chunk[chunk['Stock_symbol'] == 'NVDA']
        nvda_news_list.append(nvda_chunk)
    nvda_news = pd.concat(nvda_news_list)
    return nvda_news


def get_nasdaq_trading_days(start, end) -> pd.DatetimeIndex:
    nasdaq = mcal.get_calendar('NASDAQ')
    schedule = nasdaq.schedule(start_date=start, end_date=end)
    return schedule.index.normalize()


def assign_trading_day(dates: pd.Series, trading_days: pd.DatetimeIndex, market_close_hour: int = 16) -> pd.Series:
    # after 16. -> next day & off-days -> nearest working day
    dates = pd.to_datetime(dates)
    day = dates.dt.normalize()
    after_close = dates.dt.hour >= market_close_hour
    effective_day = day.where(~after_close, day + pd.Timedelta(days=1))
    td_values = trading_days.values
    idx = np.searchsorted(td_values, effective_day.values, side='left')
    idx = np.clip(idx, 0, len(td_values) - 1)

    return pd.Series(td_values[idx], index=dates.index)

def clean_nvda_news(news: pd.DataFrame) -> pd.DataFrame:
    news = news.copy()
    news['Date'] = pd.to_datetime(news['Date'])

    trading_days = get_nasdaq_trading_days(
        start=news['Date'].min() - pd.Timedelta(days=7),
        end=news['Date'].max() + pd.Timedelta(days=7),
    )
    news['trading_day'] = assign_trading_day(news['Date'], trading_days)
    return news

if __name__ == "__main__":
    nvda_news = find_only_nvda(DATA_PATH)
    nvda_news = clean_nvda_news(nvda_news)
    print(nvda_news[['Date', 'trading_day']].head())