import pandas as pd
from pathlib import Path
import datetime
from tqdm import tqdm
import pandas_market_calendars as mcal
import numpy as np
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nasdaq_exteral_data.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "nvda_news.csv"
chunksize = 50_000
with open(DATA_PATH, 'rb') as f:
    row_count = sum(1 for _ in f) - 1
total_chunks = row_count // chunksize + 1


def find_only_nvda_streaming(data_path, output_path, chunksize=CHUNK_SIZE):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_chunk = True

    chunks = pd.read_csv(data_path, chunksize=chunksize, usecols=['Stock_symbol', 'Date'])
    for chunk in tqdm(chunks, desc="Filtering NVDA"):
        nvda_chunk = chunk[chunk['Stock_symbol'] == 'NVDA']
        if not nvda_chunk.empty:
            nvda_chunk.to_csv(output_path, mode='a', header=first_chunk, index=False)
            first_chunk = False
        del chunk, nvda_chunk 

find_only_nvda_streaming(DATA_PATH, OUTPUT_PATH)
nvda_news = pd.read_csv(OUTPUT_PATH)

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
    nvda_news = clean_nvda_news(nvda_news)
    print(nvda_news[['Date', 'trading_day']].head())