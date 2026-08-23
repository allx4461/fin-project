from pathlib import Path
import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from tqdm import tqdm


def filter_ticker_streaming(data_path: Path, output_path: Path, ticker: str = "NVDA", chunksize: int = 50_000) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    first_chunk = True
    chunks = pd.read_csv(data_path, chunksize=chunksize, usecols=['Stock_symbol', 'Date', 'Article_title'])
    for chunk in tqdm(chunks, desc=f"Filtering {ticker}"):
        ticker_chunk = chunk[chunk['Stock_symbol'] == ticker]
        if not ticker_chunk.empty:
            ticker_chunk.to_csv(output_path, mode='a', header=first_chunk, index=False)
            first_chunk = False
        del chunk, ticker_chunk


def get_nasdaq_trading_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    nasdaq = mcal.get_calendar('NASDAQ')
    schedule = nasdaq.schedule(start_date=start, end_date=end)
    return schedule.index.normalize()


def assign_trading_day(dates: pd.Series, trading_days: pd.DatetimeIndex, market_close_hour: int = 16) -> pd.Series:
    dates = pd.to_datetime(dates)
    day = dates.dt.normalize()
    after_close = dates.dt.hour >= market_close_hour
    effective_day = day.where(~after_close, day + pd.Timedelta(days=1))

    td_values = trading_days.values
    idx = np.searchsorted(td_values, effective_day.values, side='left')
    idx = np.clip(idx, 0, len(td_values) - 1)
    return pd.Series(td_values[idx], index=dates.index).dt.strftime('%Y-%m-%d')


def clean_news_data(news: pd.DataFrame) -> pd.DataFrame:
    df = news.copy()
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert('America/New_York')
    trading_days = get_nasdaq_trading_days(
        start=df['Date'].min() - pd.Timedelta(days=7),
        end=df['Date'].max() + pd.Timedelta(days=7),
    )
    df['trading_day'] = assign_trading_day(df['Date'], trading_days)
    df = df.dropna(subset=['Article_title'])
    df['Article_title'] = df['Article_title'].astype(str).str.strip()
    df = df.drop_duplicates(subset=['trading_day', 'Article_title']).reset_index(drop=True)

    return df
