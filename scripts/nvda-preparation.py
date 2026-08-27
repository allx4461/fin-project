from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import TICKER, DATA_RAW_NEWS, DATA_PROCESSED_NEWS
from src.data import filter_ticker_streaming, clean_news_data


def main():
    print(f"1 - filtering {TICKER} from {DATA_RAW_NEWS}...")
    filter_ticker_streaming(DATA_RAW_NEWS, DATA_PROCESSED_NEWS, ticker=TICKER)

    print("2 - load filtered news")
    df_news = pd.read_csv(DATA_PROCESSED_NEWS)
    print(f"   got {len(df_news)} {TICKER}-related rows")

    print("3 - clean news: remove duplicates, NaNs, align to trading day")
    df_cleaned = clean_news_data(df_news)
    print(f"   got {len(df_cleaned)} clean rows")

    print(f"4 - saving to {DATA_PROCESSED_NEWS}")
    df_cleaned.to_csv(DATA_PROCESSED_NEWS, index=False)
    print("success")


if __name__ == "__main__":
    main()