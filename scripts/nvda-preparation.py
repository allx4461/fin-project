from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data import filter_ticker_streaming, clean_news_data

DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "nasdaq_exteral_data.csv"
OUTPUT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "nvda_news.csv"


def main():
    print(f"1 - filtrate NVDA from {DATA_RAW_PATH}...")
    filter_ticker_streaming(DATA_RAW_PATH, OUTPUT_PROCESSED_PATH, ticker="NVDA")

    print("2 - load filtrated")
    df_nvda = pd.read_csv(OUTPUT_PROCESSED_PATH)
    print(f"   got {len(df_nvda)} NVDA-related rows")

    print("3 - get rid of duplicates and Nans, corrected time")
    
    df_cleaned = clean_news_data(df_nvda)
    print(f"   got {len(df_cleaned)} NVDA-related clear rows")

    print(f"4 - {OUTPUT_PROCESSED_PATH} contains clean NVDA data")
    df_cleaned.to_csv(OUTPUT_PROCESSED_PATH, index=False)
    print("success")


if __name__ == "__main__":
    main()