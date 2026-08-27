from pathlib import Path
import sys
import yfinance as yf
import pandas as pd
import datetime as dt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import TICKER, START_DATE, END_DATE, DATA_PROCESSED_DATASET, get_sentiment_file


def load_yfinance() -> pd.DataFrame:
    prices = yf.download(TICKER, start=START_DATE, end=END_DATE).reset_index()
    prices["Date"] = pd.to_datetime(prices["Date"]).dt.strftime('%Y-%m-%d')
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    # pct_change = percent change, shift(-1) поднимает на строку вверх (доходность следующего дня)
    prices["target_return"] = prices['Close'].pct_change().shift(-1)
    prices['target_direction'] = (prices['target_return'] > 0).astype(int)
    return prices


def load_merge_all_sentiments() -> pd.DataFrame:
    df_finbert = pd.read_csv(get_sentiment_file("finbert")).rename(
        columns={
            'sentiment_score': 'finbert_score',
            'prob_pos': 'finbert_pos',
            'prob_neg': 'finbert_neg',
            'news_count': 'news_count'
        }
    )
    df_roberta = pd.read_csv(get_sentiment_file("roberta")).rename(
        columns={
            'sentiment_score': 'roberta_score',
            'prob_pos': 'roberta_pos',
            'prob_neg': 'roberta_neg',
            'news_count': 'roberta_count'
        }
    )[['trading_day', 'roberta_score', 'roberta_pos', 'roberta_neg']]

    df_vader = pd.read_csv(get_sentiment_file("vader")).rename(
        columns={
            'sentiment_score': 'vader_score',
            'prob_pos': 'vader_pos',
            'prob_neg': 'vader_neg',
            'news_count': 'vader_count'
        }
    )[['trading_day', 'vader_score', 'vader_pos', 'vader_neg']]

    all_sentiment = df_finbert.merge(
        df_roberta, on="trading_day"
    ).merge(df_vader, on="trading_day")
    return all_sentiment


def merge_sentiment_yfinance(prices: pd.DataFrame, all_sentiment: pd.DataFrame) -> pd.DataFrame:
    final_df = prices.merge(all_sentiment, left_on='Date', right_on='trading_day', how='left')
    final_df['news_count'] = final_df['news_count'].fillna(0)
    for mod in ['finbert', 'roberta', 'vader']:
        for dif in ['score', 'pos', 'neg']:
            final_df[f'{mod}_{dif}'] = final_df[f'{mod}_{dif}'].fillna(0.0)
    final_df = final_df.dropna().reset_index(drop=True)
    return final_df


def main(output_path=DATA_PROCESSED_DATASET):
    prices = load_yfinance()
    all_sentiments = load_merge_all_sentiments()
    final = merge_sentiment_yfinance(prices, all_sentiments)
    final.to_csv(output_path, index=False)
    print(f"Saved merged dataset for {TICKER} to {output_path}")


if __name__ == '__main__':
    main(DATA_PROCESSED_DATASET)
