from pathlib import Path
import sys
from typing import Literal
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import TICKER, DATA_PROCESSED_NEWS, DATA_SENTIMENT_DIR, get_sentiment_file
from src.sentiment import predict_finbert, predict_roberta, predict_vader


def run_sentiment(data_path: Path, output_dir: Path, model: Literal['finbert', 'roberta', 'vader']):
    print(f"\n--- Running sentiment analysis: {model} for {TICKER} ---")
    df = pd.read_csv(data_path, usecols=['trading_day', 'Article_title'])
    
    if model == 'finbert':
        sentiment_df = predict_finbert(df['Article_title'].tolist())
    elif model == 'roberta':
        sentiment_df = predict_roberta(df['Article_title'].tolist())
    elif model == 'vader':
        sentiment_df = predict_vader(df['Article_title'].tolist())
    else:
        return None

    df['prob_pos'] = sentiment_df['prob_pos']
    df['prob_neg'] = sentiment_df['prob_neg']
    df['prob_neu'] = sentiment_df['prob_neu']
    df['sentiment_score'] = sentiment_df['sentiment_score']

    daily_sentiment = df.groupby('trading_day').agg(
        news_count=('Article_title', 'count'),
        sentiment_score=('sentiment_score', 'mean'),
        prob_pos=('prob_pos', 'mean'),
        prob_neg=('prob_neg', 'mean'),
        prob_neu=('prob_neu', 'mean')
    ).reset_index()

    out_file = get_sentiment_file(model)
    daily_sentiment.to_csv(out_file, index=False)
    print(f"Saved {model}'s {len(daily_sentiment)} trading days to {out_file}")


def check_sentiment(ticker: str = TICKER):
    for model_name in ['finbert', 'roberta', 'vader']:
        file_path = get_sentiment_file(model_name, ticker=ticker)
        if file_path.exists():
            print(f"\n\t{model_name}'s results for {ticker}\t")
            df = pd.read_csv(file_path)
            print(df.head(5))


if __name__ == '__main__':
    run_sentiment(DATA_PROCESSED_NEWS, DATA_SENTIMENT_DIR, 'vader')
    run_sentiment(DATA_PROCESSED_NEWS, DATA_SENTIMENT_DIR, 'finbert')
    run_sentiment(DATA_PROCESSED_NEWS, DATA_SENTIMENT_DIR, 'roberta')
    check_sentiment(TICKER)