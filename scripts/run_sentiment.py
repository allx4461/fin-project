from pathlib import Path
import sys
from typing import Literal
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.sentiment import predict_finbert, predict_roberta, predict_vader

DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "nvda_news.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "sentiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_sentiment(data_path: Path, output_path: Path, model: Literal['finbert', 'roberta', 'vader']):
    print(f"\n--- Running sentiment analysis: {model} ---")
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

    out_file = output_path / f"nvda_{model}_daily.csv"
    daily_sentiment.to_csv(out_file, index=False)
    print(f"saved {model}'s {len(daily_sentiment)} trading days to {out_file}")


def check_sentiment(path: Path):
    for model_name in ['finbert', 'roberta', 'vader']:
        file_path = path / f"nvda_{model_name}_daily.csv"
        if file_path.exists():
            print(f"\n\t{model_name}'s results \t")
            df = pd.read_csv(file_path)
            print(df.head(5))


if __name__ == '__main__':
    run_sentiment(DATA_PROCESSED_PATH, OUTPUT_DIR, 'vader')

    run_sentiment(DATA_PROCESSED_PATH, OUTPUT_DIR, 'finbert')

    run_sentiment(DATA_PROCESSED_PATH, OUTPUT_DIR, 'roberta')

    check_sentiment(OUTPUT_DIR)