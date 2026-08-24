from pathlib import Path
import sys
import pandas as pd
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.sentiment import predict_finbert, predict_roberta, predict_vader
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "nvda_news.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "sentiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_sentiment(data_path,output_path,model: Literal['finbert','roberta','vader']):
    df = pd.read_csv(data_path, usecols=['trading_day','Article_title'])
    if model=='finbert':
        sentiment_df = predict_finbert(df['Article_title'].tolist())
    elif model=='roberta':
        sentiment_df = predict_roberta(df['Article_title'].tolist())
    elif model=='vader':
        sentiment_df = predict_vader(df['Article_title'].tolist())
    else: return None
    df['prob_pos'] = sentiment_df['prob_pos']
    df['prob_neg'] = sentiment_df['prob_neg']
    df['prob_neu'] = sentiment_df['prob_neu']
    df['sentiment_score'] = sentiment_df['sentiment_score']
    # сгруппировали по дням, и через агрегацию пояснили че делать дальше
    daily_sentiment = df.groupby('trading_day').agg(
        news_count=('Article_title', 'count'),
        sentiment_score=('sentiment_score', 'mean'),
        prob_pos=('prob_pos', 'mean'),
        prob_neg=('prob_neg', 'mean'),
        prob_neu=('prob_neu', 'mean')
    ).reset_index()  # trading day != idx
    daily_sentiment.to_csv(output_path / f"nvda_{model}_daily.csv", index=False)

if __name__=='__main__':
    run_sentiment(DATA_PROCESSED_PATH,OUTPUT_DIR,'finbert')
    run_sentiment(DATA_PROCESSED_PATH,OUTPUT_DIR,'roberta')
    run_sentiment(DATA_PROCESSED_PATH,OUTPUT_DIR,'vader')