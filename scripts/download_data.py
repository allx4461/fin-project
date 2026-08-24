import yfinance as yf
import pandas as pd
import datetime as dt


def load_yfinance() -> pd.DataFrame:
    prices = yf.download("NVDA", start="2015-01-01",
                         end="2024-01-01").reset_index()#дата как новый индекс
    prices["Date"] = pd.to_datetime(prices["Date"]).dt.strftime('%Y-%m-%d')
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    # pct_chande = percent change, shift поднимает на строку вверх
    prices["target_return"] = prices['Close'].pct_change().shift(-1)
    prices['target_direction'] = (prices['target_return'] > 0).astype(
        int)  # это для классификации, а выше для регрессии
    return prices


def load_merge_all_sentiments() -> pd.DataFrame:
    df_finbert = pd.read_csv("data/sentiment/nvda_finbert_daily.csv").rename(
        columns={'sentiment_score': 'finbert_score',
                 'prob_pos': 'finbert_pos',
                 'prob_neg': 'finbert_neg',
                 'news_count': 'news_count'
                 })
    df_roberta = pd.read_csv("data/sentiment/nvda_roberta_daily.csv").rename(
        columns={'sentiment_score': 'roberta_score',
                 'prob_pos': 'roberta_pos',
                 'prob_neg': 'roberta_neg',
                 'news_count': 'roberta_count'
                 # оставляем только эти колонки тк колво новостей одинаково и уже есть у финберта
                 })[['trading_day', 'roberta_score', 'roberta_pos', 'roberta_neg']]
    df_vader = pd.read_csv("data/sentiment/nvda_vader_daily.csv").rename(
        columns={'sentiment_score': 'vader_score',
                 'prob_pos': 'vader_pos',
                 'prob_neg': 'vader_neg',
                 'news_count': 'vader_count'
                 })[['trading_day', 'vader_score', 'vader_pos', 'vader_neg']]
    all_sentiment = df_finbert.merge(
        df_roberta, on="trading_day").merge(df_vader, on="trading_day")
    return all_sentiment


def merge_sentiment_yfinance(prices: pd.DataFrame, all_sentiment: pd.DataFrame) -> pd.DataFrame:
    final_df = prices.merge(all_sentiment, left_on='Date',
                            right_on='trading_day', how='left')
    final_df['news_count']=final_df['news_count'].fillna(0)
    for mod in ['finbert','roberta','vader']:
        for dif in ['score','pos','neg','neu']:
            final_df[f'{mod}_{dif}']=final_df[f'{mod}_{dif}'].fillna(0.0)
    final_df = final_df.dropna().reset_index(drop=True)#когда выкидываются наны сбивается индекс, поэтому ресетим
    return final_df


def main(output_path) -> pd.DataFrame:
    prices = load_yfinance()
    all_sentiments = load_merge_all_sentiments()
    final = merge_sentiment_yfinance(prices, all_sentiments)
    final.to_csv(output_path,index=False)#не нужна лишняя колонка с индексами


if __name__ == '__main__':
    main('data/processed/dataset.csv')
