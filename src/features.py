import pandas as pd
from pathlib import Path

PRICE_FEATURES = ['return_1d', 'return_5d', 'return_20d',
                  'price_to_sma_20', 'volatility_5d', 'volume_ratio_10d', 'hl_spread']
VADER_FEATURES = ['vader_score', 'vader_pos', 'vader_neg', 'vader_score_3d']
ROBERTA_FEATURES = ['roberta_score', 'roberta_pos',
                    'roberta_neg', 'roberta_score_3d']
FINBERT_FEATURES = ['finbert_score', 'finbert_pos', 'finbert_neg', 'finbert_score_3d',
                    'finbert_score_7d', 'finbert_diff_1d', 'news_count_ratio_7d', 'finbert_x_volume']
CALENDAR_FEATURES = ['day_of_week', 'month']


def returns(df: pd.DataFrame):
    # считаем процентное изменение за разные сроки
    df['return_1d'] = df['Close'].pct_change(1)  # day
    df['return_5d'] = df['Close'].pct_change(5)  # week
    df['return_20d'] = df['Close'].pct_change(20)  # mth


def price_to_sma(df: pd.DataFrame):
    # делим показатель на сред арифм за выбранное окно получим ratio ?> 1.0
    sma_20 = df['Close'].rolling(window=20).mean()
    df['price_to_sma_20'] = df['Close'] / sma_20


def volatility(df: pd.DataFrame):
    # стандартное отклонение дневных доходностей за выбранное окно
    df['volatility_5d'] = df['return_1d'].rolling(window=5).std()


def volume_spike(df: pd.DataFrame):
    # чем больше объем торгов тем сильнее сигнал
    df['volume_ratio_10d'] = df['Volume'] / \
        df['Volume'].rolling(window=10).mean()


def rolling_sentiment(df: pd.DataFrame):
    # смотрим на среднее в разных окнах
    df['finbert_score_3d'] = df['finbert_score'].rolling(window=3).mean()
    df['finbert_score_7d'] = df['finbert_score'].rolling(window=7).mean()
    df['roberta_score_3d'] = df['roberta_score'].rolling(window=3).mean()
    df['vader_score_3d'] = df['vader_score'].rolling(window=3).mean()


def sentiment_delta(df: pd.DataFrame):
    # изменение настроений за день
    df['finbert_diff_1d'] = df['finbert_score'] - df['finbert_score'].shift(1)


def media_attention(df: pd.DataFrame):
    # количество новостей относительно среднего за последнее окно
    df['news_count_ratio_7d'] = df['news_count'] / \
        df['news_count'].rolling(window=7).mean()
    df['news_count_ratio_7d']=df['news_count_ratio_7d'].fillna(0)


def cal_anomalies(df: pd.DataFrame):
    # "оптимизм понедельника" и тд
    dates = pd.to_datetime(df['Date'])
    df['day_of_week'] = dates.dt.dayofweek
    df['month'] = dates.dt.month


def hl_spread(df: pd.DataFrame):
    # размах изменения в течение дня
    df['hl_spread'] = (df['High']-df['Low'])/df['Close']


def finbert_x_volume(df: pd.DataFrame):
    df['finbert_x_volume'] = df['finbert_score']*df['volume_ratio_10d']

def flag_has_news(df:pd.DataFrame):
    df['has_news']=(df['news_count'] > 0).astype(int)# не знала как это пишется прикольно
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values('Date').reset_index(drop=True)
    returns(df)
    price_to_sma(df)
    volatility(df)
    volume_spike(df)
    rolling_sentiment(df)
    sentiment_delta(df)
    media_attention(df)
    cal_anomalies(df)
    hl_spread(df)
    finbert_x_volume(df)
    flag_has_news(df)
    df = df.dropna().reset_index(drop=True)
    return df


if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    input_path = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
    output_path = PROJECT_ROOT / "data" / "processed" / "nvda_features.csv"
    df = pd.read_csv(input_path)
    df_featured = create_features(df)
    df_featured.to_csv(output_path, index=False)
    print(f"dataset ready {df_featured.shape}\n")
    print("\n features list:", df_featured.columns.tolist())
