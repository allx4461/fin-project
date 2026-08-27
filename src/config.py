from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TICKER = "NVDA"

START_DATE = "2015-01-01"
END_DATE = "2024-01-01"

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


RANDOM_SEED = 42


SENTIMENT_MODELS = ["vader", "finbert", "roberta"]


DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_SENTIMENT_DIR = PROJECT_ROOT / "data" / "sentiment"

DATA_RAW_NEWS = DATA_RAW_DIR / "nasdaq_external_data.csv"
DATA_PROCESSED_NEWS = DATA_PROCESSED_DIR / f"{TICKER.lower()}_news.csv"
DATA_PROCESSED_DATASET = DATA_PROCESSED_DIR / f"{TICKER.lower()}_dataset.csv"
DATA_PROCESSED_FEATURES = DATA_PROCESSED_DIR / f"{TICKER.lower()}_features.csv"


def get_sentiment_file(model_name: str, ticker: str = TICKER) -> Path:
    """Возвращает путь к кэшу дневного сентимента для указанной модели и акции."""
    return DATA_SENTIMENT_DIR / f"{ticker.lower()}_{model_name}_daily.csv"


# 8. Results & Figures directories per tickeчr
RESULTS_DIR = PROJECT_ROOT / "results" / TICKER.upper()
FIGURES_DIR = RESULTS_DIR / "figures"

# Автоматически создаем нужные папки, если их еще нет
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DATA_SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)