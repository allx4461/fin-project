import pandas as pd
from pathlib import Path
import datetime
from tqdm import tqdm
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "nasdaq_exteral_data.csv"

chunks=pd.read_csv(DATA_PATH,chunksize=500_000)

def find_only_nvda(chunks)->pd.DataFrame:
    nvda_news_list=[]
    for chunk in tqdm(chunks, desc="Filtering NVDA"):
        nvda_chunk=chunk[chunk['Stock_symbol']=='NVDA']
        nvda_news_list.append(nvda_chunk)
    nvda_news=pd.concat(nvda_news_list)
    return nvda_news
def nvda_news_details(news:pd.DataFrame)->str:
    length=len(news)
    oldest=news['Date'].min()#должно сработать как мин строка
    newest=news['Date'].max()
    return f'resulted w {length} nvda news dated from {oldest} to {newest}\n '
print(nvda_news_details(find_only_nvda(chunks)))