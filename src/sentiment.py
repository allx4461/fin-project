import gc
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))


# region sentiment berts
def predict_finbert_or_roberta(texts: list[str], tokenizer, model, batch_size: int = 32) -> pd.DataFrame:
    all_pos = []
    all_neg = []
    all_neu = []
    all_scores = []


    label2id = {v.lower(): k for k, v in model.config.id2label.items()}
    pos_idx = label2id['positive']
    neg_idx = label2id['negative']
    neu_idx = label2id['neutral']

    for i in tqdm(range(0, len(texts), batch_size), desc="Inferencing"):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)
        pos_batch = probs[:, pos_idx].cpu().numpy()
        neg_batch = probs[:, neg_idx].cpu().numpy()
        neu_batch = probs[:, neu_idx].cpu().numpy()
        score_batch = pos_batch - neg_batch

        all_pos.extend(pos_batch)
        all_neg.extend(neg_batch)
        all_neu.extend(neu_batch)
        all_scores.extend(score_batch)

    return pd.DataFrame({
        'prob_pos': all_pos,
        'prob_neg': all_neg,
        'prob_neu': all_neu,
        'sentiment_score': all_scores
    })


def predict_finbert(texts: list[str], batch_size: int = 32) -> pd.DataFrame:
    print(f"Loading FinBERT on {device}...")
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    df_res = predict_finbert_or_roberta(texts, tokenizer=tokenizer, model=model, batch_size=batch_size)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return df_res


def predict_roberta(texts: list[str], batch_size: int = 32) -> pd.DataFrame:
    print(f"Loading RoBERTa on {device}...")
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    df_res = predict_finbert_or_roberta(texts, tokenizer=tokenizer, model=model, batch_size=batch_size)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return df_res
# endregion


# region vader
nltk.download('vader_lexicon', quiet=True)


def predict_vader(texts: list[str]) -> pd.DataFrame:
    all_pos = []
    all_neg = []
    all_neu = []
    all_scores = []
    sia = SentimentIntensityAnalyzer()

    for text in tqdm(texts, desc="VADER sentiment"):
        scores = sia.polarity_scores(str(text))
        all_pos.append(scores['pos'])
        all_neg.append(scores['neg'])
        all_neu.append(scores['neu'])
        all_scores.append(scores['compound'])

    return pd.DataFrame({
        'prob_pos': all_pos,
        'prob_neg': all_neg,
        'prob_neu': all_neu,
        'sentiment_score': all_scores
    })
# endregion
