import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

device = torch.device('cpu')  # я арендовала сервер без gpu поэтому тут так
tokenizer_finbert = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model_finbert = AutoModelForSequenceClassification.from_pretrained(
    "ProsusAI/finbert").to(device)
model_finbert.eval()  # чтоб не учился

tokenizer_roberta = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model_roberta = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest").to(device)
model_roberta.eval()  # чтоб не учился


# region sentiment berts
def predict_finbert_or_roberta(texts: list[str], tokenizer, model, batch_size: int = 64) -> pd.DataFrame:
    all_pos = []
    all_neg = []
    all_neu = []
    all_scores = []
    label2id = {v.lower(): k for k, v in model.config.id2label.items()}
    pos_idx = label2id['positive']
    neg_idx = label2id['negative']
    neu_idx = label2id['neutral']
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pos_batch = probs[:, pos_idx].cpu().numpy()
        neg_batch = probs[:, neg_idx].cpu().numpy()
        neu_batch = probs[:, neu_idx].cpu().numpy()
        score_batch = pos_batch-neg_batch
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


# endregion
# region vader
nltk.download('vader_lexicon',quiet=True)  # quiet=True это без вывода в консоль


def predict_vader(texts: list[str]) -> pd.DataFrame:
    all_pos = []
    all_neg = []
    all_neu = []
    all_scores = []
    sia = SentimentIntensityAnalyzer()
    for text in texts:
        # Output format: {'neg': 0.0, 'neu': 0.471, 'pos': 0.529, 'compound': 0.7717}
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


def predict_finbert(texts: list[str], batch_size: int = 64):
    return predict_finbert_or_roberta(texts, tokenizer=tokenizer_finbert, model=model_finbert, batch_size)


def predict_roberta(texts: list[str], batch_size: int = 64):
    return predict_finbert_or_roberta(texts, tokenizer=tokenizer_roberta, model=model_roberta, batch_size)
