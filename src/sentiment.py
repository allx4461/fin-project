import torch
import pandas as pd 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device=torch.device('cpu')#я арендовала сервер без gpu поэтому тут так
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model = model.to(device)
model.eval()#чтоб не учился 
print(model.config.id2label)

#tokenisation
text="NVIDIA reports record quarterly revenue and booming AI demand"
inputs = tokenizer(
        text, 
        return_tensors="pt",       
        padding=True,              
        truncation=True,          
        max_length=128
    )
#get raw logits [positive, negative, neutral] 
with torch.no_grad():# не учись брат и так все знаешь 
    outputs = model(**inputs)
#get proobability (0%-100%)
probs = torch.softmax(outputs.logits, dim=1)

prob_pos = probs[0][0].item()
prob_neg = probs[0][1].item()
prob_neu = probs[0][2].item()

sentiment_score = prob_pos - prob_neg

print(f'\nphrase{text} got raw outputs {outputs} \n overall sentiment score {sentiment_score}')