"""Classify NBA comments with the fine-tuned TakeMeter model.

Usage:
  python scripts/predict.py                  # runs the built-in demo set
  python scripts/predict.py "your comment"   # classify your own text
"""
import glob
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ID_TO_LABEL = {0: "analysis", 1: "hot_take", 2: "reaction"}

ckpts = glob.glob("takemeter-model/checkpoint-*")
CKPT = max(ckpts, key=lambda p: int(p.rsplit("-", 1)[-1])) if ckpts else "takemeter-model"

tok = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForSequenceClassification.from_pretrained(CKPT)
model.eval()

DEMO = [
    "THANK THE BASKETBALL GODS",
    "Holy choke of the decade",
    "I think he was more athletic in Miami probably too. He moved similarly to "
    "this but was just so damn big. I’ve never been able to find a legit "
    "number, but he had to have been at least 260 minimum. Even the freakiest "
    "of the freaky athletes do not have that explosiveness at 20lbs less than that",
    "Best pure athlete across all of sports history tbh.",
    "Bruh Bron, especially at Miami, was a 1-5 player. He could play and guard "
    "from 1-5 on the court. Wilt could’ve never played the point like that. "
    "He’s hand eye coordination and feet movement puts him above Wilt. Bron "
    "pretty much had Dwight Howard’s strength too lol. He’s the ultimate athlete.",
]

texts = sys.argv[1:] or DEMO
print(f"model: {CKPT}\n")
for t in texts:
    enc = tok(t, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0]
    i = int(probs.argmax())
    print(f"{ID_TO_LABEL[i]:>9}  conf {probs[i]:.2f}   {t[:75]}")
