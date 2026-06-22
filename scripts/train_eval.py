"""Local fine-tune + evaluate, mirroring the Colab starter notebook.

Runs DistilBERT fine-tuning on data/takemeter_dataset.csv with corrected
hyperparameters, evaluates on the test split, and computes the zero-shot Groq
baseline from the suggested_label column captured during pre-labeling
(data/reviewed.csv) on the same test split. Writes outputs/ artifacts.
"""
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, set_seed,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

set_seed(42)

LABEL_MAP = {"analysis": 0, "hot_take": 1, "reaction": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)
NAMES = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
MODEL_NAME = "distilbert-base-uncased"

# ---- Section 1/2: load + split (70/15/15, stratified, seed 42) ----
df = pd.read_csv("data/takemeter_dataset.csv")
df["label_id"] = df["label"].map(LABEL_MAP)
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["label_id"])
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"])
for d in (train_df, val_df, test_df):
    d.reset_index(drop=True, inplace=True)
print(f"train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(ex):
    return tok(ex["text"], truncation=True, max_length=256)


def make(d):
    ds = Dataset.from_pandas(
        d[["text", "label_id"]].rename(columns={"label_id": "labels"}))
    return ds.map(tokenize, batched=True)


train_ds, val_ds, test_ds = make(train_df), make(val_df), make(test_df)
collator = DataCollatorWithPadding(tokenizer=tok)


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1_macro": f1_score(p.label_ids, preds, average="macro"),
    }


# ---- Section 3: fine-tune (corrected hyperparameters) ----
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS)
args = TrainingArguments(
    output_dir="./takemeter-model",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=10,
    report_to="none",
    seed=42,
)
trainer = Trainer(
    model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=collator, processing_class=tok, compute_metrics=compute_metrics,
)
trainer.train()

# ---- Section 4: evaluate fine-tuned on test ----
out = trainer.predict(test_ds)
ft_pred = np.argmax(out.predictions, axis=-1)
ft_true = out.label_ids
ft_probs = torch.nn.functional.softmax(
    torch.tensor(out.predictions), dim=-1).numpy()
ft_acc = accuracy_score(ft_true, ft_pred)
ft_f1 = f1_score(ft_true, ft_pred, average="macro")
print("\n=== FINE-TUNED ===")
print(f"accuracy {ft_acc:.3f} | macro-F1 {ft_f1:.3f}")
print(classification_report(ft_true, ft_pred, target_names=NAMES, zero_division=0))
cm_ft = confusion_matrix(ft_true, ft_pred)
print("confusion (rows=true, cols=pred):", NAMES)
print(cm_ft)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_ft, display_labels=NAMES)
fig, ax = plt.subplots(figsize=(7, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Fine-Tuned Model — Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)

# ---- Section 5: baseline = saved Groq zero-shot predictions ----
sugg = pd.read_csv("data/reviewed.csv")[["text", "suggested_label"]]
test_m = test_df.merge(sugg, on="text", how="left")
bl_raw = [s if s in LABEL_MAP else None for s in test_m["suggested_label"]]
valid = [(LABEL_MAP[p], t) for p, t in zip(bl_raw, test_m["label_id"]) if p]
bl_pred = [a for a, _ in valid]
bl_true = [b for _, b in valid]
bl_acc = accuracy_score(bl_true, bl_pred)
bl_f1 = f1_score(bl_true, bl_pred, average="macro")
print("\n=== BASELINE (Groq zero-shot) ===")
print(f"accuracy {bl_acc:.3f} | macro-F1 {bl_f1:.3f} "
      f"(on {len(valid)}/{len(test_m)} parseable)")
print(classification_report(bl_true, bl_pred, target_names=NAMES, zero_division=0))
cm_bl = confusion_matrix(bl_true, bl_pred)
print("confusion (rows=true, cols=pred):", NAMES)
print(cm_bl)

# ---- Section 6: export ----
results = {
    "baseline_accuracy": round(bl_acc, 4),
    "baseline_macro_f1": round(bl_f1, 4),
    "finetuned_accuracy": round(ft_acc, 4),
    "finetuned_macro_f1": round(ft_f1, 4),
    "improvement": round(ft_acc - bl_acc, 4),
    "test_set_size": len(test_df),
    "baseline_parseable": len(valid),
    "label_map": LABEL_MAP,
    "model": MODEL_NAME,
}
with open("outputs/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

# predictions table for the README
rows = []
for i in range(len(test_df)):
    rows.append({
        "text": test_df.iloc[i]["text"],
        "true": ID_TO_LABEL[ft_true[i]],
        "ft_pred": ID_TO_LABEL[ft_pred[i]],
        "ft_conf": round(float(ft_probs[i][ft_pred[i]]), 3),
        "baseline_pred": bl_raw[i] if bl_raw[i] else "(unparseable)",
    })
pd.DataFrame(rows).to_csv("outputs/predictions.csv", index=False)
print("\nWrote outputs/evaluation_results.json, confusion_matrix.png, predictions.csv")
