import json
import torch
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ======================
# CONFIG
# ======================
MODEL_NAME = "bert-base-uncased"
SAVE_PATH = "models/intent_classifier"

LABELS = ["robotics", "daily", "personal", "mixed"]

# ======================
# LOAD DATA
# ======================
with open("data/intent_dataset.json") as f:
    data = json.load(f)

texts = [d["text"] for d in data]
labels = [LABELS.index(d["label"]) for d in data]

dataset = Dataset.from_dict({
    "text": texts,
    "label": labels
})

# ======================
# TOKENIZER
# ======================
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

dataset = dataset.map(tokenize)

# Split for better accuracy
dataset = dataset.train_test_split(test_size=0.1)

# ======================
# MODEL
# ======================
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS)
)

# ======================
# METRICS (IMPORTANT 🔥)
# ======================
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    logits, labels = pred
    predictions = logits.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )

    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# ======================
# TRAINING SETTINGS (SAFE FOR YOUR LAPTOP)
# ======================
training_args = TrainingArguments(
    output_dir=SAVE_PATH,

    # Stable training
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=8,  # slightly higher

    evaluation_strategy="epoch",
    save_strategy="epoch",

    logging_steps=10,

    # Performance
    fp16=True,  # RTX 4050 boost
    load_best_model_at_end=True,

    # Prevent overfitting
    weight_decay=0.01,

    # Save disk
    save_total_limit=2
)

# ======================
# TRAINER
# ======================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

# ======================
# TRAIN
# ======================
trainer.train()

# ======================
# SAVE
# ======================
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("✅ BERT training complete (TOP-TIER MODE)")
