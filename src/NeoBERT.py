import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
import re


# --- SMAPE Loss ---
class SMAPELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, y_pred, y_true):
        y_true_real = torch.expm1(y_true)
        y_pred_real = torch.expm1(y_pred)
        num = torch.abs(y_pred_real - y_true_real)
        den = (torch.abs(y_true_real) + torch.abs(y_pred_real)) / 2.0
        return torch.mean(num / (den + self.eps))


# --- Model ---
class PriceModel(nn.Module):
    def __init__(self, base_model_name, freeze_encoder=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
        hid = self.encoder.config.hidden_size
        self.regressor = nn.Linear(hid, 1)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        preds = self.regressor(cls).squeeze(-1)
        loss = None
        if labels is not None:
            loss = SMAPELoss()(preds, labels)
        return (loss, preds) if loss is not None else preds


# --- Dataset prep ---
def prepare_datasets(csv_path, data_percentage=1.0, val_frac=0.1, seed=42):
    df = pd.read_csv(csv_path)
    df['catalog_content'] = df['catalog_content'].astype(str)
    df['catalog_content'] = df['catalog_content'].apply(lambda x: re.sub(r'<.*?>', '', x))
    df['catalog_content'] = df['catalog_content'].apply(lambda x: ' '.join(x.split()))
    df = df.sample(frac=data_percentage, random_state=seed).reset_index(drop=True)
    df["labels"] = np.log1p(df["price"].values)
    train_df, val_df = train_test_split(df, test_size=val_frac, random_state=seed)
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)


def tokenize_batch(batch, tokenizer, max_length=512):
    toks = tokenizer(batch["catalog_content"], padding="max_length", truncation=True, max_length=max_length)
    return {
        "input_ids": toks["input_ids"],
        "attention_mask": toks["attention_mask"],
        "labels": batch["labels"],
    }


# --- Metric ---
def compute_metrics(pred):
    preds = pred.predictions
    if isinstance(preds, tuple):
        preds = preds[1]
    preds = preds.reshape(-1)
    labels = pred.label_ids.reshape(-1)
    preds_real = np.expm1(preds)
    labels_real = np.expm1(labels)
    denom = (np.abs(labels_real) + np.abs(preds_real)) / 2.0 + 1e-8
    smape = np.mean(np.abs(preds_real - labels_real) / denom)
    return {"smape": float(smape)}


# --- Main ---
def run(params):
    train_ds, val_ds = prepare_datasets(
        os.path.join(params["DATASET_FOLDER"], "train_filtered_cleaned.csv"),
        data_percentage=params["DATA_PERCENTAGE"],
        val_frac=0.1,
    )

    tokenizer = AutoTokenizer.from_pretrained(params["MODEL_NAME"], trust_remote_code=True)
    train_ds = train_ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True, remove_columns=val_ds.column_names)

    model = PriceModel(params["MODEL_NAME"], freeze_encoder=params["FREEZE_EMBEDDING_MODEL"])

    args = TrainingArguments(
        output_dir=params["OUTPUT_DIR"],
        per_device_train_batch_size=params["BATCH_SIZE"],
        per_device_eval_batch_size=params["BATCH_SIZE"],
        learning_rate=params["LEARNING_RATE"],
        num_train_epochs=params["NUM_EPOCHS"],
        weight_decay=params["WEIGHT_DECAY"],
        logging_strategy="steps",
        logging_steps=100,
        bf16=True
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to=["none"]
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(params["OUTPUT_DIR"])

    # Inference on test.csv if present
    test_path = os.path.join(params["DATASET_FOLDER"], "test.csv")
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        test_ds = Dataset.from_pandas(test_df)
        test_ds = test_ds.map(lambda b: tokenizer(b["catalog_content"], padding="max_length", truncation=True, max_length=512), batched=True)
        preds = trainer.predict(test_ds).predictions
        if isinstance(preds, tuple):
            preds = preds[1]
        preds = preds.reshape(-1)
        final = np.expm1(preds)
        submission = pd.DataFrame({"sample_id": test_df["sample_id"], "price": final})
        os.makedirs("submission", exist_ok=True)
        fname = f"submission/submission_{int(time.time())}.csv"
        submission.to_csv(fname, index=False)
        print("Saved submission:", fname)


if __name__ == "__main__":
    params = {
        "RUN_NAME": "Neo-BERT",
        "DATASET_FOLDER": "dataset",
        "MODEL_NAME": "chandar-lab/NeoBERT",
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 2e-5,
        "NUM_EPOCHS": 3,
        "DATA_PERCENTAGE": 1.0,
        "FREEZE_EMBEDDING_MODEL": False,
        "MIXED_PRECISION": True,
        "WEIGHT_DECAY": 0.01,
        "OUTPUT_DIR": "neo_trainer_out",
    }
    run(params)
