#!/usr/bin/env python3
# text_only_experiment_fixed.py
"""
Fixed text-only training script: avoids shape mismatch when batch size == 1
and makes validation concatenation robust. Otherwise same behavior as before.
"""

import os
import time
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import mlflow

# -------------------------
# Regressor
# -------------------------
class Regressor(nn.Module):
    def __init__(self, input_dim, width=256, depth=2, dropout=0.25):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)  # return shape (B,1) or (B,) depending on input

# -------------------------
# Dataset & vectorized collate
# -------------------------
class TextPriceDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row.get('catalog_content', "") or "")
        price = float(row['price'])  # should already be log1p
        sample_id = row.get('sample_id', idx)
        return {'text': text, 'price': price, 'sample_id': sample_id}

def make_collate_fn(tokenizer, max_length=512):
    def collate_fn(batch):
        texts = [b['text'] for b in batch]
        prices = torch.tensor([b['price'] for b in batch], dtype=torch.float)
        sample_ids = [b['sample_id'] for b in batch]
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'price': prices,
            'sample_id': sample_ids
        }
    return collate_fn

# -------------------------
# SMAPE helper (percent)
# -------------------------
def smape_percent_from_log(pred_log, true_log, eps=1e-8):
    pred = np.expm1(pred_log)
    true = np.expm1(true_log)
    num = np.abs(pred - true)
    den = (np.abs(pred) + np.abs(true)) / 2.0 + eps
    return 100.0 * np.mean(num / den)

# -------------------------
# Train / Eval
# -------------------------
def train_text_model(params):
    mlflow.set_experiment(params.get('MLFLOW_EXPERIMENT', 'TextOnly_Experiments'))
    with mlflow.start_run(run_name=params['RUN_NAME']):
        mlflow.log_params(params)

        # load csv
        csv_path = os.path.join(params['DATASET_FOLDER'], params['TRAIN_CSV'])
        df = pd.read_csv(csv_path)
        # Ensure price numeric and apply log1p
        df['price'] = df['price'].astype(float)
        df['price'] = np.log1p(df['price'])

        if params['DATA_PERCENTAGE'] < 1.0:
            df = df.sample(frac=params['DATA_PERCENTAGE'], random_state=params['SEED']).reset_index(drop=True)

        train_df, val_df = train_test_split(df, test_size=params['VAL_FRAC'], random_state=params['SEED'])

        # tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(params['MODEL_NAME'], use_fast=True, trust_remote_code=True)
        text_backbone = AutoModel.from_pretrained(params['MODEL_NAME'], trust_remote_code=True)

        # mean pooling helper
        def mean_pool(last_hidden_state, mask):
            mask = mask.unsqueeze(-1).float()  # (B, T, 1)
            summed = (last_hidden_state * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            return summed / counts

        # attach regressor
        hidden_dim = text_backbone.config.hidden_size
        regressor = Regressor(hidden_dim, width=params['REGRESSOR_WIDTH'], depth=params['REGRESSOR_DEPTH'], dropout=params['DROPOUT_RATE'])

        # combine into single container
        model = nn.Module()
        model.text_backbone = text_backbone
        model.regressor = regressor

        # freeze backbone if requested (start with frozen for better generalization)
        if not params.get('UNFREEZE_TEXT', False):
            for p in model.text_backbone.parameters():
                p.requires_grad = False

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # datasets and dataloaders
        train_ds = TextPriceDataset(train_df)
        val_ds = TextPriceDataset(val_df)
        collate_fn = make_collate_fn(tokenizer, max_length=params['MAX_TEXT_LEN'])

        train_loader = DataLoader(train_ds, batch_size=params['BATCH_SIZE'], shuffle=True,
                                  num_workers=params['NUM_WORKERS'], pin_memory=True,
                                  persistent_workers=True, prefetch_factor=params.get('PREFETCH_FACTOR', 2),
                                  collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=params['BATCH_SIZE'], shuffle=False,
                                num_workers=max(1, params['NUM_WORKERS']//2), pin_memory=True,
                                collate_fn=collate_fn)

        # optimizer and scheduler (only trainable params)
        trainable = [p for p in model.parameters() if p.requires_grad]
        print("Trainable params:", sum(p.numel() for p in trainable))
        optimizer = AdamW(trainable, lr=params['LEARNING_RATE'], weight_decay=params['WEIGHT_DECAY'])

        total_steps = math.ceil(len(train_loader) * params['NUM_EPOCHS'] / max(1, params.get('ACCUM_STEPS', 1)))
        warmup_steps = int(total_steps * params.get('WARMUP_PROPORTION', 0.03))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        # mixed precision
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        use_fp16 = params.get('USE_FP16', True)

        loss_fn = nn.SmoothL1Loss()

        best_val_smape = float('inf')
        patience = 0

        for epoch in range(1, params['NUM_EPOCHS'] + 1):
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            train_preds = []
            train_trues = []

            pbar = tqdm(train_loader, desc=f"Train E{epoch}/{params['NUM_EPOCHS']}", leave=False)
            optimizer.zero_grad()
            for step, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                prices = batch['price'].to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_fp16, dtype=torch.float16):
                    outputs = model.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
                    last_hidden = outputs.last_hidden_state  # (B, T, D)
                    pooled = mean_pool(last_hidden, attention_mask)  # (B, D)
                    preds = model.regressor(pooled)  # could be shape (B,1) or (B,)
                    # normalize preds to 1-D tensor of shape (B,)
                    if preds.dim() == 0:
                        preds = preds.unsqueeze(0)
                    else:
                        preds = preds.view(preds.size(0), -1).squeeze(-1)

                    loss = loss_fn(preds, prices) / max(1, params.get('ACCUM_STEPS', 1))

                scaler.scale(loss).backward()

                if (step + 1) % max(1, params.get('ACCUM_STEPS', 1)) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, params.get('GRAD_CLIP_NORM', 1.0))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                epoch_loss += float(loss.item() * max(1, params.get('ACCUM_STEPS', 1)))
                batch_count += 1
                # ensure numpy arrays are 1-D before appending
                train_preds.append(np.atleast_1d(preds.detach().cpu().numpy()))
                train_trues.append(np.atleast_1d(prices.detach().cpu().numpy()))

            avg_train_loss = epoch_loss / max(1, batch_count)
            # compute train SMAPE robustly
            if train_preds:
                train_preds_arr = np.concatenate(train_preds, axis=0)
                train_trues_arr = np.concatenate(train_trues, axis=0)
                train_smape = smape_percent_from_log(train_preds_arr, train_trues_arr)
            else:
                train_smape = float('nan')

            mlflow.log_metric('train_loss', avg_train_loss, step=epoch)
            mlflow.log_metric('train_smape_percent', train_smape, step=epoch)

            # Validation
            model.eval()
            val_preds = []
            val_trues = []
            val_losses = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Val E{epoch}", leave=False):
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                    prices = batch['price'].to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_fp16, dtype=torch.float16):
                        outputs = model.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
                        last_hidden = outputs.last_hidden_state
                        pooled = mean_pool(last_hidden, attention_mask)
                        preds = model.regressor(pooled)
                        if preds.dim() == 0:
                            preds = preds.unsqueeze(0)
                        else:
                            preds = preds.view(preds.size(0), -1).squeeze(-1)
                        vloss = loss_fn(preds, prices)
                    val_losses.append(float(vloss.item()))
                    val_preds.append(np.atleast_1d(preds.detach().cpu().numpy()))
                    val_trues.append(np.atleast_1d(prices.detach().cpu().numpy()))

            avg_val_loss = float(np.mean(val_losses)) if val_losses else float('nan')
            if val_preds:
                try:
                    val_preds_arr = np.concatenate(val_preds, axis=0)
                    val_trues_arr = np.concatenate(val_trues, axis=0)
                    val_smape = smape_percent_from_log(val_preds_arr, val_trues_arr)
                except Exception as e:
                    print("Warning concatenation failed:", e)
                    val_smape = float('nan')
            else:
                val_smape = float('nan')

            print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, train_smape%={train_smape:.2f}, val_loss={avg_val_loss:.4f}, val_smape%={val_smape:.2f}")
            mlflow.log_metric('val_loss', avg_val_loss, step=epoch)
            mlflow.log_metric('val_smape_percent', val_smape, step=epoch)

            # checkpoint by val_smape
            if not math.isnan(val_smape) and val_smape < best_val_smape:
                best_val_smape = val_smape
                patience = 0
                os.makedirs(params['OUTPUT_DIR'], exist_ok=True)
                ckpt = os.path.join(params['OUTPUT_DIR'], f'best_text_{int(time.time())}.pt')
                torch.save({'model_state': model.state_dict(), 'params': params, 'epoch': epoch, 'val_smape': val_smape}, ckpt)
                mlflow.log_artifact(ckpt)
                print("Saved best checkpoint:", ckpt)
            else:
                patience += 1
                if patience >= params.get('EARLY_STOPPING_PATIENCE', 4):
                    print("Early stopping.")
                    break

        # final save
        os.makedirs(params['OUTPUT_DIR'], exist_ok=True)
        final_ckpt = os.path.join(params['OUTPUT_DIR'], f'last_text_{int(time.time())}.pt')
        torch.save({'model_state': model.state_dict(), 'params': params}, final_ckpt)
        mlflow.log_artifact(final_ckpt)
        print("Finished. final ckpt:", final_ckpt)

# -------------------------
# Default params — edit as needed
# -------------------------
if __name__ == "__main__":
    params = {
        'RUN_NAME': 'TextOnly_NeoBERT_Baseline_fixed',
        'DATASET_FOLDER': 'dataset',
        'TRAIN_CSV': 'train_filtered_cleaned.csv',
        'BATCH_SIZE': 64,
        'NUM_EPOCHS': 12,
        'LEARNING_RATE': 3e-5,
        'WEIGHT_DECAY': 0.01,
        'ACCUM_STEPS': 1,
        'WARMUP_PROPORTION': 0.03,
        'DATA_PERCENTAGE': 1.0,
        'VAL_FRAC': 0.1,
        'REGRESSOR_WIDTH': 512,
        'REGRESSOR_DEPTH': 2,
        'DROPOUT_RATE': 0.25,
        'NUM_WORKERS': 12,
        'PREFETCH_FACTOR': 2,
        'MAX_TEXT_LEN': 512,
        'MODEL_NAME': 'chandar-lab/NeoBERT',
        'USE_FP16': True,
        'UNFREEZE_TEXT': False,
        'EARLY_STOPPING_PATIENCE': 4,
        'OUTPUT_DIR': 'text_only_out_fixed',
        'MLFLOW_EXPERIMENT': 'TextOnly_Experiments',
        'SEED': 42,
        'GRAD_CLIP_NORM': 1.0
    }

    # reproducibility small helpers
    random.seed(params['SEED'])
    np.random.seed(params['SEED'])
    torch.manual_seed(params['SEED'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params['SEED'])

    os.makedirs(params['OUTPUT_DIR'], exist_ok=True)
    train_text_model(params)
