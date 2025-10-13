#!/usr/bin/env python3
# badhiya_oneclick_patch.py
"""
One-click patch: augmentations + weight_decay=0.05 + dropout=0.4 + lr=1e-5 + optional MixUp.
Vectorized collate, FP16 training, SMAPE logging.
"""

import os
import time
import math
import re
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, CLIPProcessor, CLIPModel, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import mlflow
import urllib.parse
from torchvision import transforms

# H100-friendly
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

# -------------------------
# Dataset: returns raw text + padded PIL image + price (log1p already applied outside)
# -------------------------
class PriceImageDataset(Dataset):
    def __init__(self, dataframe, image_dir, target_size=224):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.target_size = target_size

    def __len__(self):
        return len(self.dataframe)

    def load_image(self, image_link):
        if not pd.notna(image_link):
            return Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        try:
            filename = os.path.basename(urllib.parse.urlparse(image_link).path)
            image_path = os.path.join(self.image_dir, filename)
            if os.path.exists(image_path):
                im = Image.open(image_path).convert("RGB")
            else:
                im = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        except Exception:
            im = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        # pad to square while preserving aspect ratio (no stretching)
        w, h = im.size
        max_side = max(w, h, self.target_size)
        im = ImageOps.pad(im, (max_side, max_side), color=(0,0,0), centering=(0.5, 0.5))
        return im

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = str(row.get('catalog_content', "") or "")
        price = float(row['price'])
        image_link = row.get('image_link', None)
        image = self.load_image(image_link)
        sample_id = row.get('sample_id', idx)
        return {'text': text, 'image': image, 'price': price, 'sample_id': sample_id}

# -------------------------
# Make collate function (vectorized tokenization + clip processing)
#    - train_transforms: PIL-level transforms applied to images before CLIPProcessor
#    - post_erase: RandomErasing applied to produced pixel_values (per-sample)
# -------------------------
def make_collate_fn(tokenizer, clip_processor, max_length=512, image_size=224, train=False, train_transforms=None, random_erasing_prob=0.0):
    reaserter = transforms.RandomErasing(p=random_erasing_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3)) if random_erasing_prob > 0.0 else None

    def collate_fn(batch):
        texts = [b['text'] for b in batch]
        images = [b['image'] for b in batch]
        prices = torch.tensor([b['price'] for b in batch], dtype=torch.float)
        sample_ids = [b['sample_id'] for b in batch]

        # PIL-level augmentations (train only)
        if train and train_transforms is not None:
            images_aug = [train_transforms(img) for img in images]  # train_transforms must return PIL after RandomResizedCrop etc.
            # but torchvision transforms return PIL or Tensor depending; ensure images_aug are PIL
            # If any transform returned tensors, cast back would be hard; we keep transforms that output PIL
            images_to_process = images_aug
        else:
            images_to_process = images

        # Tokenize batch
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

        # Use CLIPProcessor to convert PIL images -> pixel_values (handles resizing/normalization)
        processed = clip_processor(images=images_to_process, return_tensors='pt')
        pixel_values = processed['pixel_values']  # (B, 3, H, W)

        # Apply RandomErasing (tensor-level) sample-wise if requested
        if train and reaserter is not None:
            # reaserter expects 3D tensor CHW per sample; apply per sample
            pv_list = []
            for i in range(pixel_values.size(0)):
                pv = pixel_values[i]
                # RandomErasing expects values ~ [0,1] or normalized; it's ok to apply on normalized tensors
                pv_er = reaserter(pv)
                pv_list.append(pv_er.unsqueeze(0))
            pixel_values = torch.cat(pv_list, dim=0)

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'pixel_values': pixel_values,
            'price': prices,
            'sample_id': sample_ids
        }
    return collate_fn

# -------------------------
# Simple MixUp on pixel_values and prices (log space)
# -------------------------
def mixup_batch(pixel_values, prices, alpha=0.3):
    if alpha <= 0:
        return pixel_values, prices
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure >=0.5 for stability (optional)
    idx = torch.randperm(pixel_values.size(0))
    pv_mix = lam * pixel_values + (1 - lam) * pixel_values[idx]
    price_mix = lam * prices + (1 - lam) * prices[idx]
    return pv_mix, price_mix

# -------------------------
# Model & fusion (same structure, dropout parameter passed in)
# -------------------------
class Regressor(nn.Module):
    def __init__(self, input_dim, width, depth, dropout_rate):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(width, 1))
        self.regressor = nn.Sequential(*layers)

    def forward(self, x):
        return self.regressor(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, text_embed_dim, image_embed_dim, output_dim, num_heads=8):
        super().__init__()
        self.image_projection = nn.Linear(image_embed_dim, text_embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=text_embed_dim, num_heads=num_heads, batch_first=True)
        self.output_layer = nn.Linear(text_embed_dim, output_dim)

    def forward(self, text_features, image_features):
        image_features_projected = self.image_projection(image_features)
        attn_output, _ = self.attention(query=text_features, key=image_features_projected, value=image_features_projected)
        cls_attended_output = attn_output[:, 0, :]
        return self.output_layer(cls_attended_output)

class MultimodalRegressor(nn.Module):
    def __init__(self, text_model_name, clip_model_name, regressor_width, regressor_depth, dropout_rate):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name, trust_remote_code=True)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)

        # freeze CLIP vision branch by default
        for p in self.clip_model.vision_model.parameters():
            p.requires_grad = False

        text_embed_dim = self.text_model.config.hidden_size
        image_embed_dim = getattr(self.clip_model.config.vision_config, "hidden_size", None)
        if image_embed_dim is None:
            image_embed_dim = getattr(self.clip_model.vision_model.config, "hidden_size", None) or 512

        self.fusion = CrossAttentionFusion(text_embed_dim, image_embed_dim, text_embed_dim)
        self.regressor = Regressor(text_embed_dim, regressor_width, regressor_depth, dropout_rate)

    def forward(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state  # (B, T, D_text)
        image_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        image_features = getattr(image_outputs, "last_hidden_state", None)
        if image_features is None:
            pooled = getattr(image_outputs, "pooler_output", None) or getattr(image_outputs, "image_embeds", None) or image_outputs
            image_features = pooled.unsqueeze(1)
        fused = self.fusion(text_features, image_features)
        return self.regressor(fused)

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
# Train loop (vectorized collate, augmentation, optional MixUp)
# -------------------------
def train(params):
    mlflow.set_experiment(params.get('MLFLOW_EXPERIMENT', 'Multimodal_OneClick'))
    with mlflow.start_run(run_name=params['RUN_NAME']):
        mlflow.log_params(params)

        df = pd.read_csv(os.path.join(params['DATASET_FOLDER'], params['TRAIN_CSV']))
        df['price'] = df['price'].astype(float)
        df['price'] = np.log1p(df['price'])

        if params['DATA_PERCENTAGE'] < 1.0:
            df = df.sample(frac=params['DATA_PERCENTAGE'], random_state=42).reset_index(drop=True)

        train_df, val_df = train_test_split(df, test_size=params.get('VAL_FRAC', 0.1), random_state=42)

        tokenizer = AutoTokenizer.from_pretrained(params['TEXT_MODEL_NAME'], trust_remote_code=True, use_fast=True)
        try:
            clip_processor = CLIPProcessor.from_pretrained(params['CLIP_MODEL_NAME'], use_fast=True)
        except TypeError:
            clip_processor = CLIPProcessor.from_pretrained(params['CLIP_MODEL_NAME'])

        # define PIL-level train transforms (return PIL)
        train_pil_transforms = transforms.Compose([
            transforms.RandomResizedCrop(params['TARGET_IMAGE_SIZE'], scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
            # keep outputs PIL (these torchvision transforms output PIL)
        ])

        # datasets
        train_ds = PriceImageDataset(train_df, params['TRAIN_IMAGE_DIR'], target_size=params['TARGET_IMAGE_SIZE'])
        val_ds = PriceImageDataset(val_df, params['TRAIN_IMAGE_DIR'], target_size=params['TARGET_IMAGE_SIZE'])

        # collate fns
        train_collate = make_collate_fn(tokenizer, clip_processor, max_length=params['MAX_TEXT_LEN'],
                                        image_size=params['TARGET_IMAGE_SIZE'], train=True,
                                        train_transforms=train_pil_transforms, random_erasing_prob=params.get('RANDOM_ERASE_PROB', 0.2))
        val_collate = make_collate_fn(tokenizer, clip_processor, max_length=params['MAX_TEXT_LEN'],
                                      image_size=params['TARGET_IMAGE_SIZE'], train=False, train_transforms=None, random_erasing_prob=0.0)

        train_loader = DataLoader(train_ds, batch_size=params['BATCH_SIZE'], shuffle=True,
                                  num_workers=params['NUM_WORKERS'], pin_memory=True,
                                  persistent_workers=True, prefetch_factor=params.get('PREFETCH_FACTOR', 2),
                                  collate_fn=train_collate)

        val_loader = DataLoader(val_ds, batch_size=params['BATCH_SIZE'], shuffle=False,
                                num_workers=max(1, params['NUM_WORKERS']//2), pin_memory=True,
                                persistent_workers=True, prefetch_factor=max(1, params.get('PREFETCH_FACTOR', 2)),
                                collate_fn=val_collate)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MultimodalRegressor(params['TEXT_MODEL_NAME'], params['CLIP_MODEL_NAME'],
                                    params['regressor_width'], params['regressor_depth'], params['DROPOUT_RATE']).to(device)

        # freeze/unfreeze behaviour is as before (you can toggle params)
        if not params.get('UNFREEZE_TEXT', False):
            for name, p in model.text_model.named_parameters():
                p.requires_grad = False
        else:
            for name, p in model.text_model.named_parameters():
                p.requires_grad = True

        if params.get('UNFREEZE_CLIP_VISION', False):
            for n, p in model.clip_model.vision_model.named_parameters():
                p.requires_grad = True

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print("Trainable param count:", sum(p.numel() for p in trainable_params))

        optimizer = AdamW(trainable_params, lr=params['LEARNING_RATE'], weight_decay=params['WEIGHT_DECAY'])
        total_steps = math.ceil(len(train_loader) * params['NUM_EPOCHS'] / max(1, params.get('ACCUM_STEPS', 1)))
        warmup = int(total_steps * params.get('WARMUP_PROPORTION', 0.03))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)

        scaler = torch.cuda.amp.GradScaler(enabled=True)
        use_fp16 = params.get('USE_FP16', True)

        best_val_smape = float('inf')
        patience = 0

        for epoch in range(1, params['NUM_EPOCHS'] + 1):
            model.train()
            epoch_loss = 0.0
            epoch_batches = 0
            train_preds = []
            train_trues = []

            pbar = tqdm(train_loader, desc=f"Train E{epoch}/{params['NUM_EPOCHS']}", leave=False)
            optimizer.zero_grad()
            for step, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                pixel_values = batch['pixel_values'].to(device, non_blocking=True)
                prices = batch['price'].to(device, non_blocking=True)

                # Optional MixUp on images+prices (log space)
                if params.get('USE_MIXUP', False) and params.get('MIXUP_ALPHA', 0.3) > 0.0:
                    pixel_values, prices = mixup_batch(pixel_values, prices, alpha=params['MIXUP_ALPHA'])

                with torch.cuda.amp.autocast(enabled=use_fp16, dtype=torch.float16):
                    preds = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values).squeeze(-1)
                    loss = nn.SmoothL1Loss()(preds, prices) / max(1, params.get('ACCUM_STEPS', 1))

                scaler.scale(loss).backward()

                if (step + 1) % max(1, params.get('ACCUM_STEPS', 1)) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, params.get('GRAD_CLIP_NORM', 1.0))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                epoch_loss += float(loss.item() * max(1, params.get('ACCUM_STEPS', 1)))
                epoch_batches += 1

                train_preds.append(preds.detach().cpu().numpy())
                train_trues.append(prices.detach().cpu().numpy())

            avg_train_loss = epoch_loss / max(1, epoch_batches)
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
                    pixel_values = batch['pixel_values'].to(device, non_blocking=True)
                    prices = batch['price'].to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_fp16, dtype=torch.float16):
                        preds = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values).squeeze(-1)
                        vloss = nn.SmoothL1Loss()(preds, prices)
                    val_losses.append(float(vloss.item()))
                    val_preds.append(preds.detach().cpu().numpy())
                    val_trues.append(prices.detach().cpu().numpy())

            avg_val_loss = float(np.mean(val_losses)) if val_losses else float('nan')
            if val_preds:
                val_preds_arr = np.concatenate(val_preds, axis=0)
                val_trues_arr = np.concatenate(val_trues, axis=0)
                val_smape = smape_percent_from_log(val_preds_arr, val_trues_arr)
            else:
                val_smape = float('nan')

            print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, train_smape%={train_smape:.2f}, val_loss={avg_val_loss:.4f}, val_smape%={val_smape:.2f}")
            mlflow.log_metric('val_loss', avg_val_loss, step=epoch)
            mlflow.log_metric('val_smape_percent', val_smape, step=epoch)

            # checkpoint best
            if not math.isnan(val_smape) and val_smape < best_val_smape:
                best_val_smape = val_smape
                patience = 0
                os.makedirs(params['OUTPUT_DIR'], exist_ok=True)
                ckpt = os.path.join(params['OUTPUT_DIR'], f'best_{int(time.time())}.pt')
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
        final_ckpt = os.path.join(params['OUTPUT_DIR'], f'last_{int(time.time())}.pt')
        torch.save({'model_state': model.state_dict(), 'params': params}, final_ckpt)
        mlflow.log_artifact(final_ckpt)
        print("Training finished. final ckpt:", final_ckpt)

# -------------------------
# Default params (edit as desired)
# -------------------------
if __name__ == "__main__":
    params = {
        'RUN_NAME': 'Multimodal_OneClick_Patch',
        'DATASET_FOLDER': 'dataset',
        'TRAIN_CSV': 'train_filtered_cleaned.csv',
        'TRAIN_IMAGE_DIR': 'train_images',
        'TEST_IMAGE_DIR': 'test_images',
        'TEXT_MODEL_NAME': 'chandar-lab/NeoBERT',
        'CLIP_MODEL_NAME': 'openai/clip-vit-base-patch32',
        'BATCH_SIZE': 64,
        'NUM_EPOCHS': 12,
        'LEARNING_RATE': 1e-5,     # lowered LR
        'WEIGHT_DECAY': 0.05,      # updated weight decay
        'ACCUM_STEPS': 1,
        'WARMUP_PROPORTION': 0.03,
        'DATA_PERCENTAGE': 1.0,
        'VAL_FRAC': 0.1,
        'regressor_width': 512,
        'regressor_depth': 2,
        'DROPOUT_RATE': 0.4,       # increased dropout
        'NUM_WORKERS': 16,
        'PREFETCH_FACTOR': 2,
        'TARGET_IMAGE_SIZE': 224,
        'MAX_TEXT_LEN': 512,
        'USE_FP16': True,
        'USE_MIXUP': False,        # set True to enable MixUp
        'MIXUP_ALPHA': 0.3,
        'RANDOM_ERASE_PROB': 0.2,
        'UNFREEZE_TEXT': False,
        'UNFREEZE_CLIP_VISION': False,
        'GRAD_CLIP_NORM': 1.0,
        'EARLY_STOPPING_PATIENCE': 4,
        'OUTPUT_DIR': 'multimodal_oneclick_out',
        'MLFLOW_EXPERIMENT': 'Multimodal OneClick Patch'
    }

    os.makedirs(params['OUTPUT_DIR'], exist_ok=True)
    train(params)
