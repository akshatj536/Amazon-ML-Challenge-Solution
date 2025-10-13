import os
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

# --- Model Definition ---
class Regressor(torch.nn.Module):
    def __init__(self, input_dim, width, depth, dropout_rate):
        super(Regressor, self).__init__()
        layers = []
        layers.append(torch.nn.Linear(input_dim, width))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout_rate))
        for _ in range(depth - 1):
            layers.append(torch.nn.Linear(width, width))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))
        layers.append(torch.nn.Linear(width, 1))
        self.regressor = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.regressor(x)


# --- Custom SMAPE Loss Function ---
class SMAPELoss(torch.nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_true_real = torch.expm1(y_true)
        y_pred_real = torch.expm1(y_pred)

        numerator = torch.abs(y_pred_real - y_true_real)
        denominator = (torch.abs(y_true_real) + torch.abs(y_pred_real)) / 2
        
        loss = numerator / (denominator + self.epsilon)
        
        return torch.mean(loss)

# --- Dataset Definition (reading from disk) ---
class PriceDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['catalog_content']
        price = torch.tensor(row['price'], dtype=torch.float)

        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'price': price
        }

# --- Main Training Function ---
def main(params):
    print("Loading and preprocessing data...")
    train_df = pd.read_csv(os.path.join(params['DATASET_FOLDER'], 'train_filtered_cleaned.csv'))
    train_df['catalog_content'] = train_df['catalog_content'].astype(str)
    train_df['price'] = np.log1p(train_df['price'])

    if params['DATA_PERCENTAGE'] < 1.0:
        train_df = train_df.sample(frac=params['DATA_PERCENTAGE'], random_state=42)

    print("Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(params['MODEL_NAME'], trust_remote_code=True)
    model = AutoModel.from_pretrained(params['MODEL_NAME'], trust_remote_code=True)

    regressor = Regressor(model.config.hidden_size, params['regressor_width'], params['regressor_depth'], params['DROPOUT_RATE'])
    model.add_module("regressor", regressor)

    print(f"Loading model weights from {params['EXISTING_MODEL_PATH']}...")
    model.load_state_dict(torch.load(params['EXISTING_MODEL_PATH']))

    print("Creating datasets and dataloaders...")
    train_dataset = PriceDataset(train_df, tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params['BATCH_SIZE'],
        shuffle=True,
        num_workers=16,
        pin_memory=True
    )

    print("Starting fine-tuning...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=params['LEARNING_RATE'], weight_decay=params['WEIGHT_DECAY'])
    loss_fn = SMAPELoss()

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(params['NUM_EPOCHS']):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{params['NUM_EPOCHS']}"):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            prices = batch['price'].to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                predicted_price = model.regressor(cls_embedding).squeeze(-1)
                loss = loss_fn(predicted_price, prices)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - Average Training SMAPE Loss: {avg_train_loss:.4f}")

    print("Saving the fine-tuned model...")
    model_output_path = "best_model_rexbert_continued.pt"
    torch.save(model.state_dict(), model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    params = {
        'DATASET_FOLDER': 'dataset/',
        'MODEL_NAME': 'thebajajra/RexBERT-large',
        'EXISTING_MODEL_PATH': 'best_model_distillbert.pt',
        'BATCH_SIZE': 100,
        'LEARNING_RATE': 1e-6,
        'NUM_EPOCHS': 3,
        'DATA_PERCENTAGE': 1,
        'regressor_width': 256,
        'regressor_depth': 1,
        'WEIGHT_DECAY': 0.03,
        'DROPOUT_RATE': 0.3,
    }
    main(params)
