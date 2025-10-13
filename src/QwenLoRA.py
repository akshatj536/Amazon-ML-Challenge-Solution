# train_regressor_only.py
import os
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import mlflow
import numpy as np
from peft import get_peft_model, LoraConfig, PeftModel

# --- Configuration ---
PARAMS = {
    'RUN_NAME': 'QWEN embedding LoRA',
    'DATASET_FOLDER': 'dataset/',
    'MODEL_NAME': 'Qwen/Qwen3-Embedding-0.6B',
    'MODEL_SAVE_PATH': 'best_model_qwen',
    'BATCH_SIZE': 4,
    'LEARNING_RATE': 2e-4,
    'NUM_EPOCHS': 1,
    'DATA_PERCENTAGE': 0.001,
    'MIXED_PRECISION': True,
    'regressor_width': 256,
    'regressor_depth': 1,
    'WEIGHT_DECAY': 0.01,
    'DROPOUT_RATE': 0.3,
    'EARLY_STOPPING_PATIENCE': 5,
    'LORA_R': 8,
    'LORA_ALPHA': 32,
    'LORA_DROPOUT': 0.3,
}

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
        layers.append(torch.nn.ReLU())
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

# --- Dataset Definition ---
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
        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=2048)
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'price': price
        }

def main():
    with mlflow.start_run(run_name=PARAMS['RUN_NAME']):
        mlflow.log_params(PARAMS)

        print("Loading and preprocessing data...")
        df = pd.read_csv(os.path.join(PARAMS['DATASET_FOLDER'], 'train_filtered.csv'))
        df['price'] = np.log1p(df['price'])

        if PARAMS['DATA_PERCENTAGE'] < 1.0:
            df = df.sample(frac=PARAMS['DATA_PERCENTAGE'], random_state=42)

        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

        print("Initializing model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(PARAMS['MODEL_NAME'], trust_remote_code=True)
        model = AutoModel.from_pretrained(PARAMS['MODEL_NAME'], trust_remote_code=True)

        # --- PEFT LoRA Setup ---
        lora_config = LoraConfig(
            r=PARAMS['LORA_R'],
            lora_alpha=PARAMS['LORA_ALPHA'],
            lora_dropout=PARAMS['LORA_DROPOUT'],
            target_modules="all-linear",  # Auto-target all linear layers
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        regressor = Regressor(model.config.hidden_size, PARAMS['regressor_width'], PARAMS['regressor_depth'], PARAMS['DROPOUT_RATE'])

        print("Creating datasets and dataloaders...")
        train_dataset = PriceDataset(train_df, tokenizer)
        val_dataset = PriceDataset(val_df, tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=PARAMS['BATCH_SIZE'], shuffle=True, num_workers=16, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=PARAMS['BATCH_SIZE'], num_workers=16, pin_memory=True)

        print("Starting LoRA fine-tuning...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        regressor.to(device)

        optimizer = AdamW(list(model.parameters()) + list(regressor.parameters()), lr=PARAMS['LEARNING_RATE'], weight_decay=PARAMS['WEIGHT_DECAY'])
        loss_fn = SMAPELoss()
        num_training_steps = len(train_dataloader) * PARAMS['NUM_EPOCHS']
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
        scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda' and PARAMS['MIXED_PRECISION']))

        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(PARAMS['NUM_EPOCHS']):
            model.train()
            regressor.train()
            total_loss = 0.0
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{PARAMS['NUM_EPOCHS']}"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                prices = batch['price'].to(device)

                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda' and PARAMS['MIXED_PRECISION'])):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    last_hidden_state = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    mean_pooled_embedding = sum_embeddings / sum_mask
                    predicted_price = regressor(mean_pooled_embedding).squeeze(-1)
                    loss = loss_fn(predicted_price, prices)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(regressor.parameters()), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_dataloader)
            
            model.eval()
            regressor.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    prices = batch['price'].to(device)
                    with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda' and PARAMS['MIXED_PRECISION'])):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        last_hidden_state = outputs.last_hidden_state
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        mean_pooled_embedding = sum_embeddings / sum_mask
                        predicted_price = regressor(mean_pooled_embedding).squeeze(-1)
                        val_loss = loss_fn(predicted_price, prices)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f"Epoch {epoch + 1} - Train SMAPE Loss: {avg_train_loss:.4f}, Val SMAPE Loss: {avg_val_loss:.4f}")
            mlflow.log_metric("train_smape_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_smape_loss", avg_val_loss, step=epoch)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print(f"Validation loss improved. Saving model to {PARAMS['MODEL_SAVE_PATH']}.")
                model.save_pretrained(PARAMS['MODEL_SAVE_PATH'])
                torch.save(regressor.state_dict(), os.path.join(PARAMS['MODEL_SAVE_PATH'], 'regressor.pt'))
            else:
                patience_counter += 1
                if patience_counter >= PARAMS['EARLY_STOPPING_PATIENCE']:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        print("Loading best model for inference...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(PARAMS['MODEL_NAME'], trust_remote_code=True)

        # Load the base model and apply PEFT adapter
        base_model = AutoModel.from_pretrained(PARAMS['MODEL_NAME'], trust_remote_code=True)
        model = PeftModel.from_pretrained(base_model, PARAMS['MODEL_SAVE_PATH'])
        model.to(device)

        regressor = Regressor(model.config.hidden_size, PARAMS['regressor_width'], PARAMS['regressor_depth'], PARAMS['DROPOUT_RATE'])
        regressor.load_state_dict(torch.load(os.path.join(PARAMS['MODEL_SAVE_PATH'], 'regressor.pt')))
        regressor.to(device)

        run_inference(model, regressor, tokenizer, PARAMS, device)


def run_inference(model, regressor, tokenizer, params, device):
    """
    Runs inference on the test set, creates a submission file, and logs it to MLflow.
    """
    print("\nStarting inference on the test set...")
    test_csv_path = os.path.join(params['DATASET_FOLDER'], 'test.csv')
    if not os.path.exists(test_csv_path):
        print(f"Warning: test.csv not found at {test_csv_path}. Skipping inference.")
        return
        
    test_df = pd.read_csv(test_csv_path)

    class TestDataset(Dataset):
        def __init__(self, dataframe, tokenizer):
            self.dataframe = dataframe
            self.tokenizer = tokenizer
        def __len__(self):
            return len(self.dataframe)
        def __getitem__(self, idx):
            row = self.dataframe.iloc[idx]
            inputs = self.tokenizer(row['catalog_content'], return_tensors="pt", padding='max_length', truncation=True, max_length=2048)
            return {'sample_id': row['sample_id'], 'input_ids': inputs['input_ids'].squeeze(0), 'attention_mask': inputs['attention_mask'].squeeze(0)}

    test_dataset = TestDataset(test_df, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=params['BATCH_SIZE'], shuffle=False, num_workers=16)

    model.eval()
    regressor.eval()
    all_sample_ids = []
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Inference"):
            all_sample_ids.extend(batch['sample_id'].tolist() if isinstance(batch['sample_id'], torch.Tensor) else batch['sample_id'])
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda' and params['MIXED_PRECISION'])):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_pooled_embedding = sum_embeddings / sum_mask
                predicted_price_log = regressor(mean_pooled_embedding).squeeze(-1)

            all_predictions.extend(predicted_price_log.detach().cpu().float().numpy())

    final_predictions = np.expm1(all_predictions)
    submission_df = pd.DataFrame({'sample_id': all_sample_ids, 'price': final_predictions})
    submission_folder = "submission"
    os.makedirs(submission_folder, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    submission_filename = f"submission_qwen3_lora_{timestamp}.csv"
    submission_filepath = os.path.join(submission_folder, submission_filename)
    submission_df.to_csv(submission_filepath, index=False)
    print(f"\nSubmission file saved to {submission_filepath}")
    mlflow.log_artifact(submission_filepath)
    print("Submission file logged as MLflow artifact.")


if __name__ == "__main__":
    mlflow.set_experiment("QWEN3 Price Prediction")
    main()