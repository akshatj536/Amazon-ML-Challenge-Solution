# train_regressor_precomputed.py

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
import optuna

# --- Configuration ---
PARAMS = {
    'RUN_NAME': 'Precomputed-Regressor-Qwen3-0.6B',
    'DATASET_FOLDER': 'dataset/',
    'MODEL_NAME': 'Qwen/Qwen3-Embedding-0.6B',
    'MODEL_SAVE_PATH': 'best_model_pytorch_regressor_precomputed',
    'BATCH_SIZE': 32, 
    'LEARNING_RATE': 2e-5,
    'NUM_EPOCHS': 100,
    'DATA_PERCENTAGE': 1,
    'MIXED_PRECISION': True,
    'regressor_width': 512,
    'regressor_depth': 3,
    'WEIGHT_DECAY': 0.01,
    'DROPOUT_RATE': 0.3,
    'EARLY_STOPPING_PATIENCE': 5,
}

# --- Model Definition ---
class Regressor(torch.nn.Module):
    def __init__(self, input_dim, width, depth, dropout_rate):
        super(Regressor, self).__init__()
        layers = []
        layers.append(torch.nn.Linear(input_dim, width))
        layers.append(torch.nn.BatchNorm1d(width))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout_rate))
        for _ in range(depth - 1):
            layers.append(torch.nn.Linear(width, width))
            layers.append(torch.nn.BatchNorm1d(width))
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

# --- Dataset for Precomputed Embeddings ---
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, prices):
        self.embeddings = embeddings
        self.prices = prices

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            'embedding': torch.tensor(self.embeddings[idx], dtype=torch.float),
            'price': torch.tensor(self.prices[idx], dtype=torch.float)
        }

def get_embeddings(texts, model, tokenizer, device, batch_size):
    """Generates embeddings for a list of texts."""
    print(f"Generating embeddings for {len(texts)} texts...")
    model.eval()
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Batches"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled_embeddings = sum_hidden / sum_mask
        all_embeddings.append(mean_pooled_embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

def objective(trial, X_train, y_train, X_val, y_val, device, params):
    """Optuna objective function."""
    # --- Hyperparameter Search Space ---
    regressor_width = trial.suggest_int('regressor_width', 256, 1024, log=True)
    regressor_depth = trial.suggest_int('regressor_depth', 1, 4)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 0.1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # Create a new run for each trial
    with mlflow.start_run(run_name=f"Trial {trial.number}", nested=True):
        mlflow.log_params(trial.params)

        # Create Datasets and Dataloaders
        train_dataset = EmbeddingDataset(X_train, y_train)
        val_dataset = EmbeddingDataset(X_val, y_val)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

        # Initialize Regressor and Optimizer
        regressor = Regressor(X_train.shape[1], regressor_width, regressor_depth, dropout_rate).to(device)
        optimizer = AdamW(regressor.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = SMAPELoss()
        scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda' and params['MIXED_PRECISION']))
        num_training_steps = len(train_dataloader) * params['NUM_EPOCHS']
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        # Training Loop
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(params['NUM_EPOCHS']):
            regressor.train()
            total_loss = 0.0
            for batch in train_dataloader:
                optimizer.zero_grad()
                embeddings = batch['embedding'].to(device)
                prices = batch['price'].to(device)

                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda' and params['MIXED_PRECISION'])):
                    predicted_price = regressor(embeddings).squeeze(-1)
                    loss = loss_fn(predicted_price, prices)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_dataloader)

            regressor.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    embeddings = batch['embedding'].to(device)
                    prices = batch['price'].to(device)
                    with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda' and params['MIXED_PRECISION'])):
                        predicted_price = regressor(embeddings).squeeze(-1)
                        val_loss = loss_fn(predicted_price, prices)
                    total_val_loss += val_loss.item()
            
            avg_val_loss = total_val_loss / len(val_dataloader)
            mlflow.log_metric("train_smape_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_smape_loss", avg_val_loss, step=epoch)
            print(f"Trial {trial.number} Epoch {epoch + 1} - Train SMAPE: {avg_train_loss:.4f}, Val SMAPE: {avg_val_loss:.4f}")

            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= params['EARLY_STOPPING_PATIENCE']:
                    print(f"Early stopping at epoch {epoch + 1} for trial {trial.number}")
                    break
        
        return best_val_loss

def main():
    with mlflow.start_run(run_name=PARAMS['RUN_NAME']) as parent_run:
        mlflow.log_params(PARAMS)

        # 1. Load Models and Data
        print(f"Loading embedding model: {PARAMS['MODEL_NAME']}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(PARAMS['MODEL_NAME'], trust_remote_code=True)
        embedding_model = AutoModel.from_pretrained(PARAMS['MODEL_NAME'], trust_remote_code=True).to(device)

        print("Loading data...")
        df = pd.read_csv(os.path.join(PARAMS['DATASET_FOLDER'], 'train_filtered.csv'))
        df['price'] = np.log1p(df['price'])

        if PARAMS['DATA_PERCENTAGE'] < 1.0:
            df = df.sample(frac=PARAMS['DATA_PERCENTAGE'], random_state=42)

        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

        # 2. Generate Embeddings (once for efficiency)
        X_train = get_embeddings(train_df['catalog_content'].tolist(), embedding_model, tokenizer, device, PARAMS['BATCH_SIZE'])
        X_val = get_embeddings(val_df['catalog_content'].tolist(), embedding_model, tokenizer, device, PARAMS['BATCH_SIZE'])
        y_train = train_df['price'].values
        y_val = val_df['price'].values
        
        # Unload the embedding model to free up GPU memory
        del embedding_model
        torch.cuda.empty_cache()
        print("Embedding model unloaded from GPU memory.")

        # --- Optuna Study ---
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        objective_with_data = lambda trial: objective(trial, X_train, y_train, X_val, y_val, device, PARAMS)
        
        print("\n--- Starting Optuna Hyperparameter Search ---")
        study.optimize(objective_with_data, n_trials=25) # Number of trials

        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Log best params to the main mlflow run
        mlflow.log_params({f"best_{k}": v for k, v in trial.params.items()})
        mlflow.log_metric("best_val_smape_loss", trial.value)

        # --- Train final model with best hyperparameters ---
        print("\n--- Training Final Model with Best Hyperparameters ---")
        best_params = trial.params
        
        # Combine train and validation sets for final training
        X_train_full = np.vstack((X_train, X_val))
        y_train_full = np.concatenate((y_train, y_val))

        train_dataset_full = EmbeddingDataset(X_train_full, y_train_full)
        train_dataloader_full = DataLoader(train_dataset_full, batch_size=best_params['batch_size'], shuffle=True)

        final_regressor = Regressor(
            X_train.shape[1], 
            best_params['regressor_width'], 
            best_params['regressor_depth'], 
            best_params['dropout_rate']
        ).to(device)
        
        optimizer = AdamW(final_regressor.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
        loss_fn = SMAPELoss()
        scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda' and PARAMS['MIXED_PRECISION']))
        num_training_steps = len(train_dataloader_full) * PARAMS['NUM_EPOCHS'] # You might want to adjust epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        val_dataset = EmbeddingDataset(X_val, y_val)
        val_dataloader = DataLoader(val_dataset, batch_size=best_params['batch_size'], num_workers=0)

        # Training loop for the final model
        for epoch in range(PARAMS['NUM_EPOCHS']):
            final_regressor.train()
            total_loss = 0.0
            for batch in tqdm(train_dataloader_full, desc=f"Final Training Epoch {epoch + 1}/{PARAMS['NUM_EPOCHS']}"):
                optimizer.zero_grad()
                embeddings = batch['embedding'].to(device)
                prices = batch['price'].to(device)

                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda' and PARAMS['MIXED_PRECISION'])):
                    predicted_price = final_regressor(embeddings).squeeze(-1)
                    loss = loss_fn(predicted_price, prices)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_dataloader_full)
            mlflow.log_metric("final_train_smape_loss", avg_train_loss, step=epoch)

            # Evaluate on validation set
            final_regressor.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    embeddings = batch['embedding'].to(device)
                    prices = batch['price'].to(device)
                    with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda' and PARAMS['MIXED_PRECISION'])):
                        predicted_price = final_regressor(embeddings).squeeze(-1)
                        val_loss = loss_fn(predicted_price, prices)
                    total_val_loss += val_loss.item()
            
            avg_val_loss = total_val_loss / len(val_dataloader)
            mlflow.log_metric("final_val_smape_loss", avg_val_loss, step=epoch)
            print(f"Final Training Epoch {epoch + 1} - Train SMAPE: {avg_train_loss:.4f}, Val SMAPE: {avg_val_loss:.4f}")

        print("Saving final model.")
        os.makedirs(PARAMS['MODEL_SAVE_PATH'], exist_ok=True)
        torch.save(final_regressor.state_dict(), os.path.join(PARAMS['MODEL_SAVE_PATH'], 'regressor.pt'))

        # 6. Run Inference
        print("\nLoading embedding model again for inference...")
        embedding_model = AutoModel.from_pretrained(PARAMS['MODEL_NAME'], trust_remote_code=True).to(device)
        
        run_inference(embedding_model, tokenizer, final_regressor, PARAMS, device)

def run_inference(embedding_model, tokenizer, regressor_model, params, device):
    """
    Runs inference on the test set, creates a submission file, and logs it to MLflow.
    """
    print("\nStarting inference on the test set...")
    test_csv_path = os.path.join(params['DATASET_FOLDER'], 'test.csv')
    if not os.path.exists(test_csv_path):
        print(f"Warning: test.csv not found at {test_csv_path}. Skipping inference.")
        return
        
    test_df = pd.read_csv(test_csv_path)
    test_texts = test_df['catalog_content'].tolist()

    print("Generating embeddings for test data...")
    test_embeddings = get_embeddings(test_texts, embedding_model, tokenizer, device, params['BATCH_SIZE'])

    regressor_model.eval()
    all_predictions = []
    with torch.no_grad():
        # Create a simple dataloader for inference
        test_embedding_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_embeddings))
        test_dataloader = torch.utils.data.DataLoader(test_embedding_dataset, batch_size=params['BATCH_SIZE'] * 2) # Use larger batch for inference
        
        for batch in tqdm(test_dataloader, desc="Inference"):
            embeddings = batch[0].to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda' and params['MIXED_PRECISION'])):
                predicted_price_log = regressor_model(embeddings).squeeze(-1)
            all_predictions.extend(predicted_price_log.detach().cpu().float().numpy())

    final_predictions = np.expm1(all_predictions)
    submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': final_predictions})
    submission_folder = "submission"
    os.makedirs(submission_folder, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    submission_filename = f"submission_precomputed_{timestamp}.csv"
    submission_filepath = os.path.join(submission_folder, submission_filename)
    submission_df.to_csv(submission_filepath, index=False)
    print(f"\nSubmission file saved to {submission_filepath}")
    mlflow.log_artifact(submission_filepath)
    print("Submission file logged as MLflow artifact.")


if __name__ == "__main__":
    mlflow.set_experiment("QWEN3 Price Prediction")
    main()