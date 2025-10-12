# optionA.py
#
# A minimalistic script to demonstrate efficient hyperparameter optimization with Optuna.
# It generates embeddings once, runs an Optuna study, logs key results to MLflow,
# and runs inference to create a submission file.

import os
import time
import pandas as pd
import numpy as np
import torch
import xgboost as xgb
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import optuna
from optuna.integration import XGBoostPruningCallback
import mlflow

# --- Configuration ---
PARAMS = {
    'RUN_NAME': 'Qwen2-XGBoost-with-Inference',
    'DATASET_FOLDER': 'dataset/',
    'MODEL_NAME': 'Qwen/Qwen3-Embedding-4B',
    'TEXT_COLUMN': 'catalog_content',
    'TARGET_COLUMN': 'price',
    'BATCH_SIZE': 64,
    'RANDOM_STATE': 42,
    'optuna_n_trials': 30,
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

def smape_loss(y_true, y_pred):
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / (denominator + 1e-8)) * 100

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
    test_texts = test_df[params['TEXT_COLUMN']].tolist()

    print("Generating embeddings for test data...")
    test_embeddings = get_embeddings(test_texts, embedding_model, tokenizer, device, params['BATCH_SIZE'])

    print("Making predictions with final XGBoost model...")
    test_predictions = regressor_model.predict(test_embeddings)

    submission_df = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': test_predictions})

    submission_folder = "submission"
    os.makedirs(submission_folder, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    submission_filename = f"submission_xgb_{timestamp}.csv"
    submission_filepath = os.path.join(submission_folder, submission_filename)

    submission_df.to_csv(submission_filepath, index=False)
    print(f"\nSubmission file saved to {submission_filepath}")

    mlflow.log_artifact(submission_filepath)
    print("Submission file logged as MLflow artifact.")

def main():
    """Main function to run the pipeline."""
    with mlflow.start_run(run_name=PARAMS['RUN_NAME']):
        mlflow.log_params(PARAMS)

        # 1. Load Models and Data
        print(f"Loading embedding model: {PARAMS['MODEL_NAME']}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(PARAMS['MODEL_NAME'])
        embedding_model = AutoModel.from_pretrained(PARAMS['MODEL_NAME']).to(device)

        print("Loading data...")
        try:
            train_df = pd.read_csv(os.path.join(PARAMS['DATASET_FOLDER'], 'train_filtered.csv'))
        except FileNotFoundError:
            print(f"Error: Training data not found. Please ensure 'train_filtered.csv' is in '{PARAMS['DATASET_FOLDER']}'.")
            return
        
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=PARAMS['RANDOM_STATE'])

        # 2. Generate Embeddings (once for efficiency)
        X_train = get_embeddings(train_df[PARAMS['TEXT_COLUMN']].tolist(), embedding_model, tokenizer, device, PARAMS['BATCH_SIZE'])
        X_val = get_embeddings(val_df[PARAMS['TEXT_COLUMN']].tolist(), embedding_model, tokenizer, device, PARAMS['BATCH_SIZE'])
        y_train = train_df[PARAMS['TARGET_COLUMN']].values
        y_val = val_df[PARAMS['TARGET_COLUMN']].values

        # 3. Run Optuna Hyperparameter Optimization
        def objective(trial):
            xgb_params = {
                'objective': 'reg:squarederror', 'eval_metric': 'rmse',
                'n_estimators': trial.suggest_int('n_estimators', 400, 2000, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': PARAMS['RANDOM_STATE'], 'n_jobs': 18,
            }
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, callbacks=[XGBoostPruningCallback(trial, "validation_0-rmse")])
            preds = model.predict(X_val)
            return mean_squared_error(y_val, preds)

        print("\n--- Starting Optuna Hyperparameter Optimization ---")
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
        study.optimize(objective, n_trials=PARAMS['optuna_n_trials'])
        print("Optimization finished.")
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_trial_mse", study.best_value)

        # 4. Train Final Model and Evaluate
        print("\nTraining final model with best hyperparameters...")
        final_model = xgb.XGBRegressor(**study.best_params, random_state=PARAMS['RANDOM_STATE'], n_jobs=18)
        final_model.fit(X_train, y_train)

        print("Evaluating final model...")
        train_preds = final_model.predict(X_train)
        val_preds = final_model.predict(X_val)

        train_metrics = {
            "train_mse": mean_squared_error(y_train, train_preds),
            "train_r2": r2_score(y_train, train_preds),
            "train_smape": smape_loss(y_train, train_preds)
        }
        val_metrics = {
            "val_mse": mean_squared_error(y_val, val_preds),
            "val_r2": r2_score(y_val, val_preds),
            "val_smape": smape_loss(y_val, val_preds)
        }
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(val_metrics)

        print("\n--- Best Hyperparameters ---")
        print(study.best_params)
        
        print("\n--- Final Model Evaluation ---")
        print("  Training Metrics:")
        print(f"    MSE:   {train_metrics['train_mse']:.4f}")
        print(f"    R2:    {train_metrics['train_r2']:.4f}")
        print(f"    SMAPE: {train_metrics['train_smape']:.4f}%")
        print("\n  Validation Metrics:")
        print(f"    MSE:   {val_metrics['val_mse']:.4f}")
        print(f"    R2:    {val_metrics['val_r2']:.4f}")
        print(f"    SMAPE: {val_metrics['val_smape']:.4f}%")
        print("----------------------------\n")

        # 5. Run Inference on Test Set
        run_inference(embedding_model, tokenizer, final_model, PARAMS, device)

if __name__ == "__main__":
    mlflow.set_experiment("QWEN2 Price Prediction")
    main()
