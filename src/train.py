import os
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import mlflow
import matplotlib.pyplot as plt
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

# --- Dataset Definition (reading from disk) ---
class PriceDataset(Dataset):
    def __init__(self, dataframe, processor, image_dir):
        self.dataframe = dataframe
        self.processor = processor
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = row['catalog_content']
        image_url = row['image_link']
        price = torch.tensor(row['price'], dtype=torch.float)

        image_path = os.path.join(self.image_dir, os.path.basename(image_url))
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, TypeError):
            image = Image.new('RGB', (224, 224), (255, 255, 255)) # Placeholder

        # Process each item individually
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding='max_length', truncation=True)

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'price': price
        }

# --- Main Training Function ---
def main(params):
    with mlflow.start_run(run_name=params['RUN_NAME']):
        mlflow.log_params(params)

        print("Loading and preprocessing data...")
        train_df = pd.read_csv(os.path.join(params['DATASET_FOLDER'], 'train_filtered.csv'))
        train_df['price'] = np.log1p(train_df['price'])

        if params['DATA_PERCENTAGE'] < 1.0:
            train_df = train_df.sample(frac=params['DATA_PERCENTAGE'], random_state=42)

        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

        print("Initializing model and processor...")
        processor = CLIPProcessor.from_pretrained(params['MODEL_NAME'],use_fast=True)
        model = CLIPModel.from_pretrained(params['MODEL_NAME'])

        regressor = Regressor(model.config.projection_dim * 2, params['regressor_width'], params['regressor_depth'], params['DROPOUT_RATE'])
        model.add_module("regressor", regressor)

        # --- Fine-tuning strategy logic ---
        if params.get('USE_LORA') and params.get('FREEZE_EMBEDDING_MODEL'):
            raise ValueError("USE_LORA and FREEZE_EMBEDDING_MODEL cannot be True at the same time.")

        if params.get('USE_LORA'):
            print("Configuring LoRA...")
            lora_config = LoraConfig(
                r=params['lora_r'],
                lora_alpha=params['lora_alpha'],
                target_modules=["q_proj", "v_proj"],
                lora_dropout=params['lora_dropout'],
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        elif params.get('FREEZE_EMBEDDING_MODEL'):
            print("Freezing CLIP model weights. Training only the regressor head.")
            for name, param in model.named_parameters():
                if 'regressor' not in name:
                    param.requires_grad = False

        print("Creating datasets and dataloaders...")
        train_dataset = PriceDataset(train_df, processor, params['IMAGE_FOLDER'])
        val_dataset = PriceDataset(val_df, processor, params['IMAGE_FOLDER'])

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=params['BATCH_SIZE'],
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=params['BATCH_SIZE'],
            num_workers=4,
            pin_memory=True
        )

        print("Starting fine-tuning...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        optimizer = AdamW(model.parameters(), lr=params['LEARNING_RATE'], weight_decay=params['WEIGHT_DECAY'])
        loss_fn = torch.nn.MSELoss()

        # --- Mixed Precision Setup ---
        scaler = torch.cuda.amp.GradScaler()

        epoch_metrics = []
        # --- Training Loop ---
        for epoch in range(params['NUM_EPOCHS']):
            model.train()
            total_loss = 0.0
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{params['NUM_EPOCHS']}"):
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                prices = batch['price'].to(device)

                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
                    image_features = outputs.image_embeds
                    text_features = outputs.text_embeds
                    features = torch.cat([image_features, text_features], dim=1)
                    predicted_price = model.regressor(features).squeeze(-1)
                    loss = loss_fn(predicted_price, prices)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")
            mlflow.log_metric("avg_train_loss", avg_train_loss, step=epoch)

            # --- Validation Loop ---
            model.eval()
            total_val_loss = 0.0
            all_prices = []
            all_predicted_prices = []
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    pixel_values = batch['pixel_values'].to(device)
                    prices = batch['price'].to(device)

                    with torch.amp.autocast("cuda"):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
                        image_features = outputs.image_embeds
                        text_features = outputs.text_embeds
                        features = torch.cat([image_features, text_features], dim=1)
                        predicted_price = model.regressor(features).squeeze(-1)
                        val_loss = loss_fn(predicted_price, prices)

                    total_val_loss += val_loss.item()
                    all_prices.extend(prices.detach().cpu().numpy())
                    all_predicted_prices.extend(predicted_price.detach().cpu().float().numpy())

            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}")
            mlflow.log_metric("avg_val_loss", avg_val_loss, step=epoch)
            epoch_metrics.append({'epoch': epoch + 1, 'avg_train_loss': avg_train_loss, 'avg_val_loss': avg_val_loss})

        print("Saving metrics to CSV and logging to MLflow...")
        metrics_df = pd.DataFrame(epoch_metrics)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        metrics_folder = "metrics"
        os.makedirs(metrics_folder, exist_ok=True)
        metrics_filename = os.path.join(metrics_folder, f"metrics_{params['RUN_NAME']}_{timestamp}.csv")
        metrics_df.to_csv(metrics_filename, index=False)
        mlflow.log_artifact(metrics_filename)

        all_prices = np.expm1(all_prices)
        all_predicted_prices = np.expm1(all_predicted_prices)

        plt.figure(figsize=(10, 6))
        plt.scatter(all_prices, all_predicted_prices, alpha=0.5)
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.title("Actual vs. Predicted Prices")
        plt.savefig("validation_plot.png")
        mlflow.log_artifact("validation_plot.png")

        print("Saving the fine-tuned model...")
        model_output_dir = "clip-lora-finetuned-local"
        model.save_pretrained(model_output_dir)
        if params.get('USE_LORA'):
            torch.save(model.regressor.state_dict(), os.path.join(model_output_dir, 'regressor.pt'))

        print("Logging model artifacts to MLflow...")
        mlflow.log_artifacts(
            model_output_dir,
            artifact_path="clip-lora-finetuned"
        )

        # --- Run Inference on Test Set ---
        run_inference(model, processor, params, device)


def run_inference(model, processor, params, device):
    """
    Runs inference on the test set, creates a submission file, and logs it to MLflow.
    """
    print("\nStarting inference on the test set...")

    # 1. Load test data
    test_csv_path = os.path.join(params['DATASET_FOLDER'], 'test.csv')
    if not os.path.exists(test_csv_path):
        print(f"Warning: test.csv not found at {test_csv_path}. Skipping inference.")
        return
        
    test_df = pd.read_csv(test_csv_path)

    # 2. Create a custom Test Dataset and DataLoader
    class TestDataset(Dataset):
        def __init__(self, dataframe, processor, image_dir):
            self.dataframe = dataframe
            self.processor = processor
            self.image_dir = image_dir

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            row = self.dataframe.iloc[idx]
            # Use .get() for safety, fallback to index if 'id' column is missing
            item_id = row.get('id', idx) 
            text = row['catalog_content']
            image_url = row['image_link']

            image_path = os.path.join(self.image_dir, os.path.basename(image_url))
            try:
                image = Image.open(image_path).convert("RGB")
            except (FileNotFoundError, TypeError):
                image = Image.new('RGB', (224, 224), (255, 255, 255))

            inputs = self.processor(text=[text], images=image, return_tensors="pt", padding='max_length', truncation=True)

            return {
                'id': item_id,
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'pixel_values': inputs['pixel_values'].squeeze(0)
            }

    test_dataset = TestDataset(test_df, processor, params['TEST_IMAGE_FOLDER'])
    test_dataloader = DataLoader(test_dataset, batch_size=params['BATCH_SIZE'], shuffle=False, num_workers=2)

    # 3. Inference loop
    model.eval()
    all_ids = []
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Inference"):
            all_ids.extend(batch['id'])
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                features = torch.cat([image_features, text_features], dim=1)
                predicted_price_log = model.regressor(features).squeeze(-1)
            
            all_predictions.extend(predicted_price_log.detach().cpu().float().numpy())

    # 4. Create submission file
    final_predictions = np.expm1(all_predictions)

    submission_df = pd.DataFrame({'id': all_ids, 'price': final_predictions})

    submission_folder = "submission"
    os.makedirs(submission_folder, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    submission_filename = f"submission_{timestamp}.csv"
    submission_filepath = os.path.join(submission_folder, submission_filename)

    submission_df.to_csv(submission_filepath, index=False)
    print(f"\nSubmission file saved to {submission_filepath}")

    # 5. Log as artifact
    mlflow.log_artifact(submission_filepath)
    print("Submission file logged as MLflow artifact.")



if __name__ == "__main__":
    params = {
        'RUN_NAME': 'simple_mixed_precision_run',
        'DATASET_FOLDER': 'dataset/',
        'IMAGE_FOLDER': 'train_images/',
        'TEST_IMAGE_FOLDER': 'test_images/',
        'MODEL_NAME': 'openai/clip-vit-base-patch32',
        'BATCH_SIZE': 1024,
        'LEARNING_RATE': 3e-4,
        'NUM_EPOCHS': 10,
        'DATA_PERCENTAGE': 1.0,
        'USE_LORA': False,
        'FREEZE_EMBEDDING_MODEL': False,
        'MIXED_PRECISION': True,
        'lora_r': 16,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'regressor_width': 256,
        'regressor_depth': 3,
        'WEIGHT_DECAY': 0.01,
        'DROPOUT_RATE': 0.25,
    }
    mlflow.set_experiment("CLIP Price Prediction")
    main(params)
