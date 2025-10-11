import os
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


# Define the Regressor ANN
class Regressor(torch.nn.Module):
    def __init__(self, input_dim, width, depth):
        super(Regressor, self).__init__()
        layers = []
        layers.append(torch.nn.Linear(input_dim, width))
        layers.append(torch.nn.ReLU())
        for _ in range(depth - 1):
            layers.append(torch.nn.Linear(width, width))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(width, 1))
        self.regressor = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.regressor(x)


# Define the custom dataset
class PriceDataset(Dataset):
    def __init__(self, dataframe, processor, image_dir='images'):
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
        
        # Open pre-downloaded image
        image_path = os.path.join(self.image_dir, os.path.basename(image_url))
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, TypeError):
            # Handle missing images or invalid image_url types
            if isinstance(image_url, str):
                print(f"Image not found: {image_path}. Using a placeholder.")
            else:
                print(f"Invalid image URL at index {idx}: {image_url}. Using a placeholder.")
            image = Image.new('RGB', (224, 224), (255, 255, 255))


        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding='max_length', truncation=True)
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'price': price
        }

def main(params):
    with mlflow.start_run(run_name=params['RUN_NAME']):
        mlflow.log_params(params)

        # --- 2. Load and preprocess data ---
        print("Loading and preprocessing data...")
        train_df = pd.read_csv(os.path.join(params['DATASET_FOLDER'], 'train.csv'))
        
        # Sample a percentage of the data
        if params['DATA_PERCENTAGE'] < 1.0:
            print(f"Using {params['DATA_PERCENTAGE'] * 100:.0f}% of the data for training.")
            train_df = train_df.sample(frac=params['DATA_PERCENTAGE'], random_state=42)


        # Split data into training and validation sets
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
        
        # --- 3. Initialize model, processor, and LoRA ---
        print("Initializing model and processor...")
        processor = CLIPProcessor.from_pretrained(params['MODEL_NAME'])
        model = CLIPModel.from_pretrained(params['MODEL_NAME'])

        # Add a regression head
        regressor = Regressor(model.config.projection_dim * 2, params['regressor_width'], params['regressor_depth'])
        model.add_module("regressor", regressor)

        if params['USE_LORA']:
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


        # --- 4. Create datasets and dataloaders ---
        print("Creating datasets and dataloaders...")
        train_dataset = PriceDataset(train_df, processor, params['IMAGE_FOLDER'])
        val_dataset = PriceDataset(val_df, processor, params['IMAGE_FOLDER'])
        train_dataloader = DataLoader(train_dataset, batch_size=params['BATCH_SIZE'], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=params['BATCH_SIZE'])

        # --- 5. Fine-tuning loop ---
        print("Starting fine-tuning...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=params['LEARNING_RATE'])
        loss_fn = torch.nn.MSELoss()

        if params['MIXED_PRECISION']:
            scaler = torch.amp.GradScaler("cuda")

        epoch_metrics = []
        for epoch in range(params['NUM_EPOCHS']):
            model.train()
            total_loss = 0
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{params['NUM_EPOCHS']}"):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                prices = batch['price'].to(device)

                if params['MIXED_PRECISION']:
                    with torch.amp.autocast("cuda"):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
                        image_features = outputs.image_embeds
                        text_features = outputs.text_embeds
                        
                        # Combine features and predict price
                        features = torch.cat([image_features, text_features], dim=1)
                        predicted_price = model.regressor(features).squeeze(-1)

                        loss = loss_fn(predicted_price, prices)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
                    image_features = outputs.image_embeds
                    text_features = outputs.text_embeds
                    
                    # Combine features and predict price
                    features = torch.cat([image_features, text_features], dim=1)
                    predicted_price = model.regressor(features).squeeze(-1)

                    loss = loss_fn(predicted_price, prices)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")
            mlflow.log_metric("avg_train_loss", avg_train_loss, step=epoch)


            # Validation loop
            model.eval()
            total_val_loss = 0
            all_prices = []
            all_predicted_prices = []
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    pixel_values = batch['pixel_values'].to(device)
                    prices = batch['price'].to(device)

                    if params['MIXED_PRECISION']:
                        with torch.cuda.amp.autocast():
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
                            image_features = outputs.image_embeds
                            text_features = outputs.text_embeds

                            features = torch.cat([image_features, text_features], dim=1)
                            predicted_price = model.regressor(features).squeeze(-1)
                            
                            val_loss = loss_fn(predicted_price, prices)
                    else:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
                        image_features = outputs.image_embeds
                        text_features = outputs.text_embeds

                        features = torch.cat([image_features, text_features], dim=1)
                        predicted_price = model.regressor(features).squeeze(-1)
                        
                        val_loss = loss_fn(predicted_price, prices)

                    total_val_loss += val_loss.item()

                    all_prices.extend(prices.cpu().numpy())
                    all_predicted_prices.extend(predicted_price.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f}")
            mlflow.log_metric("avg_val_loss", avg_val_loss, step=epoch)
            epoch_metrics.append({'epoch': epoch + 1, 'avg_train_loss': avg_train_loss, 'avg_val_loss': avg_val_loss})

        # --- 6. Log metrics artifact and validation plot ---
        print("Saving metrics to CSV and logging to MLflow...")
        metrics_df = pd.DataFrame(epoch_metrics)
        # Create a unique filename for the metrics CSV
        import time
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        metrics_folder = "metrics"
        os.makedirs(metrics_folder, exist_ok=True)
        metrics_filename = os.path.join(metrics_folder, f"metrics_{params['RUN_NAME']}_{timestamp}.csv")
        metrics_df.to_csv(metrics_filename, index=False)
        mlflow.log_artifact(metrics_filename)

        
        plt.figure(figsize=(10, 6))
        plt.scatter(all_prices, all_predicted_prices, alpha=0.5)
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.title("Actual vs. Predicted Prices")
        plt.savefig("validation_plot.png")
        mlflow.log_artifact("validation_plot.png")


        # --- 7. Save the model ---
        print("Saving the fine-tuned model...")
        model_output_dir = "clip-lora-finetuned-local"
        model.save_pretrained(model_output_dir)
        if params['USE_LORA']:
            torch.save(model.regressor.state_dict(), os.path.join(model_output_dir, 'regressor.pt'))

        print("Logging model artifacts to MLflow...")
        mlflow.log_artifacts(
            model_output_dir,
            artifact_path="clip-lora-finetuned"
        )


if __name__ == "__main__":
    params = {
        'RUN_NAME': 'clip_price_prediction_lora-1-cpu-test',
        'DATASET_FOLDER': 'dataset/',
        'IMAGE_FOLDER': 'train_images/',
        'MODEL_NAME': 'openai/clip-vit-base-patch32',
        'BATCH_SIZE': 4,
        'LEARNING_RATE': 3e-4,
        'NUM_EPOCHS': 3,
        'DATA_PERCENTAGE': 0.0001,
        'USE_LORA': False,
        'MIXED_PRECISION': False,
        'lora_r': 1,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'regressor_width': 512,
        'regressor_depth': 2,
    }
    mlflow.set_experiment("CLIP Price Prediction")
    main(params)