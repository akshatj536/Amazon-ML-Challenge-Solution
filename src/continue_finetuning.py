import os
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
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

        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding='max_length', truncation=True)

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'price': price
        }

# --- Main Training Function ---
def main(params):
    print("Loading and preprocessing data...")
    train_df = pd.read_csv(os.path.join(params['DATASET_FOLDER'], 'train_filtered.csv'))
    train_df['price'] = np.log1p(train_df['price'])

    if params['DATA_PERCENTAGE'] < 1.0:
        train_df = train_df.sample(frac=params['DATA_PERCENTAGE'], random_state=42)

    print("Initializing model and processor...")
    processor = CLIPProcessor.from_pretrained(params['MODEL_NAME'])
    
    # Load the base model
    base_model = CLIPModel.from_pretrained(params['MODEL_NAME'])

    # Load the LoRA adapter
    model = PeftModel.from_pretrained(base_model, params['ADAPTER_MODEL_PATH'])


    regressor = Regressor(model.config.projection_dim * 2, params['regressor_width'], params['regressor_depth'], params['DROPOUT_RATE'])
    
    # Load the regressor state dict
    regressor_state_dict = torch.load(os.path.join(params['ADAPTER_MODEL_PATH'], 'regressor.pt'))
    regressor.load_state_dict(regressor_state_dict)
    
    model.add_module("regressor", regressor)


    print("Creating datasets and dataloaders...")
    train_dataset = PriceDataset(train_df, processor, params['IMAGE_FOLDER'])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=params['BATCH_SIZE'],
        shuffle=True,
        num_workers=4,
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

    print("Saving the fine-tuned model...")
    model_output_dir = "clip-lora-finetuned-local-continued"
    model.save_pretrained(model_output_dir)
    torch.save(model.regressor.state_dict(), os.path.join(model_output_dir, 'regressor.pt'))


if __name__ == "__main__":
    params = {
        'DATASET_FOLDER': 'dataset/',
        'IMAGE_FOLDER': 'train_images/',
        'MODEL_NAME': 'openai/clip-vit-base-patch32',
        'ADAPTER_MODEL_PATH': 'clip-lora-finetuned-local',
        'BATCH_SIZE': 512,
        'LEARNING_RATE': 1e-5, # Lower learning rate for continued fine-tuning
        'NUM_EPOCHS': 3,
        'DATA_PERCENTAGE': 1.0,
        'regressor_width': 256,
        'regressor_depth': 3,
        'WEIGHT_DECAY': 0.01,
        'DROPOUT_RATE': 0.25,
    }
    main(params)
