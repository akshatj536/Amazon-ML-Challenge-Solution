
import os
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm

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

# --- 1. Initialize model, processor, and LoRA ---
print("Initializing model and processor...")
MODEL_DIR = 'clip-lora-finetuned-local'
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
model = CLIPModel.from_pretrained(MODEL_DIR)

# Add and load the regression head
regressor = Regressor(model.config.projection_dim * 2, 512, 2)
regressor.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'regressor.pt')))
model.add_module("regressor", regressor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predictor(catalog_content, image_link):
    '''
    Call your model/approach here
    
    Parameters:
    - catalog_content: Text containing product title and description
    - image_link: URL to product image
    
    Returns:
    - price: Predicted price as a float
    '''
    try:
        # Open image from the test_images folder
        image_path = os.path.join('test_images', os.path.basename(image_link))
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, TypeError):
        print(f"Image not found: {image_path}. Using a placeholder.")
        image = Image.new('RGB', (224, 224), (255, 255, 255))


    inputs = processor(text=[catalog_content], images=image, return_tensors="pt", padding='max_length', truncation=True)
    
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda"):
                outputs = model(**inputs)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                
                features = torch.cat([image_features, text_features], dim=1)
                predicted_price = model.regressor(features).squeeze(-1)
        else:
            outputs = model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            
            features = torch.cat([image_features, text_features], dim=1)
            predicted_price = model.regressor(features).squeeze(-1)
    
    return predicted_price.item()

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    
    # Read test data
    print("Reading test data...")
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    # Apply predictor function to each row
    print("Generating predictions...")
    prices = []
    for index, row in tqdm(test.iterrows(), total=test.shape[0], desc="Predicting prices"):
        prices.append(predictor(row['catalog_content'], row['image_link']))
    test['price'] = prices
    
    # Select only required columns for output
    output_df = test[['sample_id', 'price']]
    output_df.rename(columns={'sample_id': 'id'}, inplace=True)
    
    # Save predictions
    output_filename = 'submission.csv'
    output_df.to_csv(output_filename, index=False)
    
    print(f"Predictions saved to {output_filename}")
    print(f"Total predictions: {len(output_df)}")
    print(f"Sample predictions:\n{output_df.head()}")
