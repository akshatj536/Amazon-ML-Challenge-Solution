import pandas as pd
import pydantic
from typing import List, Optional
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import json

# 1. Define the Pydantic schema for validation after extraction
class CatalogItem(pydantic.BaseModel):
    item_name: str
    bullet_points: Optional[List[str]]

def main():
    """
    Main function to run the hybrid extraction process using a direct transformers pipeline.
    """
    # --- Configuration ---
    MODEL_NAME = "microsoft/phi-2"
    INPUT_CSV = "/teamspace/studios/this_studio/dataset/train_filtered.csv"
    OUTPUT_CSV = "/teamspace/studios/this_studio/dataset/train_structured.csv"
    SAMPLE_SIZE = 500 # Set to None to run on the full dataset

    # --- Step 1: Load Model and Tokenizer ---
    print(f"Step 1/4: Loading model: {MODEL_NAME}. This may take a while...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Step 2: Load and Regex-process the data ---
    print("Step 2/4: Loading and processing dataset with Regex...")
    df = pd.read_csv(INPUT_CSV)

    if SAMPLE_SIZE is not None:
        print(f"--- RUNNING ON A SAMPLE OF {SAMPLE_SIZE} ROWS ---")
        df = df.head(SAMPLE_SIZE)

    df['value'] = df['catalog_content'].str.extract(r'Value:\s*([\d\.]+)').astype(float)
    df['unit'] = df['catalog_content'].str.extract(r'Unit:\s*(\w+)')
    print("Regex extraction complete.")

    # --- Step 3: LLM Extraction ---
    print("Step 3/4: Extracting complex fields with LLM...")
    results = []
    prompt_template = '''Instruct: You are an expert data parser. Parse the following text and output a single, valid JSON object with the keys "item_name" and "bullet_points".
Input: {content}
Output:'''

    for content in tqdm(df['catalog_content'], total=len(df), desc="Extracting"):
        if pd.isna(content):
            results.append({'item_name': None, 'bullet_points': []})
            continue

        prompt = prompt_template.format(content=content)
        
        try:
            # Generate text from the model
            generated_text = pipe(prompt, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)[0]['generated_text']
            
            # Extract the JSON part of the string
            json_str = generated_text.split("Output:")[1].strip()
            json_data = json.loads(json_str)
            
            # Validate with Pydantic
            validated_data = CatalogItem(**json_data)
            results.append(validated_data.model_dump())

        except (json.JSONDecodeError, IndexError, pydantic.ValidationError) as e:
            # print(f"Parsing Error: {e}") # Optional for debugging
            results.append({'item_name': 'PARSE_ERROR', 'bullet_points': []})

    print("LLM extraction complete.")
    structured_df = pd.DataFrame(results)

    # --- Step 4: Combine and Save ---
    print("Step 4/4: Combining and saving the final dataset...")
    df.reset_index(drop=True, inplace=True)
    structured_df.reset_index(drop=True, inplace=True)
    
    final_df = pd.concat([df, structured_df], axis=1)

    print(f"Saving structured data to {OUTPUT_CSV}...")
    final_df.to_csv(OUTPUT_CSV, index=False)
    print("Done.")

if __name__ == "__main__":
    main()