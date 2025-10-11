import os
import pandas as pd
from src.utils import download_images

def download_for_csv(csv_path, image_dir):
    """
    Reads a CSV, extracts image links, and uses the utility function from src.utils to download them.
    """
    print(f"--- Processing {csv_path} ---")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return

    if 'image_link' not in df.columns:
        print(f"Error: 'image_link' column not found in {csv_path}")
        return
        
    # Get a list of unique, non-empty image URLs
    image_links = df['image_link'].dropna().unique().tolist()
    
    print(f"Found {len(image_links)} unique images to download.")
    
    # Call the download function from utils.py
    download_images(image_links, image_dir)
    
    print(f"Finished processing {csv_path}.\n")


def main():
    """
    Main function to download images for train and test datasets using src.utils.
    """
    # Define paths
    train_csv = 'dataset/train.csv'
    test_csv = 'dataset/test.csv'
    train_image_dir = 'train_images'
    test_image_dir = 'test_images'

    # Download images for both datasets
    download_for_csv(train_csv, train_image_dir)
    download_for_csv(test_csv, test_image_dir)
    
    print("All image downloads attempted using src.utils.download_images.")

if __name__ == "__main__":
    main()