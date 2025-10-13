
import pandas as pd
import re

def clean_catalog_content(text):
    """
    Extracts Item Name and Bullet Points from the catalog_content string,
    and performs additional cleaning.
    """
    if not isinstance(text, str):
        return ""

    lines = text.split('\n')
    item_name = ''
    bullet_points = []

    for line in lines:
        if line.startswith("Item Name:"):
            item_name = line.replace("Item Name:", "").strip()
        elif line.startswith("Bullet Point"):
            # Remove "Bullet Point X: "
            cleaned_line = re.sub(r'Bullet Point \d+: ', '', line).strip()
            bullet_points.append(cleaned_line)

    # Combine item name and bullet points
    full_text = item_name
    if bullet_points:
        full_text += " " + " ".join(bullet_points)

    # 1. Remove special characters (e.g., ®, ™, |, ~)
    full_text = re.sub(r'[®™|~]', '', full_text)

    # 2. Normalize whitespace (replace multiple spaces with a single one)
    full_text = re.sub(r'\s+', ' ', full_text).strip()

    return full_text

# Process training data
train_df = pd.read_csv('dataset/train_filtered.csv')
train_df['catalog_content'] = train_df['catalog_content'].apply(clean_catalog_content)
train_df.to_csv('dataset/train_filtered_cleaned.csv', index=False)
print("Cleaned training data saved to dataset/train_filtered_cleaned.csv")

# Process test data
test_df = pd.read_csv('dataset/test.csv')
test_df['catalog_content'] = test_df['catalog_content'].apply(clean_catalog_content)
test_df.to_csv('dataset/test_filtered_cleaned.csv', index=False)
print("Cleaned test data saved to dataset/test_filtered_cleaned.csv")




