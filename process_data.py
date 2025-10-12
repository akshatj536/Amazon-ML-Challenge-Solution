
import pandas as pd
import langextract as lx
import csv
import os

# Define the extraction schema
instructions = """
Extract product information from the text.
The fields to extract are:
- Item Name
- Value
- Unit
- Bullet Point 1
- Bullet Point 2
- Bullet Point 3
- Bullet Point 4
- Bullet Point 5
"""

# Create examples for langextract
example1 = lx.data.ExampleData(
    text="""Item Name: La Victoria Green Taco Sauce Mild, 12 Ounce (Pack of 6)
Value: 72.0
Unit: Fl Oz
""",
    extractions=[
        lx.data.Extraction(
            extraction_class="Item Name",
            extraction_text="La Victoria Green Taco Sauce Mild, 12 Ounce (Pack of 6)",
        ),
        lx.data.Extraction(
            extraction_class="Value",
            extraction_text="72.0",
        ),
        lx.data.Extraction(
            extraction_class="Unit",
            extraction_text="Fl Oz",
        ),
    ]
)

example2 = lx.data.ExampleData(
    text="""Item Name: Salerno Cookies, The Original Butter Cookies, 8 Ounce (Pack of 4)
Bullet Point 1: Original Butter Cookies: Classic butter cookies made with real butter
Bullet Point 2: Variety Pack: Includes 4 boxes with 32 cookies total
Bullet Point 3: Occasion Perfect: Delicious cookies for birthdays, weddings, anniversaries
Bullet Point 4: Shareable Treats: Fun to give and enjoy with friends and family
Bullet Point 5: Salerno Brand: Trusted brand of delicious butter cookies since 1925
Value: 32.0
Unit: Ounce
""",
    extractions=[
        lx.data.Extraction(
            extraction_class="Item Name",
            extraction_text="Salerno Cookies, The Original Butter Cookies, 8 Ounce (Pack of 4)",
        ),
        lx.data.Extraction(
            extraction_class="Bullet Point 1",
            extraction_text="Original Butter Cookies: Classic butter cookies made with real butter",
        ),
        lx.data.Extraction(
            extraction_class="Bullet Point 2",
            extraction_text="Variety Pack: Includes 4 boxes with 32 cookies total",
        ),
        lx.data.Extraction(
            extraction_class="Bullet Point 3",
            extraction_text="Occasion Perfect: Delicious cookies for birthdays, weddings, anniversaries",
        ),
        lx.data.Extraction(
            extraction_class="Bullet Point 4",
            extraction_text="Shareable Treats: Fun to give and enjoy with friends and family",
        ),
        lx.data.Extraction(
            extraction_class="Bullet Point 5",
            extraction_text="Salerno Brand: Trusted brand of delicious butter cookies since 1925",
        ),
        lx.data.Extraction(
            extraction_class="Value",
            extraction_text="32.0",
        ),
        lx.data.Extraction(
            extraction_class="Unit",
            extraction_text="Ounce",
        ),
    ]
)


import time

# Process the data
input_path = '/teamspace/studios/this_studio/dataset/train.csv'
output_path = '/teamspace/studios/this_studio/dataset/train_processed.csv'

df = pd.read_csv(input_path)

# Prepare the output file
with open(output_path, 'w', newline='') as csvfile:
    fieldnames = ['sample_id', 'image_link', 'price', 'Item Name', 'Value', 'Unit', 'Bullet Point 1', 'Bullet Point 2', 'Bullet Point 3', 'Bullet Point 4', 'Bullet Point 5']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for index, row in df.iterrows():
        text_to_process = row['catalog_content'].strip()
        
        result = lx.extract(
            text_or_documents=text_to_process,
            prompt_description=instructions,
            examples=[example1, example2],
            api_key=os.environ.get("LANGEXTRACT_API_KEY"),
        )
        
        processed_data = {
            'sample_id': row['sample_id'],
            'image_link': row['image_link'],
            'price': row['price'],
        }
        
        if result and result.extractions:
            for extraction in result.extractions:
                if extraction.extraction_class in fieldnames:
                    processed_data[extraction.extraction_class] = extraction.extraction_text

        writer.writerow(processed_data)
        
        # Print progress
        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1}/{len(df)} rows")
        
        time.sleep(6)

print(f"Processing complete. Output saved to {output_path}")
