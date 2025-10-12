import pandas as pd
import re

def structure_catalog_content(df):
    """
    Structures the 'catalog_content' column of a DataFrame into separate columns.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'catalog_content' column.

    Returns:
        pd.DataFrame: The DataFrame with new columns for the structured data.
    """
    df['item_name'] = df['catalog_content'].str.extract(r"Item Name: (.*?)\n")
    
    # Find all bullet points and join them
    df['bullet_points'] = df['catalog_content'].str.findall(r"Bullet Point \d+: (.*?)\n").str.join('\n')
    
    df['value'] = df['catalog_content'].str.extract(r"Value: (.*?)\n")
    df['unit'] = df['catalog_content'].str.extract(r"Unit: (.*?)\n")

    return df

def process_csv_in_chunks(input_path, output_path, chunksize=10000):
    """
    Processes a large CSV file in chunks.

    Args:
        input_path (str): The path to the input CSV file.
        output_path (str): The path to the aoutput CSV file.
        chunksize (int): The number of rows to process in each chunk.
    """
    # Create a new file for the output
    first_chunk = True
    # Use a try-except block to handle potential FileNotFoundError
    try:
        for chunk in pd.read_csv(input_path, chunksize=chunksize):
            processed_chunk = structure_catalog_content(chunk)
            if first_chunk:
                # For the first chunk, write the header
                processed_chunk.to_csv(output_path, index=False, mode='w')
                first_chunk = False
            else:
                # For subsequent chunks, append without the header
                processed_chunk.to_csv(output_path, index=False, mode='a', header=False)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")


if __name__ == '__main__':
    # Define the paths for the input and output files
    train_input_path = '/teamspace/studios/this_studio/dataset/train.csv'
    train_output_path = '/teamspace/studios/this_studio/dataset/train_structured_single_column.csv'
    test_input_path = '/teamspace/studios/this_studio/dataset/test.csv'
    test_output_path = '/teamspace/studios/this_studio/dataset/test_structured_single_column.csv'

    # Process the training and test files
    print("Processing train.csv...")
    process_csv_in_chunks(train_input_path, train_output_path)
    print("Processing test.csv...")
    process_csv_in_chunks(test_input_path, test_output_path)
    print("Processing complete.")