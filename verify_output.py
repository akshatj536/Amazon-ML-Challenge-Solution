
import pandas as pd

# Ensure the full content of each column is displayed
pd.set_option('display.max_colwidth', None)

try:
    df = pd.read_csv("/teamspace/studios/this_studio/dataset/train_structured.csv")
    print("--- Verification of train_structured.csv ---")
    print(df[['item_name', 'bullet_points']])
except FileNotFoundError:
    print("Error: The file train_structured.csv was not found.")
except KeyError:
    print("Error: The file exists, but the 'item_name' or 'bullet_points' columns are missing.")
