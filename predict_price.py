import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import numpy as np
import re

def smape(y_true, y_pred):
    return np.mean((2 * np.abs(y_pred - y_true)) / (np.abs(y_true) + np.abs(y_pred))) * 100

def clean_value_column(df):
    df['value_cleaned'] = df['value'].astype(str)
    df['value_cleaned'] = df['value_cleaned'].apply(
        lambda x: re.search(r'[\d\.]+', x).group(0) if re.search(r'[\d\.]+', x) else np.nan
    )
    df['value_cleaned'] = pd.to_numeric(df['value_cleaned'], errors='coerce').fillna(0)
    return df

def main():
    print("Loading data...")
    try:
        df = pd.read_csv('dataset/train_structured_single_column.csv')
    except FileNotFoundError:
        print("Error: dataset/train_structured_single_column.csv not found.")
        return

    print("Cleaning data...")
    df = clean_value_column(df)
    df['item_name'] = df['item_name'].fillna('')
    df['bullet_points'] = df['bullet_points'].fillna('')
    df['catalog_content'] = df['catalog_content'].fillna('')

    df['all_text'] = df['item_name'] + ' ' + df['bullet_points'] + ' ' + df['catalog_content']

    features = ['value_cleaned', 'all_text']
    target = 'price'

    X = df[features]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['value_cleaned']),
            ('txt', TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2)), 'all_text')
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(random_state=42, n_jobs=-1))
    ])

    print("Training LightGBM model...")
    pipeline.fit(X_train, y_train, regressor__callbacks=[lgb.log_evaluation(period=10)])

    print("Making predictions...")
    y_pred = pipeline.predict(X_val)

    smape_score = smape(y_val, y_pred)

    print(f"SMAPE score on the validation set (with LightGBM): {smape_score:.4f}")

if __name__ == '__main__':
    main()