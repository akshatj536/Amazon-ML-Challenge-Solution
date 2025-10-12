
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import re

def smape(y_true, y_pred):
    # To avoid division by zero
    return np.mean((2 * np.abs(y_pred - y_true)) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

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

    features = ['value_cleaned', 'item_name']
    target = 'price'

    X = df[features]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['value_cleaned']),
            ('txt', TfidfVectorizer(stop_words='english', max_features=1000), 'item_name')
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    # Define a small grid of hyperparameters to search
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_leaf': [1, 2, 4]
    }

    print("Starting GridSearchCV... This may take a while.")
    # Note: Using SMAPE as a scorer for GridSearchCV is complex because it's not a built-in scorer.
    # We will use the default R^2 scorer for the grid search and then evaluate the best model with SMAPE.
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Best parameters found:", grid_search.best_params_)

    print("Making predictions with the best model...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)

    smape_score = smape(y_val, y_pred)

    print(f"SMAPE score on the validation set (with tuned RandomForest): {smape_score:.4f}")

if __name__ == '__main__':
    main()
