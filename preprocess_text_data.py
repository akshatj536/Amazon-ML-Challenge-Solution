import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# downloads required once
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text_lemmatize_stem(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)       # remove URLs
    text = re.sub(r'<.*?>', ' ', text)                # remove HTML
    text = re.sub(r'\s+', ' ', text).strip()          # normalize whitespace

    tokens = word_tokenize(text)
    # keep alphabetic tokens only (removes punctuation tokens)
    tokens = [t for t in tokens if t.isalpha()]

    # lemmatize then stem
    tokens = [stemmer.stem(lemmatizer.lemmatize(t)) for t in tokens]

    return ' '.join(tokens)

# load, process, save
df = pd.read_csv('dataset/train_structured.csv')
df['catalog_content'] = df['catalog_content'].fillna('').astype(str).apply(preprocess_text_lemmatize_stem)
df.to_csv('dataset/train_filtered_cleaned_lemm_stem.csv', index=False)
