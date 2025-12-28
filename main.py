import argparse
import pandas as pd
import yaml
from src.uznlp.preprocessing.pipeline import TextPreprocessor
from src.uznlp.vectorization.embedder import Vectorizer

def load_config(path='config/settings.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_stopwords(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f]

def main():
    # 1. Setup
    config = load_config()
    stopwords = load_stopwords(config['data']['stopwords_path'])
    
    # 2. Load Data (Simulated for this snippet)
    print("Loading raw data...")
    # df = pd.read_csv(config['data']['raw_path'])
    raw_data = [
        "O'zbekiston Respublikasi poytaxti Toshkent shahri.",
        "Бугун ҳаво ҳарорати кескин ўзгаради.", # Mixed Cyrillic
        "Sun'iy intellekt kelajak texnologiyasidir."
    ]
    
    # 3. Preprocessing Pipeline
    print("Running preprocessing pipeline...")
    processor = TextPreprocessor(stopwords=stopwords)
    
    # Apply to all documents
    clean_corpus = [processor.process(doc) for doc in raw_data]
    print(f"Cleaned Sample: {clean_corpus[1]}") # Output: "bugun havo harorati keskin o'zgaradi"
    
    # 4. Vectorization
    print("Vectorizing data...")
    vectorizer = Vectorizer(method='tfidf')
    df_vectors, features = vectorizer.fit_transform(clean_corpus)
    
    print("Top features:", features[:5])
    print("Shape:", df_vectors.shape)
    
    # 5. Save Artifacts (Production Step)
    # joblib.dump(vectorizer.model, 'models/tfidf_v1.pkl')

if __name__ == "__main__":
    main()