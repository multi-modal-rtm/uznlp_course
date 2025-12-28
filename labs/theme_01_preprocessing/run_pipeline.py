import sys
import os
import pandas as pd
import yaml
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.uznlp.preprocessing.pipeline import TextPreprocessor

def load_config():
    config_path = os.path.join(project_root, 'config/settings.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_stopwords(config):
    path = os.path.join(project_root, config['data']['stopwords_path'])
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]
    return []

def main():
    print("Starting Pipeline for uzbek_news.csv...")
    config = load_config()

    input_path = os.path.join(project_root, config['data']['raw_path'])
    output_path = os.path.join(project_root, config['data']['processed_path'])

    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return
        
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} articles.")

    stopwords = load_stopwords(config)
    processor = TextPreprocessor(stopwords=stopwords)

    print("Cleaning text (removing HTML, fixing apostrophes)...")
    df['body'] = df['body'].fillna("")
    df['clean_text'] = df['body'].apply(processor.process)
    df['token_count'] = df['clean_text'].apply(lambda x: len(x.split()))

    cols_to_save = ['category', 'title', 'body', 'clean_text', 'token_count']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[cols_to_save].to_csv(output_path, index=False)
    
    print(f"Done! Saved to: {output_path}")
    print(f"Average Tokens: {df['token_count'].mean():.1f}")

if __name__ == "__main__":
    main()