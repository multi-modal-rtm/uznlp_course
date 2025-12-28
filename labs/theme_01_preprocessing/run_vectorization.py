import sys
import os
import pandas as pd
import joblib 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

input_path = os.path.join(project_root, 'data/processed/cleaned_news.csv')
features_path = os.path.join(project_root, 'data/processed/features')
models_path = os.path.join(project_root, 'models')

os.makedirs(features_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

if not os.path.exists(input_path):
    print(f"Error: {input_path} not found. Run pipeline first.")
    sys.exit(1)

df = pd.read_csv(input_path)
corpus = df['clean_text'].fillna("").tolist()
print(f"Loaded {len(corpus)} documents.")

# 1. BAG OF WORDS (BoW)
print("\n--- 1. Generating Bag of Words ---")
count_vec = CountVectorizer(min_df=2, max_features=1000) # Limit to top 1000 words
bow_matrix = count_vec.fit_transform(corpus)

# Save the Model
joblib.dump(count_vec, os.path.join(models_path, 'bow_vectorizer.pkl'))

# Save the Data
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=count_vec.get_feature_names_out())
bow_df.to_csv(os.path.join(features_path, 'bow_features.csv'), index=False)
print(f"Saved BoW model and features (Shape: {bow_matrix.shape})")

# 2. TF-IDF
print("\n--- 2. Generating TF-IDF ---")
tfidf_vec = TfidfVectorizer(min_df=2, max_features=1000)
tfidf_matrix = tfidf_vec.fit_transform(corpus)

# Save the Model
joblib.dump(tfidf_vec, os.path.join(models_path, 'tfidf_vectorizer.pkl'))

# Save the Data
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vec.get_feature_names_out())
tfidf_df.to_csv(os.path.join(features_path, 'tfidf_features.csv'), index=False)
print(f"Saved TF-IDF model and features (Shape: {tfidf_matrix.shape})")

print(f"\nOutputs saved to:\n - {features_path}\n - {models_path}")