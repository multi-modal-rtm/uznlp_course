import sys
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# --- FIX: ROBUST PATH SETUP ---
# We use __file__ to get the location of THIS script, then go up 2 levels
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))

if project_root not in sys.path:
    sys.path.append(project_root)

# Load processed data
input_path = os.path.join(project_root, 'data/processed/cleaned_news.csv')

print(f"📂 Looking for data at: {input_path}")

if not os.path.exists(input_path):
    print("❌ Error: File not found.")
    print("Please run 'python labs/theme_01_preprocessing/run_pipeline.py' first!")
    sys.exit(1)

df = pd.read_csv(input_path)
# Handle potential NaNs by filling with empty string
corpus = df['clean_text'].fillna("").tolist()

print(f"✅ Loaded {len(corpus)} documents.")

# --- 1. Bag of Words (CountVectorizer) ---
print("\n--- 📊 Bag of Words (Frequency Count) ---")
# min_df=2: Ignore words appearing in only 1 document (likely typos or rare names)
count_vec = CountVectorizer(min_df=2, max_features=20) 
bow_matrix = count_vec.fit_transform(corpus)

print("Top 20 Frequent Words:", count_vec.get_feature_names_out())
print("Matrix Shape:", bow_matrix.shape)

# --- 2. TF-IDF (Importance Score) ---
print("\n--- ⚖️ TF-IDF (Importance Weighted) ---")
tfidf_vec = TfidfVectorizer(min_df=2, max_features=20)
tfidf_matrix = tfidf_vec.fit_transform(corpus)

print("Top 20 Important Words:", tfidf_vec.get_feature_names_out())
print("First Doc Vector (Dense):\n", tfidf_matrix[0].toarray())

print("\nTheme 1 Complete! Text is now mathematical vectors.")

# Save for Theme 2
# import joblib
# joblib.dump(tfidf_vec, 'models/tfidf_vectorizer.pkl')