import sys
import os
import pandas as pd
import joblib
import numpy as np

# --- SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Yuklash
vectorizer = joblib.load(os.path.join(project_root, 'models/tfidf_vectorizer.pkl'))
features = pd.read_csv(os.path.join(project_root, 'data/processed/features/tfidf_features.csv'))

# So'zlarning umumiy og'irligini hisoblash
print("📊 So'zlarni tahlil qilish...")
sum_scores = features.sum(axis=0)
top_words = sum_scores.sort_values(ascending=False).head(30)

print("\n--- 🏆 MODEL UCHUN ENG MUHIM 30 SO'Z ---")
print(top_words)