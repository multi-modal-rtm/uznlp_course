import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
features_path = os.path.join(project_root, 'data/processed/features/tfidf_features.csv')
data_path = os.path.join(project_root, 'data/processed/cleaned_news.csv')

print("⚡ 3-Mavzu: SGD Classifier (Education Diagnostikasi bilan)...")

# --- DATA ---
X_df = pd.read_csv(features_path)
full_df = pd.read_csv(data_path).dropna(subset=['clean_text']).reset_index(drop=True)
y_df = full_df['category']

# --- SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=y_df)

# --- PIPELINE ---
# SGD bu juda moslashuvchan algoritm.
pipeline = Pipeline([
    ('feature_selection', SelectKBest(chi2, k=5000)), 
    ('clf', SGDClassifier(
        loss='modified_huber',  # Probabilistic SVM
        penalty='l2',
        alpha=1e-3,             # Regularization
        random_state=42,
        max_iter=2000,
        class_weight='balanced'
    ))
])

print("\n⚙️  Model o'qitilmoqda...")
pipeline.fit(X_train, y_train)

# --- EVALUATION ---
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🏆 Aniqlik (Accuracy): {accuracy * 100:.2f}%")
print("-" * 60)
print(classification_report(y_test, y_pred))
print("-" * 60)

# --- DIAGNOSTIKA: EDUCATION UCHUN KALIT SO'ZLAR ---
print("\n🔍 EDUCATION kategoriyasi uchun eng muhim so'zlar:")
try:
    # Feature nomlarini olish
    feature_names = X_df.columns
    # SelectKBest tanlaganlarini olish
    mask = pipeline.named_steps['feature_selection'].get_support()
    new_features = feature_names[mask]
    
    # Model koeffitsientlari
    clf = pipeline.named_steps['clf']
    # Education klassining indeksini topamiz
    edu_idx = list(clf.classes_).index('education')
    
    # Eng katta vaznga ega so'zlar
    top20_idx = np.argsort(clf.coef_[edu_idx])[-20:]
    top20_words = [new_features[i] for i in top20_idx]
    
    print(top20_words)
    print("\nAgar bu yerda 'maktab', 'universitet' chiqsa -> Model tuzaldi!")
except Exception as e:
    print(f"Diagnostika xatosi: {e}")

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens',
            xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
plt.title('SGD Natijalari')
plt.ylabel('Asl')
plt.xlabel('Bashorat')
plt.show()