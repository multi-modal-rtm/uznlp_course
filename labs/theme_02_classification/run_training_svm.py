import sys
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

features_path = os.path.join(project_root, 'data/processed/features/tfidf_features.csv')
data_path = os.path.join(project_root, 'data/processed/cleaned_news.csv')

print("⚔️  3-Mavzu: SVM (Support Vector Machine) Modeli...")

# --- DATA ---
X_df = pd.read_csv(features_path)
full_df = pd.read_csv(data_path).dropna(subset=['clean_text']).reset_index(drop=True)
y_df = full_df['category']

# --- SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=y_df)

# --- PIPELINE ---
# SVM (LinearSVC) matnlar uchun qirol hisoblanadi.
pipeline = Pipeline([
    ('feature_selection', SelectKBest(chi2, k=2000)), # Eng muhim 2000 so'zni oladi
    ('clf', LinearSVC(class_weight='balanced', random_state=42, C=0.5, dual='auto'))
])

# CalibratedClassifierCV - SVM ga ehtimollik (probability) qo'shadi, aniqlikni oshiradi
model = CalibratedClassifierCV(pipeline)

print("\n⚙️  Model o'qitilmoqda (bu biroz vaqt oladi)...")
model.fit(X_train, y_train)

# --- EVALUATION ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🏆 Aniqlik (Accuracy): {accuracy * 100:.2f}%")
print("-" * 60)
print(classification_report(y_test, y_pred))
print("-" * 60)

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('SVM Natijalari')
plt.ylabel('Asl')
plt.xlabel('Bashorat')
plt.show()

# Modelni saqlash
joblib.dump(model, os.path.join(project_root, 'models/svm_model.pkl'))
print("💾 Model saqlandi: models/svm_model.pkl")