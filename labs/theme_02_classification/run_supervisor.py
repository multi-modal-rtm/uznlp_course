import sys
import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
features_path = os.path.join(project_root, 'data/processed/features/tfidf_features.csv')
data_path = os.path.join(project_root, 'data/processed/cleaned_news.csv')
model_path = os.path.join(project_root, 'models/svm_model.pkl')

X_df = pd.read_csv(features_path)
full_df = pd.read_csv(data_path).dropna(subset=['clean_text']).reset_index(drop=True)
y_true = full_df['category']

X_train, X_test, y_train, y_test = train_test_split(X_df, y_true, test_size=0.2, random_state=42, stratify=y_true)
indices_test = y_test.index

model = joblib.load(model_path)
y_pred_model = model.predict(X_test) 

print(f"Modelning o'zi (Sof): {accuracy_score(y_test, y_pred_model)*100:.2f}%")

def apply_supervisor_rules(text, predicted_label):
    text = text.lower()
    
    # 1-QOIDA: Agar "maktab", "universitet" bo'lsa -> Bu 99% EDUCATION
    edu_keywords = ['maktab', 'universitet', 'talaba', 'o‘quvchi', 'rektor', 'grant', 'imtihon', 'vazir']
    # Kamida 2 ta kalit so'z qatnashgan bo'lsa
    matches = sum(1 for word in edu_keywords if word in text)
    if matches >= 2:
        return 'education'

    # 2-QOIDA: Agar "futbol", "gol", "liga" bo'lsa -> SPORT
    sport_keywords = ['futbol', 'gol', 'chempionat', 'liga', 'murabbiy', 'stadion', 'medal']
    matches = sum(1 for word in sport_keywords if word in text)
    if matches >= 2:
        return 'sport'

    # 3-QOIDA: Agar "bank", "kredit", "dollar" bo'lsa -> FINANCE
    fin_keywords = ['markaziy bank', 'kredit', 'inflyatsiya', 'trillion', 'eksport', 'import']
    for kw in fin_keywords:
        if kw in text:
            return 'finance'

    return predicted_label

y_pred_supervised = []

for i, idx in enumerate(indices_test):
    original_text = full_df.loc[idx, 'clean_text']
    model_prediction = y_pred_model[i]

    final_decision = apply_supervisor_rules(original_text, model_prediction)
    y_pred_supervised.append(final_decision)

acc_improved = accuracy_score(y_test, y_pred_supervised)
print(f"Nazoratchi bilan: {acc_improved*100:.2f}%")
print(f"O'sish: {((acc_improved - accuracy_score(y_test, y_pred_model)) * 100):.2f}%")

print("-" * 60)
print(classification_report(y_test, y_pred_supervised))
print("-" * 60)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_supervised), annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Qoidalarga asoslangan model natijalari')
plt.ylabel('Asl')
plt.xlabel('Bashorat')
plt.show()