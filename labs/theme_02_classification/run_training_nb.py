import sys
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

data_path = os.path.join(project_root, 'data/processed/cleaned_news.csv')
features_path = os.path.join(project_root, 'data/processed/features/tfidf_features.csv')
models_path = os.path.join(project_root, 'models')

print("2-Mavzu: Klassifikatsiya Modeli (Naive Bayes) ishga tushmoqda...")

if not os.path.exists(features_path):
    print(f"Xatolik: {features_path} topilmadi. Theme 1 ni qayta ishga tushiring.")
    sys.exit(1)

print("Ma'lumotlar yuklanmoqda...")
# X 
X_df = pd.read_csv(features_path)
# y (original fayldan 'category' ustuni)
y_df = pd.read_csv(data_path)['category']

# X va y qatorlari soni teng bo'lishi shart
if len(X_df) != len(y_df):
    print("Xatolik: Vektorlar va Kategoriyalar soni mos kelmadi!")
    sys.exit(1)

print(f"Jami ma'lumotlar: {len(X_df)} ta")
print(f"Kategoriyalar: {y_df.unique()}")

print("\n Ma'lumotlar 80/20 nisbatda ajratilmoqda...")
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

print(f"   O'quv (Train) ma'lumotlari soni: {len(X_train)} ta")
print(f"   Test ma'lumotlari soni:   {len(X_test)} ta")

print("\nNaive Bayes modeli o'qitilmoqda ...")
model = MultinomialNB()
model.fit(X_train, y_train)
print("Model o'qidi!")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"NB modeli aniqligi (Accuracy): {accuracy * 100:.2f}%")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

model_save_path = os.path.join(models_path, 'naive_bayes_model.pkl')
joblib.dump(model, model_save_path)
print(f"Model saqlandi: {model_save_path}")

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Bashorat (Predicted)')
plt.ylabel('Asl (Actual)')
plt.show()