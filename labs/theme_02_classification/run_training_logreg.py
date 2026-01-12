import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

features_path = os.path.join(project_root, 'data/processed/features/tfidf_features.csv')
data_path = os.path.join(project_root, 'data/processed/cleaned_news.csv')

print("🧠 2-Mavzu: Logistic Regression (Balanced + Chi2) ...")

# --- DATA ---
X_df = pd.read_csv(features_path)
full_df = pd.read_csv(data_path).dropna(subset=['clean_text']).reset_index(drop=True)
y_df = full_df['category']

# O'lcham tekshiruvi
if len(X_df) != len(y_df):
    print(f"❌ Mismatch: X={len(X_df)}, y={len(y_df)}")
    sys.exit(1)

# --- SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=y_df)
# stratify=y_df -> Train va Testda kategoriyalar proporsiyasi bir xil bo'lishini ta'minlaydi

# --- PIPELINE ---
# 1. Chi2: Eng muhim 3000 ta so'zni tanlab oladi (shovqinni tashlaydi)
# 2. LogisticRegression: class_weight='balanced' (kam ma'lumotli kategoriyalar uchun)
pipeline = Pipeline([
    ('feature_selection', SelectKBest(chi2, k=1000)), 
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0))
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

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens',
            xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
plt.title('Logistic Regression (Balanced)')
plt.ylabel('Asl')
plt.xlabel('Bashorat')
plt.show()













# import sys
# import os
# import pandas as pd
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # --- SETUP ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# features_path = os.path.join(project_root, 'data/processed/features/tfidf_features.csv')
# data_path = os.path.join(project_root, 'data/processed/cleaned_news.csv')

# print("🧠 2-Mavzu: Logistic Regression Modeli ishga tushmoqda...")

# # --- DATA YUKLASH ---
# # 1. Xususiyatlarni (X) yuklaymiz
# X_df = pd.read_csv(features_path)

# # 2. Kategoriyalarni (y) yuklaymiz
# # DIQQAT: Biz X ni yasashda bo'sh qatorlarni o'chirgan edik.
# # Shuning uchun y ni olishda ham aynan o'sha qatorlarni o'chirishimiz kerak.
# full_df = pd.read_csv(data_path)
# full_df = full_df.dropna(subset=['clean_text']) # Xuddi vectorization.py dagi kabi

# y_df = full_df['category']

# # Tekshiramiz
# print(f"   X hajmi: {X_df.shape}")
# print(f"   y hajmi: {y_df.shape}")

# if len(X_df) != len(y_df):
#     print("❌ Xatolik: O'lchamlar hali ham mos emas!")
#     # Indekslarni reset qilish yordam berishi mumkin
#     y_df = y_df.reset_index(drop=True)
#     if len(X_df) != len(y_df):
#         sys.exit(1)

# # --- TRAIN/TEST SPLIT ---
# X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

# print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# # --- MODEL (LOGISTIC REGRESSION) ---
# print("\n⚙️  Model o'qitilmoqda...")
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)

# # --- EVALUATION ---
# print("\n📝 Imtihon natijalari:")
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# print(f"🏆 Aniqlik (Accuracy): {accuracy * 100:.2f}%")
# print("-" * 60)
# print(classification_report(y_test, y_pred))
# print("-" * 60)

# # Confusion Matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens',
#             xticklabels=model.classes_, yticklabels=model.classes_)
# plt.title('Logistic Regression Natijalari')
# plt.ylabel('Asl Kategoriya')
# plt.xlabel('Bashorat')
# plt.show()













# # import sys
# # import os
# # import pandas as pd
# # import joblib
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# # import seaborn as sns
# # import matplotlib.pyplot as plt

# # current_dir = os.path.dirname(os.path.abspath(__file__))
# # project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# # if project_root not in sys.path:
# #     sys.path.append(project_root)

# # features_path = os.path.join(project_root, 'data/processed/features/tfidf_features.csv')
# # data_path = os.path.join(project_root, 'data/processed/cleaned_news.csv')

# # X_df = pd.read_csv(features_path)
# # y_df = pd.read_csv(data_path)['category']

# # X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

# # print("Logistic Regression modeli o'qitilmoqda ...")
# # model = LogisticRegression(max_iter=1000) 
# # model.fit(X_train, y_train)

# # y_pred = model.predict(X_test)
# # accuracy = accuracy_score(y_test, y_pred)

# # print(f"Logistic Regression modeli aniqligi (Accuracy): {accuracy * 100:.2f}%")
# # print(classification_report(y_test, y_pred))

# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens',
# #             xticklabels=model.classes_, yticklabels=model.classes_)
# # plt.title('Logistic Regression Natijalari')
# # plt.show()