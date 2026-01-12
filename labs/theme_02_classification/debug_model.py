import sys
import os
import pandas as pd
import joblib

# --- SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Yo'llar
data_path = os.path.join(project_root, 'data/processed/cleaned_news.csv')
model_path = os.path.join(project_root, 'models/tfidf_vectorizer.pkl')

print("🕵️‍♂️ DIAGNOSTIKA BOSHLANDI...\n")

# 1. Ma'lumotlarni tekshiramiz
df = pd.read_csv(data_path)
print(f"📊 Jami maqolalar: {len(df)}")
print(f"Bosh qatorlar:\n{df['clean_text'].head(3)}\n")

# 2. Maqolalar haqiqatan ham "toza"mi?
empty_docs = df[df['clean_text'].isna() | (df['clean_text'] == "")]
if len(empty_docs) > 0:
    print(f"⚠️ DIQQAT: {len(empty_docs)} ta maqola bo'm-bo'sh! (Parsing xatosi)")
else:
    print("✅ Barcha maqolalarda matn mavjud.")

# 3. Model "Lug'ati"ni tekshiramiz (Eng muhim qism)
if not os.path.exists(model_path):
    print("❌ Vektorizator topilmadi!")
    sys.exit(1)

vectorizer = joblib.load(model_path)
feature_names = vectorizer.get_feature_names_out()

print(f"\n🧠 Model biladigan so'zlar soni: {len(feature_names)}")
print("\n--- Model uchun eng 'mashhur' 50 ta so'z ---")
# Bu yerda biz shunchaki birinchi 50 ta so'zni emas, 
# balki tasodifiy 50 tasini ko'rsak yaxshi bo'lardi, 
# lekin alifbo tartibida boshini ko'rish ham yetarli.
print(feature_names[:50])

print("\n--- XULOSA ---")
print("Agar yuqoridagi ro'yxatda 'va', 'bilan', 'uchun' kabi so'zlarni ko'rsangiz -> Stopwords ishlamagan.")
print("Agar 'div', 'class', 'href' ko'rsangiz -> Tozalash (Cleaning) ishlamagan.")
print("Agar 'dollor', 'futbol', 'maktab' ko'rsangiz -> Hammasi joyida, modelni kuchaytirish kerak.")