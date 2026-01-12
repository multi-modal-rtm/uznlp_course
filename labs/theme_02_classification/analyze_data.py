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
features_path = os.path.join(project_root, 'data/processed/features/tfidf_features.csv')

print("💀 XATOLLAR TAHLILI (DATA X-RAY)...")

# Ma'lumotlarni yuklaymiz
df = pd.read_csv(data_path).dropna(subset=['clean_text']).reset_index(drop=True)
# Asl (Xom) matnni ham ko'rishimiz kerak, shuning uchun raw datani ham o'qiymiz (agar bo'lsa)
# Hozircha clean_text ga qarab turamiz.

print(f"Jami maqolalar: {len(df)}")

# Bizga "Technology" deb belgilangan lekin aslida boshqa narsa bo'lgan maqolalar qiziq.
# Yoki "Education" deb xato topilganlar.

print("\n--- 🧐 NAMUNALARNI TEKSHIRISH ---")

# Tasodifiy 5 ta maqolani to'liq matnini o'qiymiz
sample = df.sample(5)

for idx, row in sample.iterrows():
    print(f"\n[ID: {idx}] Kategoriya: {row['category'].upper()}")
    print(f"MATN BOSHI (150 belgi): {str(row['clean_text'])[:150]}...")
    print("-" * 50)

print("\n\n--- ⚠️ GUMONLI QISQA MAQOLALAR ---")
# Matni 50 ta so'zdan kam bo'lgan maqolalar bormi?
df['word_count'] = df['clean_text'].apply(lambda x: len(str(x).split()))
short_docs = df[df['word_count'] < 20]

if not short_docs.empty:
    print(f"DIQQAT: {len(short_docs)} ta maqola juda qisqa (< 20 so'z)!")
    print(short_docs[['category', 'clean_text']].head(10))
else:
    print("Juda qisqa maqolalar yo'q.")