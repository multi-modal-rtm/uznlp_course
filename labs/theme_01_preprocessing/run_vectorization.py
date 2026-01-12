import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# --- SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
data_path = os.path.join(project_root, 'data/processed/cleaned_news.csv')
features_path = os.path.join(project_root, 'data/processed/features')
models_path = os.path.join(project_root, 'models')

os.makedirs(features_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

df = pd.read_csv(data_path).dropna(subset=['clean_text'])
corpus = df['clean_text'].tolist()

print(f"📂 Loaded {len(corpus)} documents.")

# --- 1. TF-IDF (Asosiy Matn) ---
print("\n--- 1. Generating TF-IDF ---")
tfidf_vec = TfidfVectorizer(
    min_df=3, 
    max_df=0.5, 
    max_features=3000, 
    norm='l2',
    sublinear_tf=True # Logarifmik hisoblash
)
tfidf_matrix = tfidf_vec.fit_transform(corpus)

# --- 2. QO'LBOLA KALIT SO'ZLAR (Rule-Based Features) ---
print("\n--- 2. Adding Custom Keyword Features ---")

# Har bir kategoriya uchun "Oltin so'zlar"
keywords = {
    'education': ['maktab', 'universitet', 'talaba', 'o‘quvchi', 'imtihon', 'grant', 'diplom', 'rektor', 'ta’lim', 'institut', 'vazir', 'bog‘cha'],
    'auto': ['avtomobil', 'mashina', 'haydovchi', 'yo‘l', 'gm', 'chevrolet', 'byd', 'kia', 'elektromobil', 'rul', 'motor', 'transport'],
    'finance': ['bank', 'dollar', 'kredit', 'sum', 'valyuta', 'inflyatsiya', 'markaziy', 'investitsiya', 'moliya', 'soliq', 'eksport', 'biznes'],
    'sport': ['futbol', 'gol', 'o‘yin', 'chempionat', 'liga', 'murabbiy', 'stadion', 'terma', 'jamoa', 'medal', 'sport', 'olimpiada'],
    'technology': ['internet', 'dastur', 'ilovasi', 'smartfon', 'sun’iy', 'intellekt', 'robot', 'google', 'apple', 'telegram', 'foydalanuvchi', 'tarmoq']
}

def count_keywords(text, keyword_list):
    count = 0
    words = text.split()
    for word in words:
        if word in keyword_list:
            count += 1
    return count

# Qo'shimcha ustunlarni yaratamiz
extra_features = []
for text in corpus:
    row = []
    # Har bir kategoriya uchun so'zlarni sanaymiz
    row.append(count_keywords(text, keywords['education']))
    row.append(count_keywords(text, keywords['auto']))
    row.append(count_keywords(text, keywords['finance']))
    row.append(count_keywords(text, keywords['sport']))
    row.append(count_keywords(text, keywords['technology']))
    extra_features.append(row)

extra_features_matrix = np.array(extra_features)

# --- 3. BIRLASHTIRISH ---
# TF-IDF matritsasi + (Kalit so'zlar * 5)
# Biz kalit so'zlarni 5 ga ko'paytiramiz, shunda ular model uchun 5 baravar muhimroq bo'ladi!
final_matrix = hstack([tfidf_matrix, extra_features_matrix * 5]) 

print(f"✅ Final Matrix Shape: {final_matrix.shape}")
print(f"   (3000 ta TF-IDF so'zi + 5 ta maxsus hisoblagich ustuni)")

# Saqlash (Pickle formatida, chunki sparse matrix CSV ga sig'maydi)
joblib.dump(tfidf_vec, os.path.join(models_path, 'tfidf_vectorizer.pkl'))
joblib.dump(final_matrix, os.path.join(features_path, 'combined_features.pkl'))
print("✅ Saved features to combined_features.pkl")













# import sys
# import os
# import pandas as pd
# import joblib
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# # --- SETUP ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# data_path = os.path.join(project_root, 'data/processed/cleaned_news.csv')
# features_path = os.path.join(project_root, 'data/processed/features')
# models_path = os.path.join(project_root, 'models')

# os.makedirs(features_path, exist_ok=True)
# os.makedirs(models_path, exist_ok=True)

# df = pd.read_csv(data_path).dropna(subset=['clean_text'])
# corpus = df['clean_text'].tolist()

# print(f"📂 Loaded {len(corpus)} documents.")

# # Faqat harflar
# REGEX_PATTERN = r'(?u)\b[a-zA-Z\']{3,}\b'

# # ==========================================
# # YANGILIK: N-GRAM (1, 2) va MAX_DF=0.5
# # ==========================================
# print("\n--- 1. Generating Bag of Words ---")
# count_vec = CountVectorizer(
#     min_df=3,        # Kamida 3 ta maqolada bo'lsin (xatolarni o'chiramiz)
#     max_df=0.5,      # 50% dan ko'p uchrasa - O'CHIR (umumiy so'zlar)
#     max_features=10000, # Ko'proq so'z olamiz
#     token_pattern=REGEX_PATTERN,
# )
# bow_matrix = count_vec.fit_transform(corpus)
# joblib.dump(count_vec, os.path.join(models_path, 'bow_vectorizer.pkl'))

# bow_df = pd.DataFrame(bow_matrix.toarray(), columns=count_vec.get_feature_names_out())
# bow_df.to_csv(os.path.join(features_path, 'bow_features.csv'), index=False)
# print(f"✅ Saved BoW (Shape: {bow_matrix.shape})")

# # ==========================================
# # 2. TF-IDF
# # ==========================================
# print("\n--- 2. Generating TF-IDF ---")
# tfidf_vec = TfidfVectorizer(
#     min_df=3, 
#     max_df=0.5, 
#     max_features=10000, 
#     token_pattern=REGEX_PATTERN,
#     sublinear_tf=True,
#     norm='l2'
# )
# tfidf_matrix = tfidf_vec.fit_transform(corpus)
# joblib.dump(tfidf_vec, os.path.join(models_path, 'tfidf_vectorizer.pkl'))

# tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vec.get_feature_names_out())
# tfidf_df.to_csv(os.path.join(features_path, 'tfidf_features.csv'), index=False)
# print(f"✅ Saved TF-IDF (Shape: {tfidf_matrix.shape})")













# # import sys
# # import os
# # import pandas as pd
# # import joblib
# # from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# # # --- SETUP ---
# # current_dir = os.path.dirname(os.path.abspath(__file__))
# # project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# # if project_root not in sys.path:
# #     sys.path.append(project_root)

# # input_path = os.path.join(project_root, 'data/processed/cleaned_news.csv')
# # features_path = os.path.join(project_root, 'data/processed/features')
# # models_path = os.path.join(project_root, 'models')

# # os.makedirs(features_path, exist_ok=True)
# # os.makedirs(models_path, exist_ok=True)

# # # Yuklash
# # if not os.path.exists(input_path):
# #     print(f"❌ Error: {input_path} not found.")
# #     sys.exit(1)

# # df = pd.read_csv(input_path)
# # df = df.dropna(subset=['clean_text'])
# # corpus = df['clean_text'].tolist()

# # print(f"📂 Loaded {len(corpus)} documents.")

# # # Faqat harflar (Raqamlarni ignor qilamiz)
# # REGEX_PATTERN = r'(?u)\b[a-zA-Z\']{3,}\b'

# # # ==========================================
# # # 1. BAG OF WORDS
# # # ==========================================
# # print("\n--- 1. Generating Bag of Words ---")
# # # max_df=0.5 -> Agar so'z 50% dan ko'p maqolada bo'lsa, o'chirilsin (footerlar ketadi)
# # # min_df=5 -> Agar so'z 5 tadan kam maqolada bo'lsa, o'chirilsin (noyob xatolar ketadi)
# # count_vec = CountVectorizer(
# #     min_df=3, 
# #     max_df=0.5, 
# #     max_features=3000, 
# #     token_pattern=REGEX_PATTERN
# # )
# # bow_matrix = count_vec.fit_transform(corpus)

# # joblib.dump(count_vec, os.path.join(models_path, 'bow_vectorizer.pkl'))
# # bow_df = pd.DataFrame(bow_matrix.toarray(), columns=count_vec.get_feature_names_out())
# # bow_df.to_csv(os.path.join(features_path, 'bow_features.csv'), index=False)
# # print(f"✅ Saved BoW (Shape: {bow_matrix.shape})")

# # # ==========================================
# # # 2. TF-IDF
# # # ==========================================
# # print("\n--- 2. Generating TF-IDF ---")
# # tfidf_vec = TfidfVectorizer(
# #     min_df=3, 
# #     max_df=0.5, 
# #     max_features=3000, 
# #     token_pattern=REGEX_PATTERN
# # )
# # tfidf_matrix = tfidf_vec.fit_transform(corpus)

# # joblib.dump(tfidf_vec, os.path.join(models_path, 'tfidf_vectorizer.pkl'))
# # tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vec.get_feature_names_out())
# # tfidf_df.to_csv(os.path.join(features_path, 'tfidf_features.csv'), index=False)
# # print(f"✅ Saved TF-IDF (Shape: {tfidf_matrix.shape})")

# # print("\n🎉 Vektorizatsiya tugadi! Shovqinlar avtomatik tozalandi.")













# # # import sys
# # # import os
# # # import pandas as pd
# # # import joblib
# # # from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# # # # --- PATH SETUP ---
# # # current_dir = os.path.dirname(os.path.abspath(__file__))
# # # project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# # # if project_root not in sys.path:
# # #     sys.path.append(project_root)

# # # # Paths
# # # input_path = os.path.join(project_root, 'data/processed/cleaned_news.csv')
# # # features_path = os.path.join(project_root, 'data/processed/features')
# # # models_path = os.path.join(project_root, 'models')

# # # os.makedirs(features_path, exist_ok=True)
# # # os.makedirs(models_path, exist_ok=True)

# # # # Load data
# # # if not os.path.exists(input_path):
# # #     print(f"❌ Error: {input_path} not found.")
# # #     sys.exit(1)

# # # df = pd.read_csv(input_path)
# # # # Bo'sh qatorlarni tashlab yuboramiz
# # # df = df.dropna(subset=['clean_text'])
# # # corpus = df['clean_text'].tolist()

# # # print(f"📂 Loaded {len(corpus)} documents.")

# # # # ==========================================
# # # # MUHIM O'ZGARISH: TOKEN_PATTERN
# # # # ==========================================
# # # # Biz faqat harflarni qabul qilamiz. Raqamlarni (0-9) umuman olmaymiz.
# # # # r'(?u)\b[a-zA-Z\']{3,}\b' -> Faqat harflar va apostrof, kamida 3 ta harf bo'lsin.
# # # REGEX_PATTERN = r'(?u)\b[a-zA-Z\']{3,}\b'

# # # # 1. BAG OF WORDS
# # # print("\n--- 1. Generating Bag of Words ---")
# # # # max_features ni 3000 ga ko'taramiz (so'z boyligi oshsin)
# # # count_vec = CountVectorizer(min_df=3, max_features=3000, token_pattern=REGEX_PATTERN)
# # # bow_matrix = count_vec.fit_transform(corpus)

# # # joblib.dump(count_vec, os.path.join(models_path, 'bow_vectorizer.pkl'))
# # # bow_df = pd.DataFrame(bow_matrix.toarray(), columns=count_vec.get_feature_names_out())
# # # bow_df.to_csv(os.path.join(features_path, 'bow_features.csv'), index=False)
# # # print(f"✅ Saved BoW (Shape: {bow_matrix.shape})")

# # # # 2. TF-IDF
# # # print("\n--- 2. Generating TF-IDF ---")
# # # tfidf_vec = TfidfVectorizer(min_df=3, max_features=3000, token_pattern=REGEX_PATTERN)
# # # tfidf_matrix = tfidf_vec.fit_transform(corpus)

# # # joblib.dump(tfidf_vec, os.path.join(models_path, 'tfidf_vectorizer.pkl'))
# # # tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vec.get_feature_names_out())
# # # tfidf_df.to_csv(os.path.join(features_path, 'tfidf_features.csv'), index=False)
# # # print(f"✅ Saved TF-IDF (Shape: {tfidf_matrix.shape})")

# # # print("\n🎉 Vektorizatsiya tugadi! Endi model raqamlarni ko'rmaydi.")













# # # # import sys
# # # # import os
# # # # import pandas as pd
# # # # import joblib 
# # # # from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# # # # current_dir = os.path.dirname(os.path.abspath(__file__))
# # # # project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# # # # if project_root not in sys.path:
# # # #     sys.path.append(project_root)

# # # # input_path = os.path.join(project_root, 'data/processed/cleaned_news.csv')
# # # # features_path = os.path.join(project_root, 'data/processed/features')
# # # # models_path = os.path.join(project_root, 'models')

# # # # os.makedirs(features_path, exist_ok=True)
# # # # os.makedirs(models_path, exist_ok=True)

# # # # if not os.path.exists(input_path):
# # # #     print(f"Error: {input_path} not found. Run pipeline first.")
# # # #     sys.exit(1)

# # # # df = pd.read_csv(input_path)
# # # # corpus = df['clean_text'].fillna("").tolist()
# # # # print(f"Loaded {len(corpus)} documents.")

# # # # # 1. BAG OF WORDS (BoW)
# # # # print("\n--- 1. Generating Bag of Words ---")
# # # # count_vec = CountVectorizer(min_df=2, max_features=1000) # Limit to top 1000 words
# # # # bow_matrix = count_vec.fit_transform(corpus)

# # # # # Save the Model
# # # # joblib.dump(count_vec, os.path.join(models_path, 'bow_vectorizer.pkl'))

# # # # # Save the Data
# # # # bow_df = pd.DataFrame(bow_matrix.toarray(), columns=count_vec.get_feature_names_out())
# # # # bow_df.to_csv(os.path.join(features_path, 'bow_features.csv'), index=False)
# # # # print(f"Saved BoW model and features (Shape: {bow_matrix.shape})")

# # # # # 2. TF-IDF
# # # # print("\n--- 2. Generating TF-IDF ---")
# # # # tfidf_vec = TfidfVectorizer(min_df=2, max_features=1000)
# # # # tfidf_matrix = tfidf_vec.fit_transform(corpus)

# # # # # Save the Model
# # # # joblib.dump(tfidf_vec, os.path.join(models_path, 'tfidf_vectorizer.pkl'))

# # # # # Save the Data
# # # # tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vec.get_feature_names_out())
# # # # tfidf_df.to_csv(os.path.join(features_path, 'tfidf_features.csv'), index=False)
# # # # print(f"Saved TF-IDF model and features (Shape: {tfidf_matrix.shape})")

# # # # print(f"\nOutputs saved to:\n - {features_path}\n - {models_path}")