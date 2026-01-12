import sys
import os
import pandas as pd
import re

# --- SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
data_path = os.path.join(project_root, 'data/raw/uzbek_news.csv')
output_path = os.path.join(project_root, 'data/processed/cleaned_news.csv')

# --- O'ZBEK TILI STEMMERI ---
SUFFIXES = [
    "larning", "larniki", "larimiz", "laringiz", "larim", "laring",
    "lardan", "larda", "larga", "larni", "likni", "likka", "likda",
    "ning", "niki", "imiz", "miz", "ngiz", "ingiz", "si", "ni", "ni",
    "dan", "da", "ga", "ka", "qa", "la", "lar", "lik", "siz", "iy", 
    "dagi", "dagi", "dir", "moq", "yapti", "gan", "digan", "sa"
]

def simple_uzbek_stemmer(word):
    word = word.strip()
    if len(word) < 4: return word
    
    has_changed = True
    while has_changed:
        has_changed = False
        for suffix in SUFFIXES:
            if word.endswith(suffix):
                if len(word) - len(suffix) >= 3:
                    word = word[:-len(suffix)]
                    has_changed = True
                    break
    return word

def clean_text_surgical(text):
    if not isinstance(text, str): return ""
    
    # 1. KUN.UZ SHABLONLARINI QIRQISH
    # Boshidagi "Qo'shimcha funksionallar..." dan to "... o'qiladi" gacha
    text = re.sub(r"Qoʻshimcha funksionallar.*?daqiqa o‘qiladi", " ", text, flags=re.DOTALL)
    # Oxiridagi "Tayyorlagan: Mavzuga oid"
    text = re.sub(r"Tayyorlagan:.*", " ", text, flags=re.DOTALL)
    text = re.sub(r"Mavzuga oid.*", " ", text, flags=re.DOTALL)

    # 2. DARYO.UZ SHABLONLARINI QIRQISH
    # Boshidagi valyuta kurslari va menyu
    text = re.sub(r"Toshkent USD:.*?Copy link", " ", text, flags=re.DOTALL)
    # Oxiridagi "Izohlar..." dan boshlab hammasini o'chirish
    text = re.sub(r"Izohlar Izoh qoldirish uchun.*", " ", text, flags=re.DOTALL)
    # "Simple Networking Solutions" va boshqa copyrightlar
    text = re.sub(r"© «Simple Networking Solutions».*", " ", text, flags=re.DOTALL)

    # 3. XABAR.UZ va UMUMIY REKLAMALAR
    text = re.sub(r"Reklama huquqi asosida", " ", text)
    text = re.sub(r"Saytda imloviy yoki uslubiy xato.*", " ", text, flags=re.DOTALL)
    
    # 4. STANDART TOZALASH
    text = text.lower()
    text = re.sub(r'http\S+', '', text) # URL
    text = re.sub(r'<.*?>', '', text)   # HTML
    text = re.sub(r"[^a-z‘'’a-zA-Z\s]", ' ', text) # Raqam va belgilar
    
    # 5. SO'ZLARNI AJRATISH VA STEMMING
    words = text.split()
    cleaned_words = []
    
    # Qisqa so'zlarni va qolgan-qutgan axlatlarni o'chirish
    stopwords = {"foto", "video", "manba", "toshkent", "yil", "yili"}
    
    for w in words:
        if len(w) > 2 and w not in stopwords:
            stemmed = simple_uzbek_stemmer(w)
            cleaned_words.append(stemmed)
            
    return " ".join(cleaned_words)

def main():
    print(f"🧹 Jarrohlik amaliyoti boshlandi (Surgical Cleaning)...")
    df = pd.read_csv(data_path)
    print(f"   Yuklandi: {len(df)} ta maqola")
    
    df['clean_text'] = df['body'].apply(clean_text_surgical)
    
    # Bo'shab qolganlarini o'chiramiz
    df = df[df['clean_text'].str.len() > 10]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Toza ma'lumot saqlandi: {len(df)} ta")
    
    # NAMUNA KO'RSATISH
    print("\n--- NAMUNA (Jarrohlikdan keyin) ---")
    print(df['clean_text'].iloc[0][:300])

if __name__ == "__main__":
    main()













# import sys
# import os
# import pandas as pd
# import yaml
# import time

# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '../../'))
# if project_root not in sys.path:
#     sys.path.append(project_root)

# from src.uznlp.preprocessing.pipeline import TextPreprocessor

# def load_config():
#     config_path = os.path.join(project_root, 'config/settings.yaml')
#     with open(config_path, 'r') as f:
#         return yaml.safe_load(f)

# def load_stopwords(config):
#     path = os.path.join(project_root, config['data']['stopwords_path'])
#     if os.path.exists(path):
#         with open(path, 'r', encoding='utf-8') as f:
#             return [line.strip() for line in f]
#     return []

# def main():
#     print("Starting Pipeline for uzbek_news.csv...")
#     config = load_config()

#     input_path = os.path.join(project_root, config['data']['raw_path'])
#     output_path = os.path.join(project_root, config['data']['processed_path'])

#     if not os.path.exists(input_path):
#         print(f"Error: File not found: {input_path}")
#         return
        
#     df = pd.read_csv(input_path)
#     print(f"Loaded {len(df)} articles.")

#     stopwords = load_stopwords(config)
#     processor = TextPreprocessor(stopwords=stopwords)

#     print("Cleaning text (removing HTML, fixing apostrophes)...")
#     df['body'] = df['body'].fillna("")
#     df['clean_text'] = df['body'].apply(processor.process)
#     df['token_count'] = df['clean_text'].apply(lambda x: len(x.split()))

#     cols_to_save = ['category', 'title', 'body', 'clean_text', 'token_count']
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     df[cols_to_save].to_csv(output_path, index=False)
    
#     print(f"Done! Saved to: {output_path}")
#     print(f"Average Tokens: {df['token_count'].mean():.1f}")

# if __name__ == "__main__":
#     main()