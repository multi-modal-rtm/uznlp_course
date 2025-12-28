# E'tibor bering biz bu moduldan darslar davomida foydalanmaymiz. 
# Biz \uznlp\preprocessing\stemmer_lib.py dan foydalanamiz. 
# Bu modulning maqsadi stemming qanday amalga oshirilishini ko'rsatish. 
# Stemming va lemmatizatsiyatning o'zbek tili uchun implementatsiyasini 
# https://github.com/UlugbekSalaev/UzMorphAnalyser orqali ko'rishingiz mumkin.

import re

class UzbekStemmer:
    def __init__(self):
        # So'z tarkibida navbat muhim! Avval uzunroq qo'shimchalarni olib tashlash kerak.
        self.suffixes = [
            # Egalik
            "imiz", "ingiz", "lari", "miz", "ngiz", 
            # Ko'plik
            "lar", 
            # Kelishik
            "ning", "ni", "ga", "da", "dan", 
            # Yasovchi
            "lik", "siz", "gi"
        ]

    def stem(self, word):
        """
        Takroriy ravishda so‘z oxiridagi qo‘shimchalarni olib tashlaydi.
        Agglyutinativ o‘zbek tili uchun oddiy qoidaga asoslangan yondashuv.
        """
        # We repeat the process because multiple suffixes can exist (e.g., maktab-lar-da)
        original_word = word
        cleaned = True
        
        while cleaned:
            cleaned = False
            for suffix in self.suffixes:
                if word.endswith(suffix):
                    # Ensure we don't stem too short (e.g., 'bola' -> 'bo' is bad)
                    if len(word) - len(suffix) >= 3:
                        word = word[:-len(suffix)]
                        cleaned = True
                        break # Restart loop after removing one suffix
        return word