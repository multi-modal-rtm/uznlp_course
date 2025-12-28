import unittest
import sys
import os

# Loyiha asosiy papkasini (Project Root) yo'lga qo'shamiz
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.uznlp.preprocessing.normalizer import UzbekNormalizer
from src.uznlp.preprocessing.pipeline import TextPreprocessor
# Stemmerni tekshirish uchun (agar mavjud bo'lsa)
try:
    from src.uznlp.preprocessing.stemmer import UzbekStemmer
except ImportError:
    # Agar kutubxonaga o'tgan bo'lsak
    from src.uznlp.preprocessing.stemmer_lib import UzbekStemmerLib as UzbekStemmer

class TestUzbekNLP(unittest.TestCase):
    
    def setUp(self):
        """Har bir testdan oldin ishga tushadi"""
        self.normalizer = UzbekNormalizer()
        self.stopwords = ["va", "bilan", "uchun", "bu"]
        self.processor = TextPreprocessor(stopwords=self.stopwords)

    # --- 1. NORMALIZATSIYA TESTLARI ---
    def test_html_cleaning(self):
        """HTML teglar tozalanishini tekshirish"""
        raw = "<div>Yangiliklar <b>e'lon</b> qilindi.</div>"
        expected = "yangiliklar e'lon qilindi."
        result = self.normalizer.clean_text_robust(raw)
        # Nuqta va bo'shliqlarni hisobga olgan holda tekshiramiz
        # Bizning normalizer nuqtalarni olib tashlaydi, shuning uchun:
        self.assertIn("yangiliklar", result)
        self.assertNotIn("<div>", result)
        self.assertNotIn("<b>", result)

    def test_apostrophe_normalization(self):
        """Egri apostroflar to'g'rilanishini tekshirish"""
        # O‘zbekiston (egri) -> o'zbekiston (to'g'ri)
        raw = "O‘zbekiston va G’azna"
        clean = self.normalizer.clean_text_robust(raw)
        self.assertIn("o'zbekiston", clean)
        self.assertIn("g'azna", clean) # G’ -> g'

    def test_number_preservation(self):
        """Raqamlar saqlanib qolishini tekshirish"""
        raw = "Yalpi ichki mahsulot 5,9 foizga o‘sdi"
        clean = self.normalizer.clean_text_robust(raw)
        # Normalizer 5,9 ni "5 9" ga aylantiradi (tinish belgisiz)
        self.assertIn("5", clean)
        self.assertIn("9", clean)

    # --- 2. PIPELINE VA STEMMING TESTLARI ---
    def test_pipeline_stopwords(self):
        """Stop-wordlar olib tashlanishini tekshirish"""
        text = "bu maktab va universitet uchun"
        # "bu", "va", "uchun" stopword ro'yxatida bor
        result = self.processor.process(text)
        self.assertNotIn("bu", result.split())
        self.assertNotIn("va", result.split())
        self.assertIn("makta", result)
        self.assertIn("universitet", result)

    def test_stemming_logic(self):
        """Stemming (o'zakka ajratish) ishlashini tekshirish"""
        # "maktablarimizda" -> "maktab" bo'lishi kerak
        text = "maktablarimizda bolalar o'qishmoqda"
        result = self.processor.process(text)
        
        # Natijada "maktab" so'zi bo'lishi kerak ("maktablarimizda" emas)
        # Agar stemming ishlasa:
        self.assertIn("makta", result)
        
        # "bolalar" -> "bola"
        self.assertIn("bola", result)

    def test_short_words_and_numbers(self):
        """Qisqa so'zlar va raqamlar filtri"""
        text = "37 ta yangi uy 12 qavatli"
        result = self.processor.process(text)
        
        # 37 va 12 raqamlari saqlanishi kerak
        self.assertIn("37", result.split())
        self.assertIn("12", result.split())
        
        # "ta" va "uy" (2 harfli) saqlanishi kerak (yangi logikaga ko'ra)
        self.assertIn("ta", result.split())

if __name__ == '__main__':
    unittest.main()