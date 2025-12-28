import re
from typing import List
from .normalizer import UzbekNormalizer
from .stemmer import UzbekStemmer
from .stemmer_lib import UzbekStemmerLib

class TextPreprocessor:
    def __init__(self, stopwords: List[str]):
        self.normalizer = UzbekNormalizer()
        
        # Use the library-based stemmer now
        self.stemmer = UzbekStemmerLib() 
        
        self.stopwords = set(stopwords)

    def process(self, text: str, return_tokens=False):
        # 1. Normalize
        clean_text = self.normalizer.clean_text_robust(text)
        
        # 2. Tokenize
        tokens = clean_text.split()
        
        # 3. Stopword Removal & Stemming
        processed_tokens = []
        for t in tokens:
            # Check for numbers or words > 1 char
            is_number = t.replace("'", "").isdigit()
            if t not in self.stopwords and (len(t) > 1 or is_number):
                
                # Apply Library Stemming
                lemmatized_t = self.stemmer.lemmatize(t)
                processed_tokens.append(lemmatized_t)
        
        if return_tokens:
            return processed_tokens
            
        return " ".join(processed_tokens)

# Agar biz yuqoridagi UzbekStemmerLib kabi maxsus o'zbek tili morfologik tahlilchisidan foydalanmaganimizda 
# quyidagicha UzbekStemmer classidan foydalangan bo'lar edik. 

# class TextPreprocessor:
#     def __init__(self, stopwords: List[str]):
#         self.normalizer = UzbekNormalizer()
#         self.stemmer = UzbekStemmer()
#         self.stopwords = set(stopwords)

#     def process(self, text: str, return_tokens=False):
#         # 1. Normalize
#         clean_text = self.normalizer.clean_text_robust(text)
        
#         # 2. Tokenize
#         tokens = clean_text.split()
        
#         # 3. Stopword Removal & Stemming
#         processed_tokens = []
#         for t in tokens:
#             # FIX: Allow words > 1 char OR any number (even 1 digit)
#             is_number = t.replace("'", "").isdigit()
#             if t not in self.stopwords and (len(t) > 1 or is_number):
                
#                 # Apply Stemming
#                 stemmed_t = self.stemmer.stem(t)
#                 processed_tokens.append(stemmed_t)
        
#         if return_tokens:
#             return processed_tokens
            
#         return " ".join(processed_tokens)