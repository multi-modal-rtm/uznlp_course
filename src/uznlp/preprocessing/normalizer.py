import re
from bs4 import BeautifulSoup

class UzbekNormalizer:
    """
    Production-grade text normalizer for Uzbek language.
    Handles Unicode normalization, Apostrophe unification, and Cyrillic-Latin conversion.
    """

    def __init__(self, target_script: str = 'latin'):
        self.target_script = target_script
        self.cyr2lat_map = {
            # Lowercase
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
            'ж': 'j', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
            'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
            'ф': 'f', 'х': 'x', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sh', 'ъ': "'",
            'ы': 'i', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya', 'ў': "o'", 'қ': 'q',
            'ғ': "g'", 'ҳ': 'h',
            # Uppercase
            'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'Yo',
            'Ж': 'J', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
            'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
            'Ф': 'F', 'Х': 'X', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Sh', 'Ъ': "'",
            'Ы': 'I', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya', 'Ў': "O'", 'Қ': 'Q',
            'Ғ': "G'", 'Ҳ': 'H'
        }

    def normalize_apostrophes(self, text: str) -> str:
        """
        Unifies various apostrophe characters to a single standard ASCII apostrophe.
        Critical for words like O'zbekiston, g'oz, a'lo.
        """
        return re.sub(r"[‘`’]", "'", text)

    def to_latin(self, text: str) -> str:
        """Converts Cyrillic input to Latin based on standard Uzbek transliteration."""
        result = []
        for char in text:
            lower_char = char.lower()
            if lower_char in self.cyr2lat_map:
                res = self.cyr2lat_map[lower_char]
                if char.isupper():
                    res = res.capitalize()
                result.append(res)
            else:
                result.append(char)
        return "".join(result)

    def clean_text_robust(self, text: str):
        if not isinstance(text, str):
            return ""
        
        # 1. Remove HTML Tags (Fixes 'div', 'class' artifacts)
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
        
        # 2. Normalize Apostrophes (CRITICAL FIX)
        # Converts curly quotes (‘ ’), okina (ʻ), and backticks (`) to standard (')
        text = re.sub(r"[\u2018\u2019\u02BB\u02BC\u0060]", "'", text)

        # 3. Lowercase
        text = text.lower()
        
        # 4. Remove URLs and Emails
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # 5. Filter Characters
        # Keep only: a-z, 0-9, space, and the standard apostrophe (')
        # This removes commas (5,9 -> 5 9) and other punctuation
        text = re.sub(r"[^a-z0-9'\s]", ' ', text)
        
        # 6. Collapse multiple spaces into one
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text