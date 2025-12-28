import re

class UzbekStemmer:
    def __init__(self):
        # Order matters! Longer suffixes should be removed first.
        self.suffixes = [
            # Possessive (Egalik)
            "imiz", "ingiz", "lari", "miz", "ngiz", 
            # Plural (Ko'plik)
            "lar", 
            # Case (Kelishik)
            "ning", "ni", "ga", "da", "dan", 
            # Derivational (Yasovchi - optional, usually risky to remove)
            "lik", "siz", "gi"
        ]

    def stem(self, word):
        """
        Iteratively removes suffixes from the end of the word.
        Simple rule-based approach for agglutinative Uzbek.
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