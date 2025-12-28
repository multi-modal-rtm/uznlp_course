from UzMorphAnalyser import UzMorphAnalyser

class UzbekStemmerLib:
    def __init__(self):
        self.analyzer = UzMorphAnalyser()

    def stem(self, word):
        """
        Uses the robust UzMorphAnalyser to get the stem.
        """
        try:
            # The library might return None or raise an error for unknown characters
            result = self.analyzer.stem(word)
            return result if result else word
        except:
            return word

    def lemmatize(self, word):
        """
        Returns the dictionary form (Lemma).
        Example: "maktablarimizda" -> "maktab"
        Example: "boraman" -> "bor" (Stem) vs "bormoq" (Lemma)
        """
        try:
            result = self.analyzer.lemmatize(word)
            return result if result else word
        except:
            return word