from UzMorphAnalyser import UzMorphAnalyser

class UzbekStemmerLib:
    def __init__(self):
        self.analyzer = UzMorphAnalyser()

    def stem(self, word):
        """
        "stem"ni olish uchun UzMorphAnalyserdan foydalanadi. 
        """
        try:
            result = self.analyzer.stem(word)
            return result if result else word
        except:
            return word

    def lemmatize(self, word):
        """
        Lug'at shaklini qaytaradi (Lemma).
        Misol: "maktablarimizda" -> "maktab"
        Misol: "boraman" -> "bor" (Stem) va "bormoq" (Lemma)
        """
        try:
            result = self.analyzer.lemmatize(word)
            return result if result else word
        except:
            return word