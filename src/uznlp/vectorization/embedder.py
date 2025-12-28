import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Literal, Tuple

class Vectorizer:
    def __init__(self, method: Literal['bow', 'tfidf'] = 'tfidf', max_features: int = 5000):
        self.method = method
        if method == 'bow':
            self.model = CountVectorizer(max_features=max_features)
        else:
            self.model = TfidfVectorizer(max_features=max_features)
            
    def fit_transform(self, corpus: list) -> Tuple[pd.DataFrame, list]:
        """
        Returns a DataFrame for better visualization in Labs, 
        and the feature names.
        """
        matrix = self.model.fit_transform(corpus)
        feature_names = self.model.get_feature_names_out()
        
        # Return sparse matrix wrapped in DataFrame for inspection (Careful with RAM on huge data)
        df = pd.DataFrame.sparse.from_spmatrix(matrix, columns=feature_names)
        return df, feature_names