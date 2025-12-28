# src/uznlp/models/baselines.py
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from typing import Literal

class TextClassifier:
    """
    A wrapper for classical ML models. 
    Designed to be compatible with our future Deep Learning classes.
    """
    def __init__(self, model_type: Literal['nb', 'logreg'] = 'logreg'):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=5000)
        
        if model_type == 'nb':
            self.clf = MultinomialNB()
        else:
            self.clf = LogisticRegression(max_iter=1000, class_weight='balanced')
            
        # Create a scikit-learn pipeline (Vectorization + Classification)
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('clf', self.clf)
        ])

    def train(self, X_train, y_train):
        print(f"Training {self.model_type.upper()} model on {len(X_train)} samples...")
        self.pipeline.fit(X_train, y_train)
        
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def save(self, path):
        joblib.dump(self.pipeline, path)
        print(f"Model saved to {path}")
        
    def load(self, path):
        self.pipeline = joblib.load(path)