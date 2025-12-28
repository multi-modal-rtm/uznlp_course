import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.uznlp.preprocessing.pipeline import TextPreprocessor

from src.uznlp.models.baselines import TextClassifier
from src.uznlp.models.evaluation import ModelEvaluator

def main():
    df = pd.read_csv('data/processed/uzbek_sentiment.csv')

    print("Preprocessing data...")
    processor = TextPreprocessor(stopwords=['va', 'bilan', 'uchun']) 

    df['clean_text'] = df['text'].apply(processor.process)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42
    )

    clf = TextClassifier(model_type='logreg')
    clf.train(X_train, y_train)

    y_pred = clf.predict(X_test)
    
    evaluator = ModelEvaluator(y_test, y_pred, labels=['Negative', 'Positive'])
    metrics = evaluator.get_metrics()
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Detailed Report saved to logs.")

    evaluator.plot_confusion_matrix(title="Logistic Regression Results")

    sample = "Bu mahsulot sifati juda yomon, umuman yoqmadi."
    clean_sample = processor.process(sample)
    prediction = clf.predict([clean_sample])[0]
    print(f"Test Sentence: '{sample}' -> Prediction: {prediction}")

if __name__ == "__main__":
    main()