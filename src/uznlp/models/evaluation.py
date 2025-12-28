# src/uznlp/models/evaluation.py
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ModelEvaluator:
    def __init__(self, y_true, y_pred, labels=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels

    def get_metrics(self):
        """Returns a dictionary of core metrics."""
        return {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'report': classification_report(self.y_true, self.y_pred, target_names=self.labels, output_dict=True)
        }

    def plot_confusion_matrix(self, title="Confusion Matrix"):
        """
        Visualizes the confusion matrix.
        Essential for checking if 'Negative' is confused with 'Positive'.
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.labels, yticklabels=self.labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()