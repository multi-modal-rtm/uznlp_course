"""
capstone/modules/m13_bert_classifier.py
FineTunedClassifier — Hugging Face Trainer orqali nozik sozlangan BERT-class
sentiment modeli (ijobiy / salbiy).
Shartnoma: capstone/contracts.py :: FineTunedClassifier
P13 (14-kun amaliyoti) da qurilgan. HAQIQIY pipeline moduli (m10/m12 kabi):
save/load BOR, consumed_by m15 (agent: sentiment_classify), app.py (FastAPI, M4).

Korpus: risqaliyevds/uzbek-sentiment-analysis (MIT) yoki offline mini korpus.
Yorliqlar QULFLANGAN: "ijobiy" / "salbiy" (L2 [I2] bilan bog'langan).

transformers SHART EMAS (m12 ixtiyoriy-kutubxona namunasi):
  - Kaggle yo'li: AutoModelForSequenceClassification (DistilBERT) + Trainer fine-tune.
  - Offline yo'l (transformers'siz YOKI USE_TRANSFORMERS=False): TF-IDF + sklearn
    LogisticRegression (m02 isbotlangan naqshi; tez, yuklab olishsiz).
USE_TRANSFORMERS bayrog'i yo'lni tanlaydi (mahalliy tekshirish uchun False ga majburlanadi).
"""
from __future__ import annotations

import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

try:
    import transformers  # noqa: F401
    HAS_TRANSFORMERS = True
except Exception:        # ImportError yoki torch DLL xatosi -> fallback (LogReg)
    HAS_TRANSFORMERS = False

# Yo'l tanlovi: standart -- transformers bo'lsa o'sha. Mahalliy tekshirishda
# notebook/builder buni False ga majburlaydi (yuklab olish + sekin CPU'dan qochish).
USE_TRANSFORMERS = HAS_TRANSFORMERS

# Yorliq <-> indeks (QULFLANGAN: ijobiy=1, salbiy=0)
_I2LAB = ["salbiy", "ijobiy"]
_LAB2I = {"salbiy": 0, "ijobiy": 1}


class FineTunedClassifier:
    """Fine-tune qilingan BERT (yoki TF-IDF+LogReg fallback) sentiment klassifikatori.

    Consumed by: m15 (agent tool: sentiment_classify), app.py (FastAPI, M4).
    """

    def __init__(self) -> None:
        self._mode: str | None = None       # "transformers" | "classical"
        self._tok = None                     # HF tokenizer
        self._hf = None                      # HF model
        self._vec: TfidfVectorizer | None = None
        self._clf: LogisticRegression | None = None

    # ─── o'qitish ─────────────────────────────────────────────────────────────
    def fit(self, texts: list[str], labels: list[str],
            model_name: str = "distilbert-base-multilingual-cased",
            epochs: int = 3, batch_size: int = 16, lr: float = 2e-5) -> None:
        """Sentiment modelini o'qitadi. labels: 'ijobiy' yoki 'salbiy'."""
        if len(texts) != len(labels):
            raise ValueError("texts va labels uzunligi teng bo'lishi kerak.")
        if USE_TRANSFORMERS and HAS_TRANSFORMERS:
            self._fit_transformers(texts, labels, model_name, epochs, batch_size, lr)
            self._mode = "transformers"
        else:
            self._fit_classical(texts, labels)
            self._mode = "classical"

    def _fit_transformers(self, texts, labels, model_name, epochs, batch_size, lr):
        import torch
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                   TrainingArguments, Trainer)
        torch.manual_seed(42)
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._hf = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        y = [_LAB2I[l] for l in labels]
        enc = self._tok(list(texts), truncation=True, padding=True, max_length=128)

        class _DS(torch.utils.data.Dataset):
            def __len__(self_inner):
                return len(y)

            def __getitem__(self_inner, i):
                item = {k: torch.tensor(v[i]) for k, v in enc.items()}
                item["labels"] = torch.tensor(y[i])
                return item

        args = TrainingArguments(
            output_dir="/tmp/m13_out", learning_rate=lr,
            per_device_train_batch_size=batch_size, num_train_epochs=epochs,
            warmup_steps=100, logging_steps=50, report_to=[])
        Trainer(model=self._hf, args=args, train_dataset=_DS()).train()
        self._hf.eval()

    def _fit_classical(self, texts, labels):
        # m02 naqshi: TF-IDF + LogisticRegression (tez, yuklab olishsiz)
        self._vec = TfidfVectorizer()
        X = self._vec.fit_transform(texts)
        self._clf = LogisticRegression(max_iter=1000)
        self._clf.fit(X, labels)

    # ─── bashorat ─────────────────────────────────────────────────────────────
    def predict(self, text: str) -> str:
        """'ijobiy' yoki 'salbiy' qaytaradi."""
        if self._mode == "transformers":
            return self._predict_transformers(text)
        if self._clf is None:
            raise ValueError("Avval fit() ni chaqiring.")
        return str(self._clf.predict(self._vec.transform([text]))[0])

    def predict_proba(self, text: str) -> dict[str, float]:
        """{'ijobiy': 0.87, 'salbiy': 0.13} formatida ehtimolliklar."""
        if self._mode == "transformers":
            return self._proba_transformers(text)
        if self._clf is None:
            raise ValueError("Avval fit() ni chaqiring.")
        probs = self._clf.predict_proba(self._vec.transform([text]))[0]
        out = {str(c): float(p) for c, p in zip(self._clf.classes_, probs)}
        # ikkala yorliq ham bo'lishini kafolatlaymiz
        out.setdefault("ijobiy", 0.0); out.setdefault("salbiy", 0.0)
        return out

    def _predict_transformers(self, text):
        import torch
        enc = self._tok(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = self._hf(**enc).logits[0]
        return _I2LAB[int(logits.argmax())]

    def _proba_transformers(self, text):
        import torch
        enc = self._tok(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            p = torch.softmax(self._hf(**enc).logits[0], dim=-1)
        return {_I2LAB[i]: float(p[i]) for i in range(len(_I2LAB))}

    # ─── saqlash / yuklash ──────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """transformers: save_pretrained(path/) ; classical: pickle(path)."""
        if self._mode == "transformers":
            os.makedirs(path, exist_ok=True)
            self._hf.save_pretrained(path)
            self._tok.save_pretrained(path)
        else:
            with open(path, "wb") as f:
                pickle.dump({"vec": self._vec, "clf": self._clf, "mode": "classical"}, f)

    def load(self, path: str) -> None:
        """Papka bo'lsa HF model; aks holda pickle (classical)."""
        if os.path.isdir(path):
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self._tok = AutoTokenizer.from_pretrained(path)
            self._hf = AutoModelForSequenceClassification.from_pretrained(path)
            self._hf.eval()
            self._mode = "transformers"
        else:
            with open(path, "rb") as f:
                s = pickle.load(f)
            self._vec = s["vec"]; self._clf = s["clf"]; self._mode = "classical"
