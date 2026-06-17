"""
capstone/app.py
SentimentAPI — FastAPI orqali o'zbek sentiment modelini xizmat qiladi.
Shartnoma: capstone/contracts.py :: create_sentiment_api
M4 (P16, 8-iyul) da qurilgan. Kapstone yakuniy mahsuloti (deploy artefakti).

Endpoint:
    POST /predict   request:  {"text": "matn"}
                    response: {"sentiment": "ijobiy"|"salbiy", "confidence": float}
    GET  /          xizmat haqida qisqa ma'lumot.

Model: m13 FineTunedClassifier. Mahalliy/offline: USE_TRANSFORMERS=False
(TF-IDF + LogisticRegression) — internetsiz, startda kichik korpusda fit qilinadi.
Kaggle/ishlab chiqarish: oldindan o'qitilgan modelni load() bilan yuklash mumkin.
"""
from __future__ import annotations

import os
import sys

from fastapi import FastAPI
from pydantic import BaseModel

# kapstone modullarini import yo'liga qo'shamiz
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modules"))

import m13_bert_classifier as m13   # noqa: E402

m13.USE_TRANSFORMERS = False        # offline: TF-IDF + LogReg (yuklab olishsiz)

# Startda o'qitiladigan kichik sentiment korpus (yoki saqlangan modelni yuklang)
_TRAIN_TEXTS = [
    "mahsulot juda sifatli va arzon", "yetkazib berish tez bo'ldi",
    "ajoyib xizmat, hammaga tavsiya qilaman", "buyurtma vaqtida keldi",
    "telefon tez ishlaydi, juda yoqdi", "qadoqlash zo'r edi rahmat",
    "mahsulot buzuq keldi juda xafa bo'ldim", "yetkazib berish kechikdi yomon",
    "sifati past pulga arzimaydi", "xizmat sust javob bermadi",
    "mahsulot tavsifga mos kelmadi", "umuman tavsiya qilmayman",
]
_TRAIN_LABELS = ["ijobiy"] * 6 + ["salbiy"] * 6

_clf = m13.FineTunedClassifier()
_clf.fit(_TRAIN_TEXTS, _TRAIN_LABELS)


class PredictIn(BaseModel):
    text: str


def create_sentiment_api() -> FastAPI:
    """FastAPI ilovasini yaratadi va sozlaydi (import/test uchun ham)."""
    app = FastAPI(title="O'zbek sentiment API", version="1.0")

    @app.get("/")
    def root():
        return {"service": "sentiment", "endpoint": "POST /predict",
                "labels": ["ijobiy", "salbiy"]}

    @app.post("/predict")
    def predict(body: PredictIn):
        text = body.text
        sentiment = _clf.predict(text)
        proba = _clf.predict_proba(text)
        confidence = max(proba.values()) if proba else 0.0
        return {"sentiment": sentiment, "confidence": round(float(confidence), 4)}

    return app


# uvicorn capstone.app:app uchun modul darajasidagi ilova
app = create_sentiment_api()
