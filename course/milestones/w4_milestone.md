# 4-hafta milestone (M4): Deploy + bilim testi + agent scaffold

**Sana:** 8-iyul 2026 (chorshanba) · **Topshirish muddati:** 9-iyul 2026 (payshanba, 09:00 gacha — Day 15 dan oldin)
**Format:** asinxron, mustaqil (konsultatsiya bilan) · **Qamrov:** butun kurs (m01–m13 + bilim L1–L14)

Bu — kursning **yakuniy milestone**i. U **uch vazifani** birlashtiradi va kapstone "O'zbek hujjat
yordamchisi"ni ishlab chiqarishga tayyorlaydi.

> ⚠️ **Teskari sinf (flipped):** Task A (P16 — FastAPI/Docker) L16 (MLOps ma'ruzasi) dan **oldin**
> bajariladi. Avval amalda quramiz, keyin L16 da nazariyani rasmiylashtiramiz.

```
   Task A: model -> FastAPI (POST /predict) -> Docker        (amaliy deploy)
   Task B: L1-L14 bilim testi (savol + javoblar kaliti)      (nazariy mustahkamlash)
   Task C: m15 agent scaffold (m13/m14/m12/m04 tool'lari)    (P15 ga tayyorgarlik)
```

---

## Task A — NLP modelini API sifatida joylashtirish (P16)

O'qitilgan sentiment modelni **veb-xizmat**ga aylantiramiz.

1. **Modelni saqlash/yuklash.** `m13 FineTunedClassifier` (yoki `m02`) ni saqlab, xizmatda yuklaymiz.
2. **FastAPI server.** `capstone/app.py` da `POST /predict` endpoint: `{"text": "..."}` qabul qilib,
   `{"sentiment": "ijobiy"|"salbiy", "confidence": float}` qaytaradi.
3. **JSON shartnoma.** Javob har doim bir xil tuzilishda (L15 [I3] API misoli; yorliqlar `ijobiy`/`salbiy`).
4. **Dockerfile.** Ilovani konteynerga o'rab, lokal ishga tushirish (Docker ixtiyoriy — `notebook`da kod beriladi).

Notebook: `course/practices/d16_p16_fastapi.ipynb` (**to'liq ishlangan** — so'nuvchi tayanch yo'q; konsultatsiya seansi).
Mahalliy tekshiruv: `from fastapi.testclient import TestClient`.

## Task B — Bilim testi (L1–L14)

Day 15 dan oldin yakka tartibda yechiladigan **30 savollik** test:
- **Savol varaqasi:** `course/final_test.docx` (MCQ + qisqa javob aralash).
- **Javoblar kaliti:** `course/final_test.xlsx` (har savol uchun to'g'ri javob + tegishli ma'ruza).
- **Qamrov:** L1–L14 (BoW/TF-IDF, Naive Bayes, embedding/kosinus, edit distance, n-gram/Viterbi, Word2Vec,
  RNN, LSTM/GRU, generatsiya, NER, seq2seq/attention/BLEU, Transformer/ROUGE, transfer/BERT, RAG).
  Agent (L15) va MLOps (L16) bu testga **kirmaydi**.

## Task C — Agent scaffold (P15 ga tayyorgarlik)

`capstone/modules/m15_langchain_agent.py` — m13/m14/m12/m04/m10 ni `Tool` sifatida ulovchi agent.
Maqsad: Day 16 P15 seansi **noldan emas**, ulash va sayqallash bo'lsin.

> ✅ Bu loyihada **m15 P15 da to'liq qurilgan** (scaffold emas, ishlaydigan modul). Task C bajarilgan;
> w4_check.py uni import qilib `run()` ni sinaydi.

---

## Topshirish

```
git add capstone/app.py course/final_test.docx course/final_test.xlsx
git commit -m "M4: SentimentAPI deploy + bilim testi + agent"
git push
```

## O'z-o'zini tekshirish

```
python course/milestones/w4_check.py
```
Skript uch vazifani tekshiradi: (A) `TestClient` orqali `POST /predict` → `{sentiment, confidence}`;
(B) `final_test.docx`/`.xlsx` mavjud va to'liq; (C) `m15` import + `run()`.

## Kapstone himoyasi (Day 16)

Himoyada jonli ko'rsating: **m15 agent** (o'zbekcha so'rov) + **SentimentAPI** (`POST /predict`) +
Docker bilan ishga tushirish. 16 kunlik mehnatingiz — bitta ishlaydigan tizim. 🎓
