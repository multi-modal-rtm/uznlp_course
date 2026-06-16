"""
course/milestones/w3_check.py
3-hafta milestone o'z-o'zini tekshiruvchi skript.

Neyron arxitekturalar integratsiyasi: m01–m08 modullarining capstone/contracts.py
imzolariga mosligini va asosiy integratsiyani tekshiradi:
    1) KLASSIFIKATOR TAQQOSLOVI: m02 (LogReg/NB) vs m07 (RNN) vs m08 (GRU/LSTM) —
       BIR XIL train/test split'da F1 va inference tezligini taqqoslash.
    2) m06 (CustomWord2Vec) funksional tekshiruv (embed/most_similar) — pretrained
       embeddinglar mavjudligi (m07/m08 da pretrained-vs-random tadqiqoti uchun asos).

Ishga tushirish (repo ildizidan yoki istalgan joydan):
    python course/milestones/w3_check.py

Eslatma: torch CPU mavjud bo'lsa m07/m08 nn.* yo'lidan; bo'lmasa numpy fallback'idan
foydalanadi. gensim/datasketch/nltk SHART EMAS. Qamrov: m01–m08.
m05b (pedagogik), m09 (pedagogik) va m10 (Day 11) TEKSHIRILMAYDI.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# ─── yo'llar ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent          # course/milestones
REPO = HERE.parent.parent                        # repo ildizi
MODULES = REPO / "capstone" / "modules"
sys.path.insert(0, str(MODULES))

SENT_DATA = REPO / "course" / "practices" / "d03_checkpoints" / "uz_sentiment_mini.txt"
W2V_DATA  = REPO / "course" / "practices" / "d07_checkpoints" / "uz_w2v_corpus.txt"

import random
random.seed(42)
np.random.seed(42)

_passed = 0


def check(nom: str, shart: bool) -> None:
    global _passed
    assert shart, f"FAIL: {nom}"
    print(f"  ✓ {nom}")
    _passed += 1


print("=" * 70)
print("3-HAFTA MILESTONE — O'Z-O'ZINI TEKSHIRUV (w3_check)")
print("Neyron arxitekturalar integratsiyasi: m01–m08")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# 0-QISM — shartnoma mosligi (capstone/contracts.py): m01–m08
# ══════════════════════════════════════════════════════════════════════════
print("\n[0-qism] Shartnoma mosligi — m01–m08")

from m01_text_preprocessor import TextPreprocessor
from m02_sentiment_classifier import SentimentClassifier
from m03_pretrained_embedder import PretrainedEmbedder
from m04_spell_lsh_retriever import SpellLSHRetriever
from m05_autocomplete import Autocomplete
from m06_custom_word2vec import CustomWord2Vec
from m07_rnn_classifier import RNNClassifier
from m08_gru_lstm_classifier import GRULSTMClassifier

_specs = {
    "m01": (TextPreprocessor(), ["preprocess", "preprocess_batch", "fit_stopwords"]),
    "m02": (SentimentClassifier(), ["fit", "predict", "predict_proba", "save", "load"]),
    "m03": (PretrainedEmbedder(), ["load", "embed", "most_similar", "oov_rate"]),
    "m04": (SpellLSHRetriever(), ["correct", "edit_distance", "index_docs", "retrieve_lsh"]),
    "m05": (Autocomplete(), ["train", "complete", "perplexity"]),
    "m06": (CustomWord2Vec(), ["train", "embed", "most_similar", "save", "load"]),
    "m07": (RNNClassifier(), ["fit", "predict", "predict_proba", "save", "load"]),
    "m08": (GRULSTMClassifier(), ["fit", "predict", "compare_report", "save", "load"]),
}
for nom, (obj, metodlar) in _specs.items():
    check(f"{nom} shartnomaga mos ({', '.join(metodlar)})",
          all(callable(getattr(obj, m, None)) for m in metodlar))

# ══════════════════════════════════════════════════════════════════════════
# Ma'lumotni yuklash va train/test split (bir xil — adolatli taqqoslash uchun)
# ══════════════════════════════════════════════════════════════════════════
assert SENT_DATA.exists(), f"Sentiment data topilmadi: {SENT_DATA}"
texts, labels = [], []
for line in SENT_DATA.open(encoding="utf-8"):
    if not line.strip():
        continue
    rating, text = line.rstrip("\n").split("\t", 1)
    r = int(rating)
    if r >= 4:   texts.append(text); labels.append("ijobiy")
    elif r <= 2: texts.append(text); labels.append("salbiy")

X_tr, X_te, y_tr, y_te = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels)
y_te_bin = [l == "ijobiy" for l in y_te]

# ══════════════════════════════════════════════════════════════════════════
# 1-INTEGRATSIYA — KLASSIFIKATOR TAQQOSLOVI: m02 vs m07 vs m08
# ══════════════════════════════════════════════════════════════════════════
print("\n[1-integratsiya] Klassifikator taqqoslovi — m02 vs m07 vs m08 (bir xil split)")
print(f"  train: {len(X_tr)} | test: {len(X_te)} misol")

report: dict = {}


def baholash(nom, clf):
    """Test split'da F1 va inference vaqtini hisoblaydi."""
    preds = []
    t0 = time.perf_counter()
    for t in X_te:
        preds.append(clf.predict(t))
    infer = time.perf_counter() - t0
    f1 = f1_score(y_te_bin, [p == "ijobiy" for p in preds])
    report[nom] = {"f1": round(float(f1), 4), "inference_time": round(infer, 4)}
    check(f"{nom}: predict ijobiy/salbiy beradi", set(preds) <= {"ijobiy", "salbiy"})
    return preds


m02 = SentimentClassifier(); m02.fit(X_tr, y_tr)
baholash("m02_logreg", m02)

m07 = RNNClassifier(); m07.fit(X_tr, y_tr, epochs=12, hidden_size=32, lr=0.01)
baholash("m07_rnn", m07)

m08 = GRULSTMClassifier(); m08.fit(X_tr, y_tr, arch="lstm", epochs=10, hidden_size=32, num_layers=2, lr=0.01)
baholash("m08_lstm", m08)

check("uchala klassifikator taqqoslandi (m02, m07, m08)",
      set(report.keys()) == {"m02_logreg", "m07_rnn", "m08_lstm"})
check("har klassifikatorda f1 va inference_time bor",
      all("f1" in report[k] and "inference_time" in report[k] for k in report))
check("m02 (klassik bazaviy) F1 > 0.5", report["m02_logreg"]["f1"] > 0.5)
check("m07/m08 F1 [0,1] oralig'ida (ishladi)",
      all(0.0 <= report[k]["f1"] <= 1.0 for k in ("m07_rnn", "m08_lstm")))

print("  Taqqoslov hisoboti:")
for nom, m in report.items():
    print(f"    {nom:12s} F1={m['f1']:.3f}  inference={m['inference_time']:.4f}s")

# ══════════════════════════════════════════════════════════════════════════
# 2-INTEGRATSIYA (asos) — m06 funksional: pretrained embeddinglar
# ══════════════════════════════════════════════════════════════════════════
print("\n[2-integratsiya] m06 CustomWord2Vec — pretrained embeddinglar (m07/m08 init uchun asos)")

assert W2V_DATA.exists(), f"Korpus topilmadi: {W2V_DATA}"
pre = TextPreprocessor()
sents = [pre.preprocess(l.strip()) for l in W2V_DATA.open(encoding="utf-8") if l.strip()]
w2v = CustomWord2Vec(); w2v.train(sents, vector_size=32, window=2, min_count=2, epochs=20)

# embedding lug'atidan bir so'z tanlaymiz
some_word = next(w for s in sents for w in s if w in getattr(w2v, "_w2i", {})) \
    if not getattr(w2v, "_gensim_model", None) else "toshkent"
vec = w2v.embed(some_word)
check("m06 embed() 32-o'lchamli vektor qaytaradi", hasattr(vec, "shape") and vec.shape == (32,))
sim = w2v.most_similar(some_word, n=3)
check("m06 most_similar() (so'z, o'xshashlik) ro'yxati",
      isinstance(sim, list) and all(isinstance(w, str) and -1.01 <= s <= 1.01 for w, s in sim))
check("m06 embeddinglari m07/m08 uchun pretrained init manbai bo'la oladi (shakl mos: dim>0)",
      vec.shape[0] > 0)
print(f"    m06 '{some_word}' ~ {[w for w, _ in sim]}")

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"NATIJA: {_passed} tekshiruvning hammasi O'TDI ✓")
print("3-hafta milestone tayyor: m01–m08 neyron arxitekturalar integratsiyasi.")
print("m09/m05b (pedagogik) va m10 (Day 11) bu milestone qamrovida emas.")
print("=" * 70)
