"""
capstone/modules/m14_rag_engine.py
RAGEngine — FAISS + sentence-transformers + LLM API asosida RAG qidirish tizimi.
Shartnoma: capstone/contracts.py :: RAGEngine
P14 (15-kun amaliyoti) da qurilgan. HAQIQIY pipeline moduli (m10/m12/m13 kabi):
save_index/load_index BOR, consumed_by m15 (agent tool: retrieve_docs).

Korpus: uz_kb (yangiliklar + lex.uz, 10000 chunk) yoki offline mini bilim bazasi.

sentence-transformers / faiss / LLM API SHART EMAS (m13 ixtiyoriy-kutubxona namunasi):
  - Kaggle yo'li: sentence-transformers embedding + FAISS IndexFlatIP + LLM API.
  - Offline yo'l: TF-IDF (sklearn) embedding + numpy kosinus top-k + EKSTRAKTIV javob
    (top-k sources birlashtirib; haqiqiy LLM yo'q).
USE_ST / HAS_FAISS / USE_LLM bayroqlari yo'lni tanlaydi (mahalliy tekshirishda False ga majburlanadi).
"""
from __future__ import annotations

import pickle
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import sentence_transformers  # noqa: F401
    HAS_ST = True
except Exception:                  # ImportError yoki torch DLL xatosi -> fallback
    HAS_ST = False

try:
    import faiss  # noqa: F401
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# Yo'l tanlovi. Mahalliy tekshirishda notebook/builder bularni False ga majburlaydi
# (model yuklab olish / faiss / LLM API mahalliy yo'q yoki internet talab qiladi).
USE_ST = HAS_ST       # embedding uchun sentence-transformers ishlatilsinmi
USE_LLM = False       # haqiqiy LLM API ulanishi (kalit + internet); standart: ekstraktiv

_ST_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _l2_normalize(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms


class RAGEngine:
    """FAISS/sentence-transformers (yoki TF-IDF/numpy) asosidagi RAG tizimi.

    Consumed by: m15 (agent tool: retrieve_docs).
    """

    def __init__(self) -> None:
        self._docs: list[str] = []
        self._emb: np.ndarray | None = None      # (n, d) normallangan embeddinglar
        self._mode: str | None = None            # "st" | "tfidf"
        self._st = None                          # sentence-transformers model
        self._vec: TfidfVectorizer | None = None # TF-IDF fallback
        self._faiss = None                       # FAISS indeks (Kaggle)

    # ─── embedding ──────────────────────────────────────────────────────────
    def _embed(self, texts: list[str], fit: bool) -> np.ndarray:
        if USE_ST and HAS_ST:
            if self._st is None:
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer(_ST_MODEL)
            self._mode = "st"
            M = np.asarray(self._st.encode(list(texts)), dtype=np.float32)
        else:
            self._mode = "tfidf"
            if fit:
                # char n-gram: o'zbek qo'shimchalarini hisobga oladi
                # ("poytaxt" va "poytaxti" umumiy n-gramlar bilan yaqin bo'ladi)
                self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
                M = self._vec.fit_transform(texts).toarray().astype(np.float32)
            else:
                M = self._vec.transform(texts).toarray().astype(np.float32)
        return _l2_normalize(M)

    # ─── indekslash ─────────────────────────────────────────────────────────
    def index(self, texts: list[str], batch_size: int = 32) -> None:
        """Hujjatlarni embedding qilib indeksga qo'shadi."""
        self._docs = list(texts)
        self._emb = self._embed(self._docs, fit=True)
        if HAS_FAISS and self._mode == "st":
            import faiss
            self._faiss = faiss.IndexFlatIP(self._emb.shape[1])
            self._faiss.add(self._emb)

    # ─── qidiruv ────────────────────────────────────────────────────────────
    def _retrieve(self, question: str, k: int):
        q = self._embed([question], fit=False)[0]
        if self._faiss is not None:
            sims, idx = self._faiss.search(q[None, :], k)
            return list(idx[0]), list(sims[0])
        sims = self._emb @ q                      # kosinus (normallangan)
        order = np.argsort(sims)[::-1][:k]
        return list(order), [float(sims[i]) for i in order]

    def _format_prompt(self, question: str, sources: list[str]) -> str:
        context = "\n".join(sources)
        return f"Kontekst:\n{context}\n\nSavol: {question}\nJavob:"

    def answer(self, question: str, k: int = 3) -> dict[str, Any]:
        """Savolga RAG pipeline orqali javob qaytaradi.

        Returns: {"answer": str, "sources": list[str], "confidence": float}.
        """
        if self._emb is None:
            raise ValueError("Avval index() ni chaqiring.")
        k = min(k, len(self._docs))
        idx, sims = self._retrieve(question, k)
        sources = [self._docs[i] for i in idx]
        prompt = self._format_prompt(question, sources)
        if USE_LLM:
            ans = self._llm_call(prompt)          # Kaggle: haqiqiy LLM API
        else:
            ans = " ".join(sources)               # offline: ekstraktiv javob
        conf = float(np.clip(sims[0], 0.0, 1.0)) if sims else 0.0
        return {"answer": ans, "sources": sources, "confidence": conf}

    def _llm_call(self, prompt: str) -> str:
        # Kaggle yo'li: bu yerda OpenAI/HF Inference API chaqiriladi.
        # from openai import OpenAI; client.chat.completions.create(...)
        raise NotImplementedError("LLM API faqat Kaggle (USE_LLM=True) yo'lida.")

    # ─── saqlash / yuklash ──────────────────────────────────────────────────
    def save_index(self, path: str) -> None:
        """Indeks va hujjatlarni saqlaydi (offline: pickle)."""
        state = {"docs": self._docs, "emb": self._emb, "mode": self._mode}
        if self._mode == "tfidf":
            state["vec"] = self._vec
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_index(self, path: str) -> None:
        """Saqlangan indeksni yuklaydi."""
        with open(path, "rb") as f:
            s = pickle.load(f)
        self._docs = s["docs"]; self._emb = s["emb"]; self._mode = s["mode"]
        self._vec = s.get("vec")
        self._faiss = None
        if HAS_FAISS and self._mode == "st" and self._emb is not None:
            import faiss
            self._faiss = faiss.IndexFlatIP(self._emb.shape[1])
            self._faiss.add(self._emb)
