"""
capstone/modules/m03_pretrained_embedder.py
PretrainedEmbedder — oldindan o'qitilgan so'z vektorlarini boshqaradi.
Shartnoma: capstone/contracts.py :: PretrainedEmbedder
P3 (4-kun amaliyoti) da qurilgan; m04 (LSH), m08 (RNN init) tomonidan ishlatiladi.

Kaggle: gensim KeyedVectors .kv (cc_uz_100k.kv) yuklanadi.
Offline: word2vec matn formati (.vec) gensimsiz, toza numpy bilan o'qiladi.
"""
from __future__ import annotations

import numpy as np


class PretrainedEmbedder:
    """Oldindan o'qitilgan Word2Vec/.kv embeddinglarini boshqaradi.

    Consumed by: m04 (LSH), m08 (GRU/LSTM pretrained init).
    """

    def __init__(self) -> None:
        self._words: list[str] = []
        self._w2i: dict[str, int] = {}
        self._raw: np.ndarray | None = None   # (n, dim) xom vektorlar
        self._norm: np.ndarray | None = None  # (n, dim) L2-normallashtirilgan (kosinus uchun)
        self._dim: int = 0

    # ─── yuklash ──────────────────────────────────────────────────────────────
    def load(self, path: str) -> None:
        """Gensim KeyedVectors (.kv) yoki word2vec matn (.vec) faylni yuklaydi."""
        if path.endswith(".kv"):
            from gensim.models import KeyedVectors   # faqat Kaggle/onlayn
            kv = KeyedVectors.load(path)
            words = list(kv.index_to_key)
            mat = np.asarray(kv.vectors, dtype=np.float32)
        else:
            words, mat = self._load_text(path)
        self._words = words
        self._w2i = {w: i for i, w in enumerate(words)}
        self._raw = mat.astype(np.float32)
        self._dim = mat.shape[1]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._norm = (mat / norms).astype(np.float32)

    @staticmethod
    def _load_text(path: str) -> tuple[list[str], np.ndarray]:
        words, rows = [], []
        with open(path, encoding="utf-8") as f:
            first = f.readline().split()
            # sarlavha "n dim" bo'lsa o'tkazib yuboramiz, aks holda ma'lumot
            if not (len(first) == 2 and first[0].isdigit() and first[1].isdigit()):
                words.append(first[0])
                rows.append(np.asarray(first[1:], dtype=np.float32))
            for line in f:
                parts = line.rstrip("\n").split(" ")
                if len(parts) < 2:
                    continue
                words.append(parts[0])
                rows.append(np.asarray(parts[1:], dtype=np.float32))
        return words, np.vstack(rows)

    # ─── asosiy metodlar ────────────────────────────────────────────────────────
    def embed(self, word: str) -> np.ndarray:
        """So'z uchun xom vektor; OOV uchun sifr-vektori (shape (dim,) float32)."""
        i = self._w2i.get(word)
        if i is None:
            return np.zeros(self._dim, dtype=np.float32)
        return self._raw[i].copy()

    def most_similar(self, word: str, n: int = 5) -> list[tuple[str, float]]:
        """Kosinus bo'yicha eng o'xshash n ta so'z: [(so'z, o'xshashlik), ...]."""
        i = self._w2i.get(word)
        if i is None:
            return []
        sims = self._norm @ self._norm[i]
        order = np.argsort(-sims)
        out: list[tuple[str, float]] = []
        for j in order:
            if int(j) == i:
                continue
            out.append((self._words[int(j)], float(sims[int(j)])))
            if len(out) >= n:
                break
        return out

    def oov_rate(self, texts: list[list[str]]) -> float:
        """Tokenlar orasida lug'atda yo'q so'zlar ulushi [0,1]."""
        total = oov = 0
        for toks in texts:
            for t in toks:
                total += 1
                if t not in self._w2i:
                    oov += 1
        return oov / total if total else 0.0
