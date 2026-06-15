"""
capstone/modules/m05_autocomplete.py
Autocomplete — N-gram til modeli asosida so'z/ibora to'ldirish.
Shartnoma: capstone/contracts.py :: Autocomplete
P5 (6-kun amaliyoti) da qurilgan; m01 (TextPreprocessor) normalizatsiyasidan foydalanadi.
Consumed by: Day 16 (pipeline demo).

nltk SHART EMAS — n-gram toza-python bilan sanaladi. Til modeli uchun stop-so'zlar
SAQLANADI (autocomplete funksional so'zlarni ham bashorat qilishi kerak), shuning uchun
m01.preprocess (stopword/stemming) emas, balki m01 normalizatsiyasi (apostrof + kichik harf)
ishlatiladi.
"""
from __future__ import annotations

import math
import re
import pickle
from collections import Counter

try:
    from m01_text_preprocessor import TextPreprocessor
except ImportError:
    from .m01_text_preprocessor import TextPreprocessor

_TOK = re.compile(r"[a-z][a-z']*")


class Autocomplete:
    """N-gram til modeli asosida so'z/ibora to'ldirish.

    Consumed by: Day 16 (pipeline demo).
    """

    def __init__(self) -> None:
        self._pre = TextPreprocessor()      # m01 — normalizatsiya uchun
        self._n = 2
        self._ctx: Counter = Counter()      # (n-1)-gram kontekst soni
        self._ngram: Counter = Counter()    # (kontekst, so'z) soni
        self._vocab: set = set()

    def _tokenize(self, text: str) -> list[str]:
        """m01 normalizatsiyasi (apostrof + kichik harf); barcha so'zlar saqlanadi."""
        return _TOK.findall(self._pre._normalize(text))

    def train(self, texts: list[list[str]], n: int = 2) -> None:
        """N-gram modelini sanab o'qitadi. texts — tokenlangan jumlalar ro'yxati."""
        self._n = n
        self._ctx.clear(); self._ngram.clear(); self._vocab.clear()
        for toks in texts:
            self._vocab.update(toks)
            for i in range(n - 1, len(toks)):
                ctx = tuple(toks[i - n + 1:i])
                self._ctx[ctx] += 1
                self._ngram[(ctx, toks[i])] += 1

    def _p_laplace(self, ctx: tuple, w: str) -> float:
        v = len(self._vocab)
        return (self._ngram[(ctx, w)] + 1) / (self._ctx[ctx] + v) if v else 0.0

    def complete(self, prefix: str, k: int = 3) -> list[str]:
        """Prefiksdan keyingi eng ehtimoliy k ta so'zni qaytaradi."""
        toks = self._tokenize(prefix)
        ctx = tuple(toks[-(self._n - 1):]) if self._n > 1 else ()
        cands = sorted({w for (c, w) in self._ngram if c == ctx})
        if not cands:
            cands = sorted(self._vocab)
        cands.sort(key=lambda w: -self._p_laplace(ctx, w))
        return cands[:k]

    def perplexity(self, text: str) -> float:
        """Matn uchun perplexity (add-1 smoothing bilan)."""
        toks = self._tokenize(text)
        if len(toks) < self._n:
            return float("inf")
        log_sum, count = 0.0, 0
        for i in range(self._n - 1, len(toks)):
            ctx = tuple(toks[i - self._n + 1:i])
            log_sum += math.log(self._p_laplace(ctx, toks[i]))
            count += 1
        return math.exp(-log_sum / count) if count else float("inf")

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"n": self._n, "ctx": self._ctx,
                         "ngram": self._ngram, "vocab": self._vocab}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            s = pickle.load(f)
        self._pre = TextPreprocessor()
        self._n, self._ctx = s["n"], s["ctx"]
        self._ngram, self._vocab = s["ngram"], s["vocab"]
