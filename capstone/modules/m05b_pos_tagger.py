"""
capstone/modules/m05b_pos_tagger.py
POSTagger — Yashirin Markov Modeli + Viterbi orqali so'z turkumini teglash.
Shartnoma: capstone/contracts.py :: POSTagger
P5 (6-kun amaliyoti) da qurilgan. Pedagogik demo — yakuniy pipelineda ishlatilmaydi.

Viterbi log fazoda hisoblanadi (underflow oldini olish — L5 [L4]-slayd).
"""
from __future__ import annotations

import math
import pickle
from collections import Counter

try:
    from m01_text_preprocessor import TextPreprocessor
except ImportError:
    from .m01_text_preprocessor import TextPreprocessor

_EPS = 1e-12


class POSTagger:
    """Yashirin Markov Modeli + Viterbi orqali so'z turkumini teglash."""

    def __init__(self) -> None:
        self._pre = TextPreprocessor()      # m01 — token normalizatsiyasi
        self._states: list[str] = []
        self._pi: dict = {}
        self._A: dict = {}                  # (from, to) -> prob
        self._B: dict = {}                  # (state, word) -> prob

    def train(self, tagged_sentences: list) -> None:
        """HMM parametrlarini (pi, A, B) sanab hisoblaydi."""
        init, trans, emit = Counter(), Counter(), Counter()
        tag_tot, first_tot = Counter(), 0
        for sent in tagged_sentences:
            if not sent:
                continue
            first_tot += 1
            init[sent[0][1]] += 1
            for j, (w, t) in enumerate(sent):
                emit[(t, w.lower())] += 1
                tag_tot[t] += 1
                if j > 0:
                    trans[(sent[j - 1][1], t)] += 1
        self._states = sorted(tag_tot)
        self._pi = {s: init[s] / first_tot for s in self._states} if first_tot else {}
        from_tot = Counter()
        for (a, b), c in trans.items():
            from_tot[a] += c
        self._A = {}
        for a in self._states:
            for b in self._states:
                self._A[(a, b)] = trans[(a, b)] / from_tot[a] if from_tot[a] else 0.0
        self._B = {(t, w): c / tag_tot[t] for (t, w), c in emit.items()}

    def _emit(self, state: str, word: str) -> float:
        return self._B.get((state, word.lower()), 1e-6)   # OOV uchun kichik ehtimol

    def tag(self, tokens: list[str]) -> list:
        """Viterbi (log fazo) bilan har token uchun teg bashorat qiladi."""
        if not tokens or not self._states:
            return [(w, "") for w in tokens]
        toks = [self._pre._normalize(t) for t in tokens]
        S = self._states

        def lg(x):
            return math.log(x + _EPS)

        delta = [{s: lg(self._pi.get(s, 0.0)) + lg(self._emit(s, toks[0])) for s in S}]
        back = [{}]
        for i in range(1, len(toks)):
            d, bk = {}, {}
            for s in S:
                best = max(S, key=lambda p: delta[i - 1][p] + lg(self._A.get((p, s), 0.0)))
                d[s] = delta[i - 1][best] + lg(self._A.get((best, s), 0.0)) + lg(self._emit(s, toks[i]))
                bk[s] = best
            delta.append(d); back.append(bk)
        last = max(S, key=lambda s: delta[-1][s])
        path = [last]
        for i in range(len(toks) - 1, 0, -1):
            last = back[i][last]
            path.insert(0, last)
        return list(zip(tokens, path))
