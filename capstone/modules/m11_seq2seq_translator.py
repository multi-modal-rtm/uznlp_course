"""
capstone/modules/m11_seq2seq_translator.py
Seq2SeqTranslator — LSTM enkoder-dekoder + Bahdanau attention bilan tarjima.
Shartnoma: capstone/contracts.py :: Seq2SeqTranslator
P11 (12-kun amaliyoti) da qurilgan. Pedagogik demo (consumed_by: []) — yakuniy
pipeline da ishlatilmaydi.

Korpus: OPUS-100 uz-en (~20k) yoki offline mini parallel korpus. BLEU demo-sifatli
(kichik data) — bu pedagogik halol.

torch SHART EMAS (m09/m10 ixtiyoriy-kutubxona namunasi):
  - Kaggle yo'li: LSTM enkoder + Bahdanau attention + LSTM dekoder, teacher forcing.
  - Offline yo'l (torch'siz): soddalashtirilgan LUG'AT-asosli tarjimon (so'z-tekislash).
    (To'liq attention-seq2seq numpy BPTT JUDA og'ir — qurilmaydi.)
HAS_TORCH bayrog'i yo'lni tanlaydi. bleu() toza-python (nltk-ixtiyoriy).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

PAD, SOS, EOS = 0, 1, 2
_SPECIAL = ["<pad>", "<sos>", "<eos>"]


if HAS_TORCH:
    class _Seq2SeqAttn(nn.Module):
        def __init__(self, src_v, tgt_v, dim, hid):
            super().__init__()
            self.emb_s = nn.Embedding(src_v, dim, padding_idx=PAD)
            self.emb_t = nn.Embedding(tgt_v, dim, padding_idx=PAD)
            self.enc = nn.LSTM(dim, hid, batch_first=True)
            self.cell = nn.LSTMCell(dim + hid, hid)
            self.Wq = nn.Linear(hid, hid)
            self.Wk = nn.Linear(hid, hid)
            self.v = nn.Linear(hid, 1)
            self.out = nn.Linear(hid + hid, tgt_v)
            self.hid = hid

        def attention(self, s, enc_out):
            # Bahdanau: e_i = v^T tanh(Wq s + Wk h_i); alpha = softmax(e)
            score = self.v(torch.tanh(self.Wq(s).unsqueeze(1) + self.Wk(enc_out)))  # (B,T,1)
            alpha = torch.softmax(score, dim=1)                                     # (B,T,1)
            context = (alpha * enc_out).sum(1)                                      # (B,H)
            return context

        def forward(self, src, tgt):
            enc_out, (h, c) = self.enc(self.emb_s(src))     # enc_out: (B,T,H)
            s, cc = h[0], c[0]                              # (B,H)
            logits = []
            for t in range(tgt.size(1)):
                ctx = self.attention(s, enc_out)
                inp = torch.cat([self.emb_t(tgt[:, t]), ctx], dim=1)
                s, cc = self.cell(inp, (s, cc))
                logits.append(self.out(torch.cat([s, ctx], dim=1)))
            return torch.stack(logits, dim=1)               # (B,T,tgt_v)


class Seq2SeqTranslator:
    """LSTM enkoder-dekoder + Bahdanau attention tarjimon (pedagogik)."""

    def __init__(self, embed_dim: int = 32, hidden_size: int = 64) -> None:
        self._dim = embed_dim
        self._hid = hidden_size
        self._src2i: dict[str, int] = {}
        self._tgt2i: dict[str, int] = {}
        self._i2tgt: dict[int, str] = {}
        self._max_len = 50
        self._model = None            # torch
        self._dict = None             # offline: src_word -> tgt_word

    def _vocab(self, texts):
        words = sorted({w for t in texts for w in t.split()})
        v = {tok: i for i, tok in enumerate(_SPECIAL)}
        for w in words:
            v[w] = len(v)
        return v

    # ─── o'qitish ─────────────────────────────────────────────────────────────
    def train(self, src_texts: list[str], tgt_texts: list[str],
              epochs: int = 10, max_len: int = 50) -> None:
        """Parallel korpusda tarjimonni o'qitadi."""
        self._max_len = max_len
        self._src2i = self._vocab(src_texts)
        self._tgt2i = self._vocab(tgt_texts)
        self._i2tgt = {i: w for w, i in self._tgt2i.items()}
        if HAS_TORCH:
            self._train_torch(src_texts, tgt_texts, epochs)
        else:
            self._train_dict(src_texts, tgt_texts)

    def _encode(self, text, vocab, add_eos=True):
        ids = [vocab.get(w, PAD) for w in text.split()]
        return ids + ([EOS] if add_eos else [])

    def _train_torch(self, src_texts, tgt_texts, epochs):
        torch.manual_seed(42)
        src = [self._encode(t, self._src2i) for t in src_texts]
        tgt = [[SOS] + self._encode(t, self._tgt2i) for t in tgt_texts]
        Ts, Tt = max(len(s) for s in src), max(len(t) for t in tgt)
        X = torch.tensor([s + [PAD] * (Ts - len(s)) for s in src])
        Y = torch.tensor([t + [PAD] * (Tt - len(t)) for t in tgt])
        self._model = _Seq2SeqAttn(len(self._src2i), len(self._tgt2i), self._dim, self._hid)
        opt = torch.optim.Adam(self._model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
        self._model.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = self._model(X, Y[:, :-1])          # teacher forcing
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), Y[:, 1:].reshape(-1))
            loss.backward(); opt.step()
        self._model.eval()

    def _train_dict(self, src_texts, tgt_texts):
        # so'z-tekislash: src so'z bilan birga eng ko'p uchragan tgt so'z
        co = defaultdict(Counter)
        for s, t in zip(src_texts, tgt_texts):
            sw, tw = s.split(), t.split()
            for a in sw:
                for b in tw:
                    co[a][b] += 1
        self._dict = {a: c.most_common(1)[0][0] for a, c in co.items()}

    # ─── tarjima ──────────────────────────────────────────────────────────────
    def translate(self, text: str) -> str:
        """Manba matnini maqsad tilga tarjima qiladi (greedy)."""
        if HAS_TORCH and self._model is not None:
            return self._translate_torch(text)
        if not self._dict:
            return ""
        words = [self._dict.get(w, "") for w in text.split()]
        return " ".join(w for w in words if w)

    def _translate_torch(self, text):
        with torch.no_grad():
            src = torch.tensor([self._encode(text, self._src2i)])
            enc_out, (h, c) = self._model.enc(self._model.emb_s(src))
            s, cc = h[0], c[0]
            prev = torch.tensor([SOS]); out_words = []
            for _ in range(self._max_len):
                ctx = self._model.attention(s, enc_out)
                inp = torch.cat([self._model.emb_t(prev), ctx], dim=1)
                s, cc = self._model.cell(inp, (s, cc))
                logits = self._model.out(torch.cat([s, ctx], dim=1))
                nxt = int(logits.argmax(1))
                if nxt == EOS:
                    break
                if nxt not in (PAD, SOS):
                    out_words.append(self._i2tgt.get(nxt, ""))
                prev = torch.tensor([nxt])
        return " ".join(w for w in out_words if w)

    # ─── BLEU (toza-python; nltk-ixtiyoriy) ─────────────────────────────────────
    def bleu(self, references: list[list[str]], hypotheses: list[str]) -> float:
        """Korpus BLEU (4-gram, brevity penalty). references[i] — etalon tokenlar;
        hypotheses[i] — tarjima (str yoki tokenlar)."""
        def toks(x):
            return x.split() if isinstance(x, str) else list(x)
        weights = [0.25, 0.25, 0.25, 0.25]
        p_num = [0] * 4; p_den = [0] * 4
        ref_len = hyp_len = 0
        for ref, hyp in zip(references, hypotheses):
            r, h = toks(ref), toks(hyp)
            ref_len += len(r); hyp_len += len(h)
            for n in range(1, 5):
                r_ng = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
                h_ng = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
                overlap = sum((h_ng & r_ng).values())
                p_num[n - 1] += overlap
                p_den[n - 1] += max(sum(h_ng.values()), 1)
        # geometrik o'rtacha (silliqlash bilan)
        s = 0.0
        for w, num, den in zip(weights, p_num, p_den):
            p = num / den if den else 0.0
            s += w * math.log(p) if p > 0 else w * math.log(1e-9)
        bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
        return float(bp * math.exp(s))
