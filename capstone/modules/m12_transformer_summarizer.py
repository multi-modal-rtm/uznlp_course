"""
capstone/modules/m12_transformer_summarizer.py
TransformerSummarizer — mini Transformer enkoder-dekoder bilan matn umumlashtirish.
Shartnoma: capstone/contracts.py :: TransformerSummarizer
P12 (13-kun amaliyoti) da qurilgan. HAQIQIY pipeline moduli (m10 kabi):
save/load BOR, consumed_by [15, 16] (m15 agent: summarize_text; Day 16 pipeline).

Korpus: Wikipedia uz lead-paragraph juftlari (CC-BY-SA) yoki offline mini korpus.
ROUGE demo-sifatli (kichik data) -- bu pedagogik halol.

torch SHART EMAS (m10/m12 haqiqiy-modul namunasi):
  - Kaggle yo'li: nn.Transformer enkoder-dekoder + sinusoidal positional encoding,
    teacher forcing + Adam; greedy decode bilan summarize.
  - Offline yo'l (torch'siz): soddalashtirilgan EKSTRAKTIV xulosalagich
    (chastota-asosli gap tanlash). To'liq Transformer numpy BPTT QURILMAYDI (mo'rt).
HAS_TORCH bayrog'i yo'lni tanlaydi. rouge1() toza-python, DICT qaytaradi.
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

PAD, SOS, EOS = 0, 1, 2
_SPECIAL = ["<pad>", "<sos>", "<eos>"]


def _sinusoidal_pe(n_pos: int, d: int) -> np.ndarray:
    """Sinusoidal positional encoding (L12 [I2] formulasi)."""
    pe = np.zeros((n_pos, d))
    pos = np.arange(n_pos)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / d)
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


if HAS_TORCH:
    class _TransformerSeq2Seq(nn.Module):
        def __init__(self, src_v, tgt_v, d_model, nhead, num_layers=2, ff=256, max_len=256):
            super().__init__()
            self.emb_s = nn.Embedding(src_v, d_model, padding_idx=PAD)
            self.emb_t = nn.Embedding(tgt_v, d_model, padding_idx=PAD)
            pe = torch.tensor(_sinusoidal_pe(max_len, d_model), dtype=torch.float32)
            self.register_buffer("pe", pe)
            self.transformer = nn.Transformer(
                d_model=d_model, nhead=nhead,
                num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                dim_feedforward=ff, batch_first=True)
            self.out = nn.Linear(d_model, tgt_v)
            self.d_model = d_model

        def _embed(self, x, emb):
            e = emb(x) * math.sqrt(self.d_model)
            return e + self.pe[:x.size(1)].unsqueeze(0)

        def forward(self, src, tgt):
            s = self._embed(src, self.emb_s)
            t = self._embed(tgt, self.emb_t)
            T = tgt.size(1)
            tgt_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt.device), diagonal=1)
            src_kpm = (src == PAD)
            tgt_kpm = (tgt == PAD)
            h = self.transformer(s, t, tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_kpm,
                                  tgt_key_padding_mask=tgt_kpm,
                                  memory_key_padding_mask=src_kpm)
            return self.out(h)


class TransformerSummarizer:
    """Mini Transformer enkoder-dekoder asosida matn qisqartirish (haqiqiy modul).

    Consumed by: m15 (agent tool: summarize_text), Day 16 (pipeline).
    """

    def __init__(self) -> None:
        self._src2i: dict[str, int] = {}
        self._tgt2i: dict[str, int] = {}
        self._i2tgt: dict[int, str] = {}
        self._d_model = 128
        self._nhead = 4
        self._model = None            # torch
        self._df: Counter | None = None   # offline ekstraktiv: hujjat-chastota

    # ─── lug'at va kodlash ──────────────────────────────────────────────────────
    def _vocab(self, texts):
        words = sorted({w for t in texts for w in t.split()})
        v = {tok: i for i, tok in enumerate(_SPECIAL)}
        for w in words:
            v[w] = len(v)
        return v

    def _encode(self, text, vocab):
        return [vocab.get(w, PAD) for w in text.split()] + [EOS]

    # ─── o'qitish ─────────────────────────────────────────────────────────────
    def train(self, src_texts: list[str], tgt_texts: list[str],
              epochs: int = 10, d_model: int = 128, nhead: int = 4) -> None:
        """Maqola-xulosa juftlarida umumlashtirgichni o'qitadi."""
        self._d_model = d_model
        self._nhead = nhead
        self._src2i = self._vocab(src_texts)
        self._tgt2i = self._vocab(tgt_texts)
        self._i2tgt = {i: w for w, i in self._tgt2i.items()}
        if HAS_TORCH:
            self._train_torch(src_texts, tgt_texts, epochs)
        else:
            self._train_extractive(src_texts)

    def _train_torch(self, src_texts, tgt_texts, epochs):
        torch.manual_seed(42)
        src = [self._encode(t, self._src2i) for t in src_texts]
        tgt = [[SOS] + self._encode(t, self._tgt2i) for t in tgt_texts]
        Ts, Tt = max(len(s) for s in src), max(len(t) for t in tgt)
        X = torch.tensor([s + [PAD] * (Ts - len(s)) for s in src])
        Y = torch.tensor([t + [PAD] * (Tt - len(t)) for t in tgt])
        self._model = _TransformerSeq2Seq(len(self._src2i), len(self._tgt2i),
                                           self._d_model, self._nhead)
        opt = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
        self._model.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = self._model(X, Y[:, :-1])          # teacher forcing
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), Y[:, 1:].reshape(-1))
            loss.backward(); opt.step()
        self._model.eval()

    def _train_extractive(self, src_texts):
        # ekstraktiv uchun global so'z chastotasi (gap tanlash ballari uchun)
        df = Counter()
        for t in src_texts:
            df.update(re.findall(r"[\w']+", t.lower()))
        self._df = df

    # ─── xulosa ──────────────────────────────────────────────────────────────
    def summarize(self, text: str, max_length: int = 60) -> str:
        """Berilgan matn uchun qisqa xulosa generatsiya qiladi."""
        if HAS_TORCH and self._model is not None:
            return self._summarize_torch(text, max_length)
        return self._summarize_extractive(text, max_length)

    def _summarize_torch(self, text, max_length):
        self._model.eval()
        with torch.no_grad():
            src = torch.tensor([self._encode(text, self._src2i)])
            ys = [SOS]
            for _ in range(max_length):
                tgt = torch.tensor([ys])
                logits = self._model(src, tgt)
                nxt = int(logits[0, -1].argmax())
                if nxt == EOS:
                    break
                ys.append(nxt)
        words = [self._i2tgt.get(i, "") for i in ys[1:]]
        return " ".join(w for w in words if w)

    def _summarize_extractive(self, text, max_length):
        # chastota-asosli gap tanlash; asl tartibni saqlaydi
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        if not sents:
            return ""
        freq = self._df if self._df else Counter(re.findall(r"[\w']+", text.lower()))

        def score(s):
            toks = re.findall(r"[\w']+", s.lower())
            return sum(freq[t] for t in toks) / (len(toks) + 1e-9)

        ranked = sorted(range(len(sents)), key=lambda k: score(sents[k]), reverse=True)
        chosen, n = set(), 0
        for k in ranked:
            wc = len(sents[k].split())
            if n + wc > max_length and chosen:
                break
            chosen.add(k); n += wc
        return " ".join(sents[k] for k in range(len(sents)) if k in chosen)

    # ─── ROUGE-1 (toza-python; DICT qaytaradi) ──────────────────────────────────
    def rouge1(self, references: list[str], hypotheses: list[str]) -> dict[str, float]:
        """ROUGE-1 precision, recall, F1 ni qaytaradi (korpus o'rtachasi).

        references[i], hypotheses[i] -- matn (str). Qaytaradi:
        {"precision": float, "recall": float, "f1": float}, har biri [0,1].
        """
        ps, rs, fs = [], [], []
        for ref, hyp in zip(references, hypotheses):
            r, h = ref.split(), hyp.split()
            overlap = sum((Counter(h) & Counter(r)).values())   # clipping
            p = overlap / len(h) if h else 0.0
            rec = overlap / len(r) if r else 0.0
            f = 2 * p * rec / (p + rec) if (p + rec) else 0.0
            ps.append(p); rs.append(rec); fs.append(f)
        n = len(ps) or 1
        return {"precision": sum(ps) / n, "recall": sum(rs) / n, "f1": sum(fs) / n}

    # ─── saqlash / yuklash ──────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        state = {"src2i": self._src2i, "tgt2i": self._tgt2i, "i2tgt": self._i2tgt,
                 "d_model": self._d_model, "nhead": self._nhead, "has_torch": HAS_TORCH}
        if HAS_TORCH and self._model is not None:
            state["torch"] = self._model.state_dict()
        else:
            state["df"] = self._df
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            s = pickle.load(f)
        self._src2i, self._tgt2i = s["src2i"], s["tgt2i"]
        self._i2tgt = s["i2tgt"]
        self._d_model, self._nhead = s["d_model"], s["nhead"]
        if HAS_TORCH and "torch" in s:
            self._model = _TransformerSeq2Seq(len(self._src2i), len(self._tgt2i),
                                              self._d_model, self._nhead)
            self._model.load_state_dict(s["torch"]); self._model.eval()
            self._df = None
        else:
            self._df = s.get("df"); self._model = None
