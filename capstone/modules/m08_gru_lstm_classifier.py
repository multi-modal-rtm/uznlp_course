"""
capstone/modules/m08_gru_lstm_classifier.py
GRULSTMClassifier — GRU yoki LSTM asosida matn (sentiment) tasniflash va taqqoslash.
Shartnoma: capstone/contracts.py :: GRULSTMClassifier
P8 (9-kun amaliyoti) da qurilgan; m01 (TextPreprocessor) ustiga quriladi.
Consumed by: m13 (baseline vs BERT), Day 16 (pipeline).

torch SHART EMAS (m07 ixtiyoriy-kutubxona namunasi):
  - Kaggle yo'li: nn.Embedding + nn.LSTM/nn.GRU(num_layers) + nn.Linear, CrossEntropyLoss + Adam.
  - Offline yo'l (torch'siz): tasodifiy-init GRU/LSTM forward (reservoir) + sklearn LogReg readout
    -- mo'rt BPTT'siz, ishonchli. Forward dinamikasi haqiqiy GRU/LSTM.
HAS_TORCH bayrog'i yo'lni tanlaydi. Yorliqlar: 'ijobiy' / 'salbiy' (qulflangan).
"""
from __future__ import annotations

import pickle
import time

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from m01_text_preprocessor import TextPreprocessor
except ImportError:
    from .m01_text_preprocessor import TextPreprocessor


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle yo'li: PyTorch GRU/LSTM klassifikatori
# ─────────────────────────────────────────────────────────────────────────────
if HAS_TORCH:
    class _TorchSeq(nn.Module):
        def __init__(self, vocab, dim, hidden, n_layers, n_cls, arch):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim, padding_idx=0)
            rnn_cls = nn.LSTM if arch == "lstm" else nn.GRU
            self.rnn = rnn_cls(dim, hidden, num_layers=n_layers, batch_first=True)
            self.out = nn.Linear(hidden, n_cls)
            self.arch = arch

        def forward(self, x):
            e = self.emb(x)
            out, h = self.rnn(e)
            last = h[0] if self.arch == "lstm" else h     # LSTM: (h, c)
            return self.out(last[-1])                      # oxirgi qatlam h_T


class GRULSTMClassifier:
    """GRU yoki LSTM bilan matn tasnifi va GRU-vs-LSTM taqqoslash.

    Consumed by: m13 (baseline vs BERT), Day 16 (pipeline).
    """

    def __init__(self, embed_dim: int = 32) -> None:
        self._pre = TextPreprocessor()
        self._dim = embed_dim
        self._w2i: dict[str, int] = {}
        self._labels: list[str] = []
        self._arch = "lstm"
        self._hidden = 128
        self._layers = 2
        self._model = None          # torch
        self._np = None             # offline: {arch, reservoir, readout}
        self._train_cache = None    # (seqs, ys) -- compare_report uchun

    # ─── kodlash ──────────────────────────────────────────────────────────────
    def _encode(self, text: str) -> list[int]:
        toks = self._pre.preprocess(text) if text.strip() else []
        ids = [self._w2i[t] for t in toks if t in self._w2i]
        return ids if ids else [0]

    def _prepare(self, texts, labels):
        token_lists = [self._pre.preprocess(t) if t.strip() else [] for t in texts]
        vocab = sorted({t for toks in token_lists for t in toks})
        self._w2i = {w: i + 1 for i, w in enumerate(vocab)}
        self._labels = sorted(set(labels))
        l2i = {lab: i for i, lab in enumerate(self._labels)}
        seqs = [[self._w2i[t] for t in toks if t in self._w2i] or [0] for toks in token_lists]
        ys = [l2i[lab] for lab in labels]
        return seqs, ys

    # ─── o'qitish ─────────────────────────────────────────────────────────────
    def fit(self, texts: list[str], labels: list[str], arch: str = "lstm",
            epochs: int = 10, hidden_size: int = 128, num_layers: int = 2,
            lr: float = 1e-3) -> None:
        """GRU yoki LSTM klassifikatorini o'qitadi (arch='lstm'|'gru')."""
        self._arch, self._hidden, self._layers = arch, hidden_size, num_layers
        seqs, ys = self._prepare(texts, labels)
        self._train_cache = (seqs, ys, epochs, hidden_size, num_layers, lr)
        if HAS_TORCH:
            self._model = self._fit_torch(seqs, ys, arch, epochs, hidden_size, num_layers, lr)
        else:
            self._np = self._fit_numpy(seqs, ys, arch, hidden_size)

    def _pad(self, seqs):
        T = max(len(s) for s in seqs)
        return [s + [0] * (T - len(s)) for s in seqs]

    def _fit_torch(self, seqs, ys, arch, epochs, hidden, layers, lr):
        torch.manual_seed(42)
        X = torch.tensor(self._pad(seqs), dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        model = _TorchSeq(len(self._w2i) + 1, self._dim, hidden, layers, len(self._labels), arch)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
        model.eval()
        return model

    # offline: tasodifiy-init rekurrent enkoder (reservoir) + LogReg readout
    def _reservoir(self, arch, hidden):
        rng = np.random.RandomState(42)
        V, d, H = len(self._w2i) + 1, self._dim, hidden
        P = {"E": rng.randn(V, d) * 0.3}
        n_gate = 4 if arch == "lstm" else 3
        P["W"] = rng.randn(n_gate, H, H + d) * (1.0 / np.sqrt(H + d))
        P["arch"], P["H"] = arch, H
        return P

    def _encode_state(self, P, seq):
        H, d, arch = P["H"], self._dim, P["arch"]
        h = np.zeros(H); c = np.zeros(H)
        for tid in seq:
            z = np.concatenate([h, P["E"][tid]])
            if arch == "lstm":
                f = _sigmoid(P["W"][0] @ z); i = _sigmoid(P["W"][1] @ z)
                o = _sigmoid(P["W"][2] @ z); g = np.tanh(P["W"][3] @ z)
                c = f * c + i * g; h = o * np.tanh(c)
            else:  # gru
                zt = _sigmoid(P["W"][0] @ z); r = _sigmoid(P["W"][1] @ z)
                zr = np.concatenate([r * h, P["E"][tid]])
                ht = np.tanh(P["W"][2] @ zr); h = (1 - zt) * h + zt * ht
        return h

    def _fit_numpy(self, seqs, ys, arch, hidden):
        from sklearn.linear_model import LogisticRegression
        P = self._reservoir(arch, hidden)
        Hs = np.array([self._encode_state(P, s) for s in seqs])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Hs, ys)
        return {"reservoir": P, "readout": clf}

    # ─── bashorat ─────────────────────────────────────────────────────────────
    def _proba(self, text):
        ids = self._encode(text)
        if HAS_TORCH and self._model is not None:
            with torch.no_grad():
                z = self._model(torch.tensor([ids], dtype=torch.long))[0].numpy()
            e = np.exp(z - z.max()); return e / e.sum()
        h = self._encode_state(self._np["reservoir"], ids)
        return self._np["readout"].predict_proba([h])[0]

    def predict(self, text: str) -> str:
        return self._labels[int(np.argmax(self._proba(text)))]

    # ─── GRU vs LSTM taqqoslash ─────────────────────────────────────────────────
    def compare_report(self) -> dict:
        """Ikkala arxitekturani bir xil vazifada o'qitib taqqoslaydi.

        Returns: {'lstm': {f1, accuracy, inference_time}, 'gru': {...}}.
        """
        from sklearn.metrics import f1_score
        assert self._train_cache is not None, "Avval fit() chaqiring."
        seqs, ys, epochs, hidden, layers, lr = self._train_cache
        report: dict = {}
        for arch in ("lstm", "gru"):
            if HAS_TORCH:
                model = self._fit_torch(seqs, ys, arch, epochs, hidden, layers, lr)
                X = torch.tensor(self._pad(seqs), dtype=torch.long)
                t0 = time.perf_counter()
                with torch.no_grad():
                    preds = model(X).argmax(1).numpy()
                infer = time.perf_counter() - t0
            else:
                np_state = self._fit_numpy(seqs, ys, arch, hidden)
                Hs = np.array([self._encode_state(np_state["reservoir"], s) for s in seqs])
                t0 = time.perf_counter()
                preds = np_state["readout"].predict(Hs)
                infer = time.perf_counter() - t0
            acc = float(np.mean(preds == np.array(ys)))
            f1 = float(f1_score(ys, preds, average="macro"))
            report[arch] = {"f1": round(f1, 4), "accuracy": round(acc, 4),
                            "inference_time": round(infer, 4)}
        return report

    # ─── saqlash / yuklash ──────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        state = {"w2i": self._w2i, "labels": self._labels, "dim": self._dim,
                 "arch": self._arch, "hidden": self._hidden, "layers": self._layers,
                 "has_torch": HAS_TORCH}
        if HAS_TORCH and self._model is not None:
            state["torch"] = self._model.state_dict()
        else:
            state["np"] = self._np
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            s = pickle.load(f)
        self._w2i, self._labels = s["w2i"], s["labels"]
        self._dim, self._arch = s["dim"], s["arch"]
        self._hidden, self._layers = s["hidden"], s["layers"]
        if HAS_TORCH and "torch" in s:
            self._model = _TorchSeq(len(self._w2i) + 1, self._dim, self._hidden,
                                    self._layers, len(self._labels), self._arch)
            self._model.load_state_dict(s["torch"]); self._model.eval()
            self._np = None
        else:
            self._np = s.get("np"); self._model = None
