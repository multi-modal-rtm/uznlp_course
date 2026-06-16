"""
capstone/modules/m10_ner_tagger.py
NERTagger — Bi-LSTM asosida IOB2 formatida nomlangan obyektlarni teglash.
Shartnoma: capstone/contracts.py :: NERTagger
P10 (11-kun amaliyoti) da qurilgan. Consumed by: m15 (agent: extract_entities), Day 16.

Korpus: WikiANN uz (~200 jumlat) yoki offline IOB2 mini-korpus. F1 past kutiladi
(kam ma'lumot) -- bu o'zbek kabi kam-resursli tillar uchun pedagogik halol holat.

torch SHART EMAS (m08 ixtiyoriy-kutubxona namunasi):
  - Kaggle yo'li: nn.Embedding + nn.LSTM(bidirectional) + nn.Linear (per-token teg).
  - Offline yo'l (torch'siz): tasodifiy-init Bi-LSTM forward (reservoir) + sklearn LogReg
    readout (per-token). Mo'rt BPTT'siz, barqaror.
HAS_TORCH bayrog'i yo'lni tanlaydi.
"""
from __future__ import annotations

import pickle

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


if HAS_TORCH:
    class _BiLSTMNER(nn.Module):
        def __init__(self, vocab, dim, hid, n_tags):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim, padding_idx=0)
            self.lstm = nn.LSTM(dim, hid, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(2 * hid, n_tags)

        def forward(self, x):
            out, _ = self.lstm(self.emb(x))      # (B, T, 2*hid)
            return self.fc(out)                  # (B, T, n_tags)


class NERTagger:
    """Bi-LSTM (yoki reservoir) IOB2 NER teglagichi.

    Consumed by: m15 (agent tool: extract_entities), Day 16 (pipeline).
    """

    def __init__(self, embed_dim: int = 32, hidden_size: int = 64) -> None:
        self._dim = embed_dim
        self._hidden = hidden_size
        self._w2i: dict[str, int] = {}        # 0 = PAD/UNK
        self._tags: list[str] = []
        self._t2i: dict[str, int] = {}
        self._model = None                    # torch
        self._np = None                       # offline: {reservoir, readout}

    # ─── kodlash ──────────────────────────────────────────────────────────────
    def _tokenize(self, text: str) -> list[str]:
        return text.split()

    def _build(self, tagged_sentences):
        vocab, tagset = set(), set()
        for sent in tagged_sentences:
            for tok, tag in sent:
                vocab.add(tok); tagset.add(tag)
        self._w2i = {w: i + 1 for i, w in enumerate(sorted(vocab))}     # 0 = PAD/UNK
        self._tags = sorted(tagset)
        if "O" in self._tags:                                          # O ni boshiga
            self._tags.remove("O"); self._tags = ["O"] + self._tags
        self._t2i = {t: i for i, t in enumerate(self._tags)}

    def _encode_words(self, tokens):
        return [self._w2i.get(t, 0) for t in tokens]

    # ─── o'qitish ─────────────────────────────────────────────────────────────
    def fit(self, tagged_sentences, epochs: int = 15, lr: float = 0.01) -> None:
        """IOB2 teglangan jumlalarda NER modelini o'qitadi.

        Args: tagged_sentences -- [[(token, teg), ...], ...].
        """
        self._build(tagged_sentences)
        if HAS_TORCH:
            self._fit_torch(tagged_sentences, epochs, lr)
        else:
            self._fit_numpy(tagged_sentences)

    def _fit_torch(self, sents, epochs, lr):
        torch.manual_seed(42)
        seqs = [[self._w2i[t] for t, _ in s] for s in sents]
        tagseqs = [[self._t2i[g] for _, g in s] for s in sents]
        T = max(len(s) for s in seqs)
        X = torch.tensor([s + [0] * (T - len(s)) for s in seqs], dtype=torch.long)
        Y = torch.tensor([t + [-100] * (T - len(t)) for t in tagseqs], dtype=torch.long)
        self._model = _BiLSTMNER(len(self._w2i) + 1, self._dim, self._hidden, len(self._tags))
        opt = torch.optim.Adam(self._model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self._model.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = self._model(X)                          # (B,T,n_tags)
            loss = loss_fn(logits.reshape(-1, len(self._tags)), Y.reshape(-1))
            loss.backward(); opt.step()
        self._model.eval()

    # offline: tasodifiy-init Bi-LSTM forward + LogReg readout
    def _reservoir(self):
        rng = np.random.RandomState(42)
        V, d, H = len(self._w2i) + 1, self._dim, self._hidden
        return {"E": rng.randn(V, d) * 0.3,
                "Wf": rng.randn(4, H, H + d) / np.sqrt(H + d),     # forward LSTM
                "Wb": rng.randn(4, H, H + d) / np.sqrt(H + d),     # backward LSTM
                "H": H}

    def _lstm_pass(self, P, W, emb_seq):
        H = P["H"]; h = np.zeros(H); c = np.zeros(H); out = []
        for e in emb_seq:
            z = np.concatenate([h, e])
            f = _sigmoid(W[0] @ z); i = _sigmoid(W[1] @ z)
            o = _sigmoid(W[2] @ z); g = np.tanh(W[3] @ z)
            c = f * c + i * g; h = o * np.tanh(c); out.append(h.copy())
        return out

    def _encode_states(self, P, word_idx):
        emb = [P["E"][i] for i in word_idx]
        fwd = self._lstm_pass(P, P["Wf"], emb)
        bwd = self._lstm_pass(P, P["Wb"], emb[::-1])[::-1]
        # xususiyat: joriy token embeddingi (so'z identifikatori) + chap/o'ng holat
        return np.array([np.concatenate([e, f, b]) for e, f, b in zip(emb, fwd, bwd)])

    def _fit_numpy(self, sents):
        from sklearn.linear_model import LogisticRegression
        P = self._reservoir()
        feats, ys = [], []
        for s in sents:
            wi = [self._w2i[t] for t, _ in s]
            H = self._encode_states(P, wi)
            feats.extend(H); ys.extend([g for _, g in s])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(np.array(feats), ys)
        self._np = {"reservoir": P, "readout": clf}

    # ─── bashorat ─────────────────────────────────────────────────────────────
    def predict(self, text: str) -> list[tuple[str, str]]:
        """Matn tokenlari uchun (token, IOB2-teg) juftlarini qaytaradi."""
        tokens = self._tokenize(text)
        if not tokens:
            return []
        wi = self._encode_words(tokens)
        if HAS_TORCH and self._model is not None:
            with torch.no_grad():
                logits = self._model(torch.tensor([wi], dtype=torch.long))[0].numpy()
            tags = [self._tags[int(np.argmax(row))] for row in logits]
        else:
            H = self._encode_states(self._np["reservoir"], wi)
            tags = [str(t) for t in self._np["readout"].predict(H)]
        return list(zip(tokens, tags))

    def entities(self, text: str) -> list[dict]:
        """IOB2 teglardan entitylarni yig'adi: [{'text','label','start','end'}, ...]."""
        tagged = self.predict(text)
        result, pos = [], 0
        cur = None
        spans = []
        for tok, tag in tagged:
            start = text.find(tok, pos); end = start + len(tok); pos = end
            spans.append((tok, tag, start, end))
        for tok, tag, start, end in spans:
            if tag.startswith("B-"):
                if cur:
                    result.append(cur)
                cur = {"text": tok, "label": tag[2:], "start": start, "end": end}
            elif tag.startswith("I-") and cur and cur["label"] == tag[2:]:
                cur["text"] += " " + tok; cur["end"] = end
            else:
                if cur:
                    result.append(cur); cur = None
        if cur:
            result.append(cur)
        return result

    # ─── saqlash / yuklash ──────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        state = {"w2i": self._w2i, "tags": self._tags, "t2i": self._t2i,
                 "dim": self._dim, "hidden": self._hidden, "has_torch": HAS_TORCH}
        if HAS_TORCH and self._model is not None:
            state["torch"] = self._model.state_dict()
        else:
            state["np"] = self._np
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            s = pickle.load(f)
        self._w2i, self._tags, self._t2i = s["w2i"], s["tags"], s["t2i"]
        self._dim, self._hidden = s["dim"], s["hidden"]
        if HAS_TORCH and "torch" in s:
            self._model = _BiLSTMNER(len(self._w2i) + 1, self._dim, self._hidden, len(self._tags))
            self._model.load_state_dict(s["torch"]); self._model.eval()
            self._np = None
        else:
            self._np = s.get("np"); self._model = None
