"""
capstone/modules/m09_text_generator.py
TextGenerator — char-darajali LSTM (yoki char n-gram) bilan matn generatsiya.
Shartnoma: capstone/contracts.py :: TextGenerator
P9 (10-kun amaliyoti) da qurilgan. Pedagogik demo (consumed_by: []) — yakuniy
pipeline da ishlatilmaydi.

torch SHART EMAS (m07/m08 ixtiyoriy-kutubxona namunasi):
  - Kaggle yo'li: char-LSTM (nn.Embedding + nn.LSTM(num_layers) + nn.Linear),
    next-char CrossEntropyLoss + Adam.
  - Offline yo'l (torch'siz): char n-gram (m05 g'oyasi) + temperature sampling --
    mo'rt BPTT'siz, barqaror.
HAS_TORCH bayrog'i yo'lni tanlaydi. Char-darajali: train xom MATN qatorini oladi.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _temp_sample(logits: np.ndarray, temperature: float, rng: np.random.RandomState) -> int:
    """Temperature softmax (L9 [I2]) bo'yicha indeks tanlaydi."""
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 1e-6:                       # T->0: argmax (greedy)
        return int(np.argmax(logits))
    z = logits / temperature
    z -= z.max()                                  # barqaror softmax
    p = np.exp(z); p /= p.sum()
    return int(rng.choice(len(p), p=p))


if HAS_TORCH:
    class _CharLSTM(nn.Module):
        def __init__(self, vocab, hidden, layers):
            super().__init__()
            self.emb = nn.Embedding(vocab, hidden)
            self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
            self.fc = nn.Linear(hidden, vocab)

        def forward(self, x, h=None):
            e = self.emb(x)
            out, h = self.lstm(e, h)
            return self.fc(out), h


class TextGenerator:
    """Char-darajali LSTM yoki char n-gram bilan matn generatsiya (pedagogik)."""

    def __init__(self, ngram_order: int = 4) -> None:
        self._chars: list[str] = []
        self._c2i: dict[str, int] = {}
        self._hidden = 128
        self._layers = 2
        self._model = None            # torch
        self._ngram = None            # offline: {order, counts}
        self._order = ngram_order
        self._rng = np.random.RandomState(42)

    def _build_vocab(self, text: str) -> None:
        self._chars = sorted(set(text))
        self._c2i = {c: i for i, c in enumerate(self._chars)}

    # ─── o'qitish ─────────────────────────────────────────────────────────────
    def train(self, text: str, epochs: int = 20, hidden_size: int = 128) -> None:
        """Char-darajali generativ modelni o'qitadi (xom matn)."""
        if not text:
            raise ValueError("train(): matn bo'sh bo'lmasligi kerak.")
        self._hidden = hidden_size
        self._build_vocab(text)
        if HAS_TORCH:
            self._train_torch(text, epochs, hidden_size)
        else:
            self._train_ngram(text)

    def _train_torch(self, text, epochs, hidden, seq_len=40):
        torch.manual_seed(42)
        data = np.array([self._c2i[c] for c in text], dtype=np.int64)
        # chunklar: (input, target=keyingi belgi)
        xs, ys = [], []
        for i in range(0, len(data) - seq_len - 1, seq_len):
            xs.append(data[i:i + seq_len])
            ys.append(data[i + 1:i + seq_len + 1])
        if not xs:                                    # qisqa matn: butun ketma-ketlik bitta chunk
            xs.append(data[:-1]); ys.append(data[1:])
        X = torch.tensor(np.array(xs), dtype=torch.long)
        Y = torch.tensor(np.array(ys), dtype=torch.long)
        self._model = _CharLSTM(len(self._chars), hidden, self._layers)
        opt = torch.optim.Adam(self._model.parameters(), lr=5e-3)
        loss_fn = nn.CrossEntropyLoss()
        self._model.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits, _ = self._model(X)
            loss = loss_fn(logits.reshape(-1, len(self._chars)), Y.reshape(-1))
            loss.backward()
            opt.step()
        self._model.eval()

    def _train_ngram(self, text):
        k = self._order - 1                            # kontekst uzunligi (oldingi k belgi)
        counts: dict = defaultdict(Counter)
        for i in range(1, len(text)):
            ctx = text[max(0, i - k):i]                # i dan OLDINGI k belgi
            counts[ctx][text[i]] += 1
        self._ngram = {"k": k, "counts": counts}

    # ─── generatsiya ──────────────────────────────────────────────────────────
    def generate(self, seed: str, length: int = 200, temperature: float = 0.7) -> str:
        """Seed dan boshlab `length` ta belgi autoregressiv generatsiya qiladi."""
        if HAS_TORCH and self._model is not None:
            return self._generate_torch(seed, length, temperature)
        return self._generate_ngram(seed, length, temperature)

    def _generate_torch(self, seed, length, temperature):
        result = seed
        idx = [self._c2i.get(c, 0) for c in seed] or [0]
        with torch.no_grad():
            inp = torch.tensor([idx])
            out, h = self._model(inp)
            last = idx[-1]
            for _ in range(length):
                out, h = self._model(torch.tensor([[last]]), h)
                logits = out[0, -1].numpy()
                last = _temp_sample(logits, temperature, self._rng)
                result += self._chars[last]
        return result

    def _generate_ngram(self, seed, length, temperature):
        k = self._ngram["k"]; counts = self._ngram["counts"]
        result = seed
        for _ in range(length):
            ctx = result[-k:]
            dist = counts.get(ctx)
            while dist is None and len(ctx) > 0:      # backoff: qisqaroq kontekst
                ctx = ctx[1:]
                dist = counts.get(ctx)
            if not dist:
                result += self._chars[self._rng.randint(len(self._chars))]
                continue
            chars = list(dist.keys())
            logits = np.log(np.array([dist[c] for c in chars], dtype=np.float64))
            j = _temp_sample(logits, temperature, self._rng)
            result += chars[j]
        return result
