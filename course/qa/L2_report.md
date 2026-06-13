# QA Report: Day 2 Lecture (L2)

**Artifact**: `course/lectures/d02_klassik_tasnif.tex`
**Date**: 2026-06-13
**Reviewer**: Claude Code (automated gates)

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| pdflatex ×2, zero `^!` errors | DEFERRED | Overleaf (per feedback_pdflatex_policy) |
| Overfull `\hbox` >10 pt | DEFERRED | Overleaf |
| Slide count 40–44 rendered (exception) | **PASS** | 43 rendered (37 body + 1 maketitle + 5 AtBeginSection) |
| All archetypes [A]–[S] present | **PASS** | See table below |
| Terminology grep clean | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` |
| Self-contained (no `\input`/`\includegraphics`/`\bibliography`) | **PASS** | 0 matches |
| Locked [I2] values | **PASS** | ratio=3.375 exact; rounded display ≈3.38 not shown on slide |
| Traceability comment | **PASS** | Lines 692 and 1374 cite `Ma'ruza L2 [I2]-slayd` |

**Overall (non-compile gates): ALL PASS**

---

## Slide Count Detail

- Body frames: 37
  - Section 1 "Kirish": 4 (B, C, D, E)
  - Section 2 "Logistik Regressiya": 8 (F1, G1, H1a, H1b, I1, J1, K1, L1)
  - Section 3 "Naive Bayes": 8 (F2, G2, H2a, H2b, I2, J2, K2, L2)
  - Section 4 "Metrikalar": 8 (F3, G3, H3a, H3b, I3, J3, K3, L3)
  - Section 5 "Etika": 3 (F4, G4, J4)
  - Section 6 "Xulosa": 6 (M, N, O, P, Q, R)
- Appendix frames: 3 (S1, S2, S3) — not counted
- Maketitle: 1
- AtBeginSection firings: 5 (6 sections, fires n−1)
- **Rendered total: 43** — within exception ceiling 40–44

---

## Archetype Checklist

| Archetype | Frame(s) | Present |
|-----------|----------|---------|
| [A] Title (maketitle) | `\maketitle` | ✓ |
| [B] Recap quiz (2 Q + \pause + 2 A) | "Takrorlash: L1 Natijalarini Tekshiramiz" | ✓ |
| [C] Measurable objectives (4 verbs) | "Bugungi Maqsadlar" | ✓ |
| [D] Plan with time-budget table | "Dars Rejasi: To'rt Tsikl" | ✓ |
| [E] Problem-first motivation | "Muammo: Vektori Bor — Sinfi Yo'q" | ✓ |
| [F] Intuition ×4 | F1, F2, F3, F4 | ✓ |
| [G] defbox + `\bunda{}` ×4 | G1 (LR), G2 (NB), G3 (metrics), G4 (ethics risks) | ✓ |
| [H] Step-by-step derivation ×3 | H1a, H1b (LR); H2a, H2b (NB); H3a, H3b (metrics) | ✓ |
| [I] Hand-worked example ×3 | I1 (LR forward pass), I2 (NB locked), I3 (Model B metrics) | ✓ |
| [J] warnbox→\pause→okbox ×4 | J1, J2, J3, J4 | ✓ |
| [K] Code↔formula bridge ×3 | K1 (LR), K2 (NB), K3 (metrics) | ✓ |
| [L] Pitfall ×3 | L1 (imbalanced), L2 (CI assumption), L3 (macro vs weighted) | ✓ |
| [M] Uzbek-language angle | "O'zbek Tili va Klassifikatsiya Xususiyatlari" | ✓ |
| [N] Synthesis comparison table | "LR va NB: Qaysi Birini Tanlash?" (8-row table) | ✓ |
| [O] Seminal paper + discussion Q | McCallum & Nigam (1998) | ✓ |
| [P] Objectives checkmarked (`\bajarildi`) | "Maqsadlar Bajarildi" | ✓ |
| [Q] Bridge to practice (TikZ pipeline) | "Ertangi P2: Sentiment Klassifikatori Qurasiz" | ✓ |
| [R] References | "Adabiyotlar" (4 items) | ✓ |
| [S] Appendix backup slides | S1 (sigmoid), S2 (Bayes proof), S3 (F-beta) | ✓ |

Notes:
- Ethics cycle [G4] uses warnboxes (3 risk categories) instead of a formula defbox; no `\bunda{}` because ethics has no symbol key. Ethics cycle intentionally omits [H] and [I] per spec ("no [H]/[I]").
- [fragile] applied to all 3 lstlisting frames: K1, K2, K3.
- TikZ in [Q] uses `align=center` in all node styles (not `text centered`) — safe for `\\` line breaks.

---

## Locked Hand-Example Values ([I2])

Training corpus: pos=["yaxshi film","ajoyib film"], neg=["yomon film"]

| Computation | Value |
|-------------|-------|
| \|V\| | 4 {yaxshi,ajoyib,yomon,film} |
| P(pos) | 2/3 |
| P(neg) | 1/3 |
| N\_pos tokens | 4 |
| N\_neg tokens | 2 |
| P(film\|pos) | (2+1)/(4+4) = 3/8 |
| P(yaxshi\|pos) | (1+1)/(4+4) = 1/4 |
| P(film\|neg) | (1+1)/(2+4) = 1/3 |
| P(yaxshi\|neg) | (0+1)/(2+4) = **1/6** ← zero-count demo |
| score\_pos | (2/3)(1/4)(3/8) = 6/96 = 1/16 = 0.0625 |
| score\_neg | (1/3)(1/6)(1/3) = 2/108 = 1/54 ≈ 0.0185 |
| ratio | 54/16 = 27/8 = **3.375** (display: ≈3.38) |
| Verdict | musbat (positive) |

P2 assert: `abs(ratio - 3.375) < 0.01`

---

## Additional Numeric Values (for P2 alignment)

**[I1] LR forward pass** (x=[1.0,0.0], w=[0.8,−1.2], b=−0.3):
- z = 0.5; σ(0.5) ≈ 0.622; prediction: musbat; loss: −ln(0.622) ≈ 0.475

**[J1] LR your-turn** (x=[0,1]):
- z = −1.5; σ(−1.5) ≈ 0.182; prediction: salbiy

**[J2] NB your-turn** (test: "yomon", α=1):
- P(yomon|pos) = 1/8; P(yomon|neg) = 1/3
- ratio = 3/4 = 0.75 < 1 → salbiy

**[I3] Model B metrics** (TP=60, TN=820, FP=80, FN=40):
- A = 0.88; P ≈ 0.429; R = 0.60; F1 ≈ 0.50

**[J3] Model C your-turn** (TP=80, TN=780, FP=120, FN=20):
- A = 0.86; P = 0.40; R = 0.80; F1 ≈ 0.533

---

## Pending

- pdflatex ×2 compile via Overleaf (DEFERRED — same policy as L1)
- L2 PDF to be committed after successful Overleaf compile
