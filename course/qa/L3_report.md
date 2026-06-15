# QA Report — L3: Vektor fazo modellari va semantik munosabatlar

**Date:** 2026-06-15
**Artifact:** `course/lectures/d03_vektor_embedding.tex`
**Lecture:** Official topic №3 — "Vektor fazo modellari, semantik munosabatlar va ularni vizualizatsiya qilish"
**Delivered:** Day 3 afternoon (18-iyun 2026, 11:00–12:20, A.A. Abdulali)
**Paired practice:** P3 (Day 4, 19-iyun 2026, I.R. Atadjanov — `PretrainedEmbedder`, m03)
**Continuity:** L2 (klassik tasnif) → L3 (embeddinglar); morning P2 (sentiment classifier) referenced in [B]

---

## Gate Results

| Gate | Result | Notes |
|---|---|---|
| Terminology grep (`professor\|talaba\|student\|o'qituvchi`) | **PASS** | 0 defects |
| Slide count (rendered, body only) | **PASS (exception)** | ~40 slides — follows L1/L2 precedent for 4-subitem official lectures |
| Self-contained (.tex) | **PASS** | No `\input`, `\include`, `\includegraphics`, `\bibliography`, `\setmainfont` |
| Archetype structure [A]–[S] | **PASS** | All present; 4 cycles, [M] mandatory Uzbek slide included |
| `\bunda{}` symbol key on every [G] | **PASS** | G1, G2, G3, G4 each carry `\bunda{}` (+ S1 appendix) |
| `[fragile]` on every lstlisting frame | **PASS** | 4 lstlisting frames (K2, K3, K4, L2) all `[fragile]` |
| Frame begin/end balance | **PASS** | 37 `\begin{frame}` = 37 `\end{frame}` |
| Environment balance | **PASS** | columns 17/17, tcolorbox 39/39, tikzpicture 4/4, lstlisting 4/4, align* 3/3 |
| Encoding hygiene (no U+2019 / Cyrillic / smart-quotes / BOM) | **PASS** | All apostrophes ASCII `'`; T1-safe; verified in lstlisting blocks |
| Locked [I2] traceability value | **PASS** | `cos(a,b) = 2/3 ≈ 0.667` — matches course_map.yaml hand_example exactly |
| pdflatex compile ×2 | **PASS** | MiKTeX (local) — 0 `^!` errors, 0 `Overfull \hbox (>10pt)`, 54 pages output |
| Visual PDF review (54 pages) | **PASS** | All pages rendered at 90 dpi and inspected; 4 defects found and fixed (see below) |

---

## Post-Compile Visual Review — Defects Found & Fixed

The compiled PDF (54 rendered pages incl. `\pause` overlays) was rendered to PNG
and every page inspected. Four defects were found and corrected; the deck now
compiles cleanly (0 `^!`, 0 Overfull >10pt) and all four pages re-verified visually.

| # | Frame | Defect | Root cause | Fix |
|---|---|---|---|---|
| 1 | [F1] | `"issiq ___ ichdim"` — "ichdim" rendered as a subscript, blank lost | bare `___` in text mode → LaTeX math subscript | `\underline{\hspace{0.7cm}}` blank |
| 2 | [J1] | `"yangi ___ sotib oldim"`, `"___ ni haydadi"` — same subscript corruption | bare `___` in text mode | `\underline{\hspace{0.7cm}}` blanks |
| 3 | [I4] | **Fatal compile error** — `tcb` unknown key `markazlashtirilgan)` | comma inside unbraced tcolorbox `title=...(2D, markazlashtirilgan)` parsed as key separator | wrap title in braces `title={...}` |
| 4 | [H3] | "1-qadam." ran onto the "Savol:" line instead of a new line | `\vspace` without a paragraph break | inserted blank line (paragraph break) before `\vspace` |

Note: defect #3 was latent — Overleaf's pgfkeys tolerated the stray key and still
emitted a PDF, but local MiKTeX treated it as fatal (no PDF). Bracing the title is
the correct, portable fix.

---

## Slide-Count Breakdown

| Metric | Value |
|---|---|
| `\begin{frame}` in source (incl. 1 in `\AtBeginSection` macro, 3 appendix) | 37 |
| Content frames before `\appendix` | 33 (B,C,D,E + cycles 5+7+6+5 + M,N,O,P,Q,R) |
| `\maketitle` (standalone) | +1 |
| `\section` count | 6 |
| AtBeginSection "Prezentatsiya rejasi" firings | +6 |
| **Estimated rendered slides (body)** | **40** |
| `\pause` overlays (body) | 11 |
| Appendix [S] frames (excluded) | 3 (S1, S2, S3) |

**Slide-count judgment call (for human review):** course_map.yaml does not
carry an explicit slide-count exception field for L3. However, L3 is a
4-subitem official lecture identical in scope to L1 (48 rendered) and L2 (43
rendered), both of which were approved under the documented exception for full
4-cycle topical lectures. L3 at ~40 rendered slides follows that established
precedent and is leaner than both. Flagging for explicit approval.

---

## Sub-Item Coverage (4 official sub-items)

| Official sub-item | Cycle | Slides covering it |
|---|---|---|
| 1. So'z embeddinglari: distributiv gipoteza va zich vektorlar | 1 | [F1][G1][H1][I1][J1] |
| 2. Vektorlararo o'xshashlik: kosinus o'xshashlik nazariyasi | 2 | [F2][G2][H2][I2★][J2][K2][L2] |
| 3. So'z analogiyalari va semantik munosabatlar | 3 | [F3][G3][H3][I3][J3][K3] |
| 4. PCA bilan yuqori o'lchamli embeddinglarni vizualizatsiya | 4 | [F4][G4][I4][J4][K4] |

★ [I2] carries the locked traceability number for P3's first assert.

---

## Per-Slide Archetype Coverage Table

| # | Archetype | Frame title (sentence case) | Sub-item |
|---|---|---|---|
| 1 | **[A]** | *(title — `\maketitle`)* | — |
| 2 | *(AtBegSec)* | Prezentatsiya rejasi (Kirish) | — |
| 3 | **[B]** | Takrorlash: bugun ertalab nima qurdik? | — |
| 4 | **[C]** | Siz bugungi dars oxirida quyidagilarni bajara olasiz | — |
| 5 | **[D]** | Bugungi dars rejasi | — |
| 6 | **[E]** | Muammo: TF-IDF "qiziq" va "qiziqarli"ni begona deb biladi | — |
| 7 | *(AtBegSec)* | Prezentatsiya rejasi (Embeddinglar) | — |
| 8 | **[F1]** | So'zni qanday qilib raqamga aylantiramiz? Siyrakdan zichgacha | 1 |
| 9 | **[G1]** | So'z embeddingi: rasmiy ta'rif | 1 |
| 10 | **[H1]** | Nega one-hot vektorlar ma'noni ushlay olmaydi? | 1 |
| 11 | **[I1]** | Qo'lda misol: bir xil so'zlar, ikki xil ifoda | 1 |
| 12 | **[J1]** | Sizning vazifangiz: qaysi juft yaqinroq? | 1 |
| 13 | *(AtBegSec)* | Prezentatsiya rejasi (Kosinus) | — |
| 14 | **[F2]** | Intuitsiya: o'xshashlik — vektorlar orasidagi burchak | 2 |
| 15 | **[G2]** | Kosinus o'xshashligi: rasmiy ta'rif | 2 |
| 16 | **[H2]** | Keltirib chiqarish: nega normaga bo'lamiz? | 2 |
| 17 | **[I2] ★** | Qo'lda hisob: ikki vektorning kosinus o'xshashligi (P3 assert) | 2 |
| 18 | **[J2]** | Sizning vazifangiz: kosinusni hisoblang | 2 |
| 19 | **[K2]** | Kod ↔ formula: cosine_similarity | 2 |
| 20 | **[L2]** | Kosinus tuzog'i: 1D massiv cosine_similarity ni sindiradi | 2 |
| 21 | *(AtBegSec)* | Prezentatsiya rejasi (Analogiyalar) | — |
| 22 | **[F3]** | Intuitsiya: ma'no farqi — vektorlar orasidagi yo'nalish | 3 |
| 23 | **[G3]** | So'z analogiyasi: rasmiy ta'rif | 3 |
| 24 | **[H3]** | Analogiyani qadamma-qadam yechamiz | 3 |
| 25 | **[I3]** | Qo'lda hisob: analogiyani 2D vektorlarda yechamiz | 3 |
| 26 | **[J3]** | Sizning vazifangiz: analogiyani yeching | 3 |
| 27 | **[K3]** | Kod ↔ formula: most_similar va analogiya | 3 |
| 28 | *(AtBegSec)* | Prezentatsiya rejasi (Vizualizatsiya) | — |
| 29 | **[F4]** | Intuitsiya: 300 o'lchamli fazoni qanday "ko'ramiz"? | 4 |
| 30 | **[G4]** | PCA: asosiy komponentlar tahlili (nazariy ta'rif) | 4 |
| 31 | **[I4]** | Qo'lda misol: qaysi o'q ko'proq ma'lumot saqlaydi? | 4 |
| 32 | **[J4]** | Sizning vazifangiz: PCA nimani saqlaydi, nimani yo'qotadi? | 4 |
| 33 | **[K4]** | Kod ↔ formula: PCA bilan 2D vizualizatsiya | 4 |
| 34 | *(AtBegSec)* | Prezentatsiya rejasi (Xulosa) | — |
| 35 | **[M]** | Tayyor embeddinglar o'zbek tili uchun cheklangan qamrovga ega | All |
| 36 | **[N]** | Qachon nima ishlatamiz? Siyrak va zich vektorlar taqqoslovi | 1–2 |
| 37 | **[O]** | Seminal maqola: GloVe (Pennington va b., 2014) | All |
| 38 | **[P]** | Bugungi maqsadlar bajarildi | All |
| 39 | **[Q]** | Ertaga P3: tayyor embeddinglar bilan ishlaysiz | All |
| 40 | **[R]** | Adabiyotlar | — |
| *(41)* | **[S1]** *(appendix)* | Kosinus va Evklid masofa bog'liqligi | 2 |
| *(42)* | **[S2]** *(appendix)* | Vektor, norma va skalyar ko'paytma eslatmasi | — (prereq) |
| *(43)* | **[S3]** *(appendix)* | PCA matematik asoslari | 4 |

---

## Archetype Completeness

| Archetype | Required | Present | Notes |
|---|---|---|---|
| [A] Title | yes | ✓ | `\maketitle` |
| [B] Recap quiz (2 Q + `\pause` + 2 A) | yes | ✓ | References L2 (TF-IDF input) + morning P2 + sets up meaning gap |
| [C] Measurable objectives | yes | ✓ | 4 verbs: keltirib chiqara / hisoblay / yecha / kodda bog'lay |
| [D] Plan + time-budget table | yes | ✓ | 4 cycles, 80 min |
| [E] Problem-first motivation | yes | ✓ | Reuses L1 [O] discussion Q ("qiziq vs qiziqarli") — one-hot cosine = 0 |
| [F] Intuition (×4) | yes | ✓ | F1 (sparse↔dense), F2 (angle), F3 (vector arithmetic), F4 (PCA) |
| [G] Defbox + `\bunda` (×4) | yes | ✓ | G1 embedding, G2 cosine, G3 analogy, G4 PCA — all carry `\bunda{}` |
| [H] Derivation steps | yes | ✓ | H1 (one-hot orthogonality), H2 (cosine), H3 (analogy) — cycle 4 theory-only, no H (matches L1) |
| [I] Hand example (×4) | yes | ✓ | I1, I2★ (locked 2/3), I3 (analogy→moskva), I4 (PCA variance) |
| [J] Your-turn warnbox→`\pause`→okbox (×4) | yes | ✓ | J1, J2, J3, J4 — all four cycles |
| [K] Code ↔ formula bridge | yes | ✓ | K2 cosine_similarity, K3 most_similar, K4 PCA — map to P3 tasks |
| [L] Pitfall + real error | yes | ✓ | L2 — 1D array reshape error (common with 1D embedding vectors) |
| [M] Uzbek-language (mandatory) | yes | ✓ | Pretrained embedding OOV coverage; agglutination; forward to L6/P6 |
| [N] Synthesis comparison table | yes | ✓ | Sparse (TF-IDF) vs Dense (embedding), 7-row decision table |
| [O] Seminal paper + discussion Q | yes | ✓ | GloVe (Pennington et al. 2014); discussion Q on low-resource Uzbek |
| [P] Objectives checkmarked (`\bajarildi`) | yes | ✓ | `\bajarildi` ×4 mirroring [C] |
| [Q] Bridge to practice (TikZ pipeline) | yes | ✓ | TikZ → m03 PretrainedEmbedder; P3 first assert snippet |
| [R] References | yes | ✓ | 5 references (GloVe, Word2Vec, J&M, Firth, sklearn) |
| [S] Appendix backups | yes | ✓ | S1 cosine↔Euclid, S2 vector/norm refresher, S3 PCA math |

All archetype slots present. ✓

---

## Locked Hand-Example Value ([I2]) — P3 Traceability

Corpus (verbatim from course_map.yaml `hand_example`):
- $\mathbf{a}=(1,1,1,0)$ — "nlp juda qiziq"
- $\mathbf{b}=(1,1,0,1)$ — "nlp juda foydali"
- $V$ = (nlp, juda, qiziq, foydali)

| Computation | Value |
|---|---|
| $\mathbf{a}\cdot\mathbf{b}$ | $1{+}1{+}0{+}0 = 2$ |
| $\lVert\mathbf{a}\rVert = \lVert\mathbf{b}\rVert$ | $\sqrt{3}$ |
| $\cos(\mathbf{a},\mathbf{b})$ | $2/3 \approx \mathbf{0.667}$ |

P3 first assert (Day 4): `assert abs(cos_val - 0.667) < 1e-3`
Traceability comment in .tex (line in [I2] / [Q]): `# Ma'ruza L3 [I2]-slayd`

**Additional numeric values for P3 alignment:**
- [J2] cosine your-turn: `a=(1,1,1,0,0,0)`, `c=(0,1,0,0,1,1)` → `cos = 1/3 ≈ 0.333`
- [I3] analogy: `toshkent − uzbekiston + rossiya = (7,1) = moskva`
- [J3] analogy: `qirol − erkak + ayol = (3,4) = malika`
- [I4] PCA: Var(x)≈8, Var(y)≈0.02 → PC1≈x-axis, explained_variance≈99.8%
- [J4] PCA: explained_variance_ratio [0.31, 0.18] → 49% retained

---

## Content Notes

- **GPU**: `gpu_required: false` (L3). Lecture only; no executable cells. CPU-only.
- **Continuity**: [B] recap ties to L2 classification + this-morning P2; [E] resolves L1's
  leftover [O] discussion question ("qiziq vs qiziqarli"); [Q] bridges to P3 (m03).
- **Seminal paper**: GloVe Pennington et al. (2014) — matches `seminal_paper` in course_map.yaml.
- **Uzbek angle** [M]: pretrained-embedding low coverage / OOV for Uzbek; matches `uzbek_angle`.
  Forward-references L6/P6 (training Word2Vec from scratch) per course arc.
- **Style**: follows the manually-polished L1 etalon — Uzbek sentence case, first-person plural
  ("hisoblaymiz", "ko'ramiz"), full-statement frame titles, ASCII apostrophe throughout.
  (NB: L2 still uses Title Case from before the sentence-case rule; L3 follows the L1 etalon.)
- **AtBeginSection title** uses "Prezentatsiya rejasi" (L1 etalon), not the template's "Qayerdamiz?".

---

**Status: PASS (all gates incl. compile)** — terminology, archetypes, `\bunda{}` keys,
fragile frames, environment balance, encoding hygiene, and locked [I2] traceability all clear.
Compile gate now **PASS locally** (MiKTeX, pdflatex ×2): 0 `^!` errors, 0 `Overfull \hbox (>10pt)`,
54 pages. All 54 rendered pages were visually inspected; 4 defects found and fixed (table above).
Compiled `course/lectures/d03_vektor_embedding.pdf` committed alongside the source.

**One judgment call awaiting human approval:** slide count (~40) exceeds the standard 22–28
ceiling, following the L1/L2 4-subitem-lecture exception precedent (course_map.yaml has no
explicit exception field for L3).
