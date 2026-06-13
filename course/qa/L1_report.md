# QA Report — L1: NLP Asoslari va Matnni Qayta Ishlash

**Date:** 2026-06-13  
**Artifact:** `course/lectures/d01_nlp_asoslari.tex`  
**Lecture:** Official topic №1 — "NLP asoslari va matnni qayta ishlash texnologiyalari"  
**Delivered:** Day 1 afternoon (15-iyun 2026, 11:00–12:20, A.A. Abdulali)  
**Paired practice:** P1 (Day 2, 16-iyun 2026, I.R. Atadjanov)

---

## Gate Results

| Gate | Result | Notes |
|---|---|---|
| Terminology grep | **PASS** | 0 defects — professor/talaba/student/o'qituvchi not found |
| Slide count (rendered, body only) | **PASS** | ~41 slides (target 38-42; L1 approved exception) |
| Self-contained (.tex) | **PASS** | No `\input`, `\include`, `\includegraphics`, `\bibliography`, `\setmainfont` |
| Archetype structure (all 24 checks) | **PASS** | All archetypes present; see coverage table below |
| pdflatex compile | **PASS** | MiKTeX 25.12 local; zero `^!` errors; 2.83pt overfull only (< 10pt) |

---

## Slide-Count Breakdown

| Metric | Value |
|---|---|
| `\begin{frame}` in body (before `\appendix`) | 35 |
| `\maketitle` (standalone, not in frame) | +1 |
| `\section` count | 6 |
| AtBeginSection extra firings (n_sections − 1) | +5 |
| **Estimated rendered slides (body)** | **41** |
| Appendix [S] frames (excluded) | 3 |

---

## Overleaf Upload

Upload **one file only**: `course/lectures/d01_nlp_asoslari.tex`

No external assets. All packages are standard CTAN (inputenc, fontenc, lmodern,
amsmath, amssymb, booktabs, xcolor, listings, tcolorbox[skins], tikz with
arrows.meta + positioning). Beamer Boadilla theme is built-in on Overleaf.

**Post-compile verification:**
```
pdflatex twice → check log for:
  ^!                           # must be empty
  Overfull \hbox (>10pt)       # must be empty
```

---

## Sub-Item Coverage (4 official sub-items)

| Official sub-item | Slides covering it | Cycles |
|---|---|---|
| 1. NLPga kirish, tarixi, sentiment tahlili | 8–12 (Section 2) | [F1][G1][H1][I1][J1] |
| 2. Tokenizatsiya, normalizatsiya, stop-words, stemming, lemmatizatsiya | 14–21 (Section 3) | [F2][G2][H2a][H2b][I2][J2][K2][L2] |
| 3. BoW va TF-IDF | 23–29 (Section 4) | [F3][G3][H3a][H3b][I3][J3][K3] |
| 4. NLPning asosiy vazifalari va ilovalari | 31–34 (Section 5) | [F4][G4][I4][J4] |

---

## Per-Slide Archetype Coverage Table

Rendered slide sequence (estimated). AtBeginSection "Qayerdamiz?" slides noted separately.

| # | Archetype | Frame Title | Sub-item |
|---|---|---|---|
| 1 | **[A]** | *(title — `\maketitle`)* | — |
| 2 | *(AtBegSec)* | Qayerdamiz? (Section 1 ToC) | — |
| 3 | **[B]** | 16 Kunda Siz Ishlaydigan NLP Tizimi Qurasiz | — |
| 4 | **[C]** | Bugungi Dars Oxirida Siz Nima Qila Olasiz? | — |
| 5 | **[D]** | Dars Rejasi: To'rt Tsikl, 80 Daqiqa | — |
| 6 | **[E]** | Muammo: "Yaxshi" Kalit So'zi Salbiy Sharhni Musbat Deb Belgilaydi | — |
| 7 | *(AtBegSec)* | Qayerdamiz? (Section 2 ToC) | — |
| 8 | **[F1]** | NLP Hamma Joyda: Siz Ham Undan Foydalanasiz | Sub-item 1 |
| 9 | **[G1]** | NLP — Tabiiy Tilni Qayta Ishlash: Rasmiy Ta'rif | Sub-item 1 |
| 10 | **[H1]** | NLP Tarixi: Qoidalardan Neyron Tarmoqlargacha | Sub-item 1 |
| 11 | **[I1]** | Sentiment Tahlili: Sharh Musbatmi Yoki Salbiy? | Sub-item 1 |
| 12 | **[J1]** | Sizning Vazifangiz: Sentiment Taxmin Qiling | Sub-item 1 |
| 13 | *(AtBegSec)* | Qayerdamiz? (Section 3 ToC) | — |
| 14 | **[F2]** | Xom Matn Shovqin To'la — Tozalash Shart | Sub-item 2 |
| 15 | **[G2]** | Preprocessing Pipeline: Besh Bosqich | Sub-item 2 |
| 16 | **[H2a]** | Tokenizatsiya: O'zbek Apostrofida Yashirin Muammo | Sub-item 2 |
| 17 | **[H2b]** | Stop-Words Filtrlash va Stemming: Muhim So'zlarni Qoldiring | Sub-item 2 |
| 18 | **[I2]** | Qo'lda Misol: Besh Bosqichdan O'tish | Sub-item 2 |
| 19 | **[J2]** | Sizning Vazifangiz: Matnni Qo'lda Tozalang | Sub-item 2 |
| 20 | **[K2]** | Kod ↔ Pipeline: Uzbek Regex Tokenizer | Sub-item 2 |
| 21 | **[L2]** | Tuzog': Apostrof Birligi Tokenizer'ni Sindiradi | Sub-item 2 |
| 22 | *(AtBegSec)* | Qayerdamiz? (Section 4 ToC) | — |
| 23 | **[F3]** | BoW: Har Bir Hujjat So'zlar To'plami Sifatida | Sub-item 3 |
| 24 | **[G3]** | So'z Qopi (Bag of Words) — Rasmiy Ta'rif | Sub-item 3 |
| 25 | **[H3a]** | TF: Hujjatdagi So'z Chastotasi (Term Frequency) | Sub-item 3 |
| 26 | **[H3b]** | IDF: Noyob So'zlar Mukofotlanadi (Inverse Document Frequency) | Sub-item 3 |
| 27 | **[I3] ★** | Qo'lda Hisob: 3 Hujjat, TF-IDF Matritsasi (P1 Assert Raqamlari) | Sub-item 3 |
| 28 | **[J3]** | Sizning Vazifangiz: TF-IDF(python, D2) Hisoblang | Sub-item 3 |
| 29 | **[K3]** | Kod ↔ Formula: TfidfVectorizer va Qo'lda Hisob | Sub-item 3 |
| 30 | *(AtBegSec)* | Qayerdamiz? (Section 5 ToC) | — |
| 31 | **[F4]** | NLP Nima Qila Oladi? Vazifalar Xaritasi | Sub-item 4 |
| 32 | **[G4]** | Asosiy NLP Vazifalari va O'zbek Ilovalari | Sub-item 4 |
| 33 | **[I4]** | To'liq Pipeline: Sharhdan Sentiment Ballgacha | Sub-item 4 |
| 34 | **[J4]** | Sizning Vazifangiz: Ilovani NLP Vazifasiga Moslang | Sub-item 4 |
| 35 | *(AtBegSec)* | Qayerdamiz? (Section 6 ToC) | — |
| 36 | **[M]** | O'zbek Tili: Agglutinatsiya TF-IDF Ni Chalg'itadi | All |
| 37 | **[N]** | Qachon Nima Ishlatish? BoW va TF-IDF Taqqoslov Jadvali | Sub-items 2–3 |
| 38 | **[O]** | Salton (1975): 50 Yillik Poydevor | Sub-item 3 |
| 39 | **[P]** | Bugungi Maqsadlar Bajarildi | All |
| 40 | **[Q]** | Ertaga P1: Preprocessing Pipeline Qurasiz | All |
| 41 | **[R]** | Adabiyotlar | — |
| *(42)* | **[S1]** *(appendix)* | S1: Logarifm va IDF Hisoblash | Sub-item 3 |
| *(43)* | **[S2]** *(appendix)* | S2: BoW Matritsasi Qo'lda Qurilishi — Algoritm | Sub-item 3 |
| *(44)* | **[S3]** *(appendix)* | S3: Stemming Algoritmlari va O'zbek Uchun Moslash | Sub-item 2 |

★ Slide 27 [I3] carries the traceability numbers that P1 asserts check:
- `TF-IDF('nlp', D1) = 0.405` ← `assert abs(nlp_score - 0.405) < 1e-3`
- `TF-IDF('qiziq', D1) = 1.099` ← `assert abs(qiziq_score - 1.099) < 1e-3`
- Corpus (verbatim from course_map.yaml `hand_example`): D1="nlp qiziq", D2="python foydali", D3="nlp foydali"
- Comment in .tex: `% P1 (2-kun) birinchi assert bilan solishtiring`

---

## Archetype Completeness

| Archetype | Required | Present | Notes |
|---|---|---|---|
| [A] Title | yes | ✓ | `\maketitle` |
| [B] Hook / Recap | yes | ✓ | Course through-line (first lecture, no recap) |
| [C] Objectives | yes | ✓ | 4 measurable objectives with verbs |
| [D] Plan + time-budget | yes | ✓ | 4-cycle table, 80 min |
| [E] Problem-first motivation | yes | ✓ | Naive keyword matching fails on negation |
| [F] Intuition (×4) | yes | ✓ | [F1][F2][F3][F4] — each cycle has one |
| [G] Defbox + `\bunda` | yes | ✓ | [G1][G2][G3][G4] — each cycle has one |
| [H] Derivation steps | yes | ✓ | [H1] history; [H2a][H2b]; [H3a][H3b] TF+IDF |
| [I] Hand example | yes | ✓ | [I1][I2][I3][I4] — [I3] carries P1 assert numbers |
| [J] Your-turn warnbox→okbox | yes | ✓ | [J1][J2][J3][J4] — all four cycles |
| [K] Code ↔ formula bridge | yes | ✓ | [K2] preprocessing; [K3] TfidfVectorizer |
| [L] Pitfall slide | yes | ✓ | [L2] apostrof birligi muammosi |
| [M] Uzbek language (mandatory) | yes | ✓ | Agglutinatsiya + TF-IDF vocabulary explosion |
| [N] Synthesis comparison table | yes | ✓ | BoW vs TF-IDF 7-row table |
| [O] Seminal paper + discussion Q | yes | ✓ | Salton (1975); discussion Q on semantic similarity |
| [P] Objectives checkmarked | yes | ✓ | `\bajarildi` ×4 |
| [Q] Bridge to practice | yes | ✓ | TikZ pipeline + P1 assert snippet |
| [R] References | yes | ✓ | 5 references |
| [S] Appendix backups | yes | ✓ | Math refresher, BoW algorithm, stemming |

All 19 archetype slots present. ✓

---

## Content Notes

- **GPU**: `gpu_required: false` (L1). All code targets CPU (sklearn, regex). No GPU cells.
- **Uzbek apostrophe**: Both U+2019 and ASCII `'` handled in regex — taught explicitly in [H2a][L2].
- **Traceability**: [I3] comment and footer text link to P1 assert values verbatim from course_map.yaml.
- **Seminal paper**: Salton et al. (1975) — matches `seminal_paper` field in course_map.yaml.
- **Uzbek angle**: Agglutinatsiya + TF-IDF vocabulary explosion — matches `uzbek_angle` field in course_map.yaml.
- **Discussion question** [O]: "TF-IDF 'qiziq' va 'qiziqarli'ni bir xil ma'no deb ko'ra oladimi?" — carries over to P1.

---

**Status: PASS** — all gates clear including compile.  
MiKTeX 25.12 local; `pdflatex ×2`; zero errors; 48 PDF pages (41 unique slides + 4 pause overlays + 3 appendix); max overfull 2.83pt (< 10pt gate).

**Fixes applied post-initial-QA** (all from lstlisting + encoding issues found during MiKTeX compile):
- `[fragile]` added to 4 frames: H2a, K2, K3, S3 (every frame with lstlisting)
- U+2019 / U+2018 curly apostrophes removed from lstlisting content (T1 encoding incompatible)
- Cyrillic `ни` → Latin `ni` at line 873
- TikZ node styles: `text centered` → `align=center` (required for `\\` line-breaks in nodes)
- UTF-8 BOM removed from file start
- I2 table: `\footnotesize` → `\tiny` for long token strings (overfull fix)
- K3 tabular: `p{2.3cm}` → `p{2.7cm}` + `\footnotesize` (overfull fix)
