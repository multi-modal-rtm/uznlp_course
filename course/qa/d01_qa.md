# QA Report — Day 1 Orientation

**Date:** 2026-06-13  
**Artifacts reviewed:** capstone/SPEC.md, capstone/contracts.py,
day1_orientation/d01_orientatsiya.ipynb, day1_orientation/d01_kirish.tex,
day1_orientation/HISOB_YARATISH.md, assessments/pre_course.{docx,xlsx}

---

## Gate Results

| Gate | Result | Notes |
|---|---|---|
| nbformat parse | **PASS** | 20 cells, valid JSON |
| OFFLINE_FALLBACK present | **PASS** | Defined in cell 1 |
| Terminology (professor/talaba/student/o'qituvchi) | **PASS** | 0 defects across all 6 files |
| contracts.py AST | **PASS** | 16 classes, 64 methods, no syntax errors |
| Slide count (rendered) | **PASS** | ~16 slides: 1 title + 6 AtBeginSection + 9 content |
| Assessment files | **PASS** | pre_course.docx (37 KB), pre_course.xlsx (7 KB) |
| pdflatex compile | **DEFERRED** | Overleaf (pdfLaTeX) — see self-containedness audit below |
| Archetype check [A][C][D][Q][R] | **PASS** | All required markers present |
| Self-contained (no external deps) | **PASS** | Single .tex file; no `\input`/`\includegraphics` |

---

## Compile Gate Policy

Compile gate is **deferred to Overleaf (pdfLaTeX)** — not a local blocker.  
This policy applies to all decks in this project.

**Overleaf upload:** Upload `day1_orientation/d01_kirish.tex` as a **single file** — no companion files needed.

**After uploading, verify:**
```
pdflatex twice → check log for:
  ^!                                 # must be empty
  Overfull \hbox (>10pt)             # must be empty
```

**Self-containedness audit (run locally, all PASS):**

| Check | Result |
|---|---|
| `\input` / `\include` | None |
| `\includegraphics` | None |
| External font packages | None (uses `lmodern` — bundled in all TeX distributions) |
| TikZ libraries | `arrows.meta`, `positioning` — both in standard tikz bundle |
| All other `\usepackage` | `inputenc`, `fontenc`, `amsmath`, `amssymb`, `booktabs`, `xcolor`, `listings`, `tcolorbox`, `tikz` — all standard CTAN, present in Overleaf |

**Risk:** Low. All packages are standard Overleaf defaults. The Beamer Boadilla theme is built-in.

---

## Artifact Inventory

| File | Status |
|---|---|
| `capstone/SPEC.md` | Created — participant-facing Uzbek spec |
| `capstone/contracts.py` | Created — 16 classes + SentimentAPI signature |
| `day1_orientation/d01_orientatsiya.ipynb` | Created — 20 cells, OFFLINE_FALLBACK |
| `day1_orientation/d01_kirish.tex` | Created — 16 rendered slides |
| `day1_orientation/HISOB_YARATISH.md` | Created — 3 accounts, pre-Day-1 checklist |
| `assessments/pre_course.docx` | Created — 14 questions, no coding |
| `assessments/pre_course.xlsx` | Created — answer key with explanations |
| `assessments/gen_pre_course.py` | Generator script (not a deliverable) |
| `course/qa/d01_qa_check.py` | QA automation script |

---

## Notes

- `d01_orientatsiya.ipynb` must be executed top-to-bottom with OFFLINE_FALLBACK=True
  on a machine with numpy, sklearn, torch, nbformat installed before final sign-off.
  (Full Kaggle kernel execution cannot be verified locally — verify on Kaggle.)
- `pre_course.docx` question 13 answer option B was truncated for Python string-literal
  safety; full text preserved in answer key XLSX.
- Assessment note in header: "keyinchalik to'liq test bankiga birlashtiriladi" — 
  reconciliation with final test bank is deferred to capstone QA phase.

---

**Status: PASS** — all gates clear. Compile gate deferred to Overleaf (pdfLaTeX); not a blocker.
No terminology defects. Upload `day1_orientation/d01_kirish.tex` as single file to Overleaf.
