# QA Report: 4-hafta milestone (M4 / w4) — COURSE FINALE

**Artifacts**:
- `course/milestones/w4_milestone.md` (brief) + `course/milestones/w4_check.py` (self-check)
- **Task A**: `capstone/app.py` (SentimentAPI, FastAPI) + `course/practices/d16_p16_fastapi.ipynb` (+ `_SOLUTIONS`)
- **Task B**: `course/final_test.docx` (30 questions, L1–L14) + `course/final_test.xlsx` (answer key)
- **Task C**: `capstone/modules/m15_langchain_agent.py` (already built in P15 — verified here)
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local execution)
**Type**: milestone (Wednesday M4, flipped) — the final course artifact

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| **w4_check.py runs locally** | **PASS** | **14/14 checks passed**, 0 exceptions (3 tasks) on Python 3.13.14 |
| **Task A — TestClient POST /predict** | **PASS** | `TestClient(app).post("/predict", json={"text": ...})` → 200; `{"sentiment": ijobiy/salbiy, "confidence": float∈[0,1]}` (e.g. ijobiy, 0.5745) |
| app.py contract | **PASS** | `create_sentiment_api() -> FastAPI`; module-level `app`; `POST /predict {text}→{sentiment, confidence}`; `GET /` |
| **Locked labels** | **PASS** | `/predict` returns `ijobiy`/`salbiy` (L2 [I2]); JSON shape matches L15 [I3]; 0 `musbat`/`manfiy` |
| **Task B — final_test.docx** | **PASS** | python-docx opens; **30 numbered questions**, L1–L14 coverage (MCQ + short answer) |
| **Task B — final_test.xlsx** | **PASS** | openpyxl opens; answer key 31 rows (header + 30: #, javob, ma'ruza) |
| **Task C — m15 agent** | **PASS** | `import m15` OK; `DocumentAssistantAgent().run()` → str; `route()` L15 [I1] (sentiment_classify, conf>0.7) |
| P16 notebook JSON valid | **PASS** | nbformat 4.5; student + SOLUTIONS (identical — fully_worked); all `id` |
| P16 executes (TestClient, local) | **PASS** | all code cells exec'd top-to-bottom; `/predict` assert passes |
| P16 fully_worked (no blanks) | **PASS** | per course_map `notebook_style: fully_worked_primm` — no faded scaffold; PRIMM + self-check assert |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi`; 0 `musbat\|manfiy` (all committed artifacts) |
| ASCII apostrophe / no U+2019 / no Cyrillic / no BOM | **PASS** | all text artifacts + builder source clean; docx/xlsx generated from Cyrillic-free source |
| No GPU / Docker build | **PASS** | CPU-only; m13 LogReg offline; Dockerfile shown (not built — Docker optional) |

**Overall: ALL GATES PASS — COURSE COMPLETE** 🎓

---

## Three Tasks (course_map M4)

### Task A — Deploy (P16): SentimentAPI
- `capstone/app.py`: FastAPI app loads **m13 FineTunedClassifier** once at startup (`USE_TRANSFORMERS=False`
  → TF-IDF + LogReg, offline), fit on a small inline balanced corpus. `POST /predict {text}` →
  `{sentiment, confidence}`; `confidence = max(predict_proba)`.
- `d16_p16_fastapi.ipynb`: **fully_worked** PRIMM (no graded blanks) — env, TestClient demo, full app.py walkthrough,
  Dockerfile, contract self-check assert, capstone-defense close.
- Verified locally via `fastapi.testclient.TestClient` (httpx present). Docker shown as reference (not built).

### Task B — Knowledge test (L1–L14)
- `final_test.docx` (python-docx): 30 questions across all 14 pre-Day-15 lectures (BoW/TF-IDF, NB, embedding/cosine,
  edit distance, n-gram/Viterbi, Word2Vec, RNN, LSTM/GRU, generation/temperature, NER/IOB2, seq2seq/attention/BLEU,
  Transformer/ROUGE, transfer/BERT/BCE, RAG). MCQ + short answer; many anchored to the course's locked
  hand-examples (IDF=0.405, cosine=2/3, σ(1.5)=0.818, α[3]=0.665, ROUGE F1=0.750, BCE σ(2.0)=0.880, RAG=700, ...).
- `final_test.xlsx` (openpyxl): answer key — # / correct answer / lecture.
- Agent (L15) and MLOps (L16) excluded per `scope: L1–L14`.

### Task C — Agent scaffold (already done)
- course_map M4 Task C asks for an m15 **scaffold**; in this project **m15 was fully built in P15**
  (`m15_langchain_agent.py`, ReAct router + 5 tools). w4 verifies it (import + `run()` + locked `route()`),
  does not rebuild.

---

## Continuity / Locked Threads

- `POST /predict` JSON `{sentiment, confidence}` reproduces **L15 [I3]** (model-as-API) and the project-wide
  **locked sentiment labels `ijobiy`/`salbiy`** (L2 [I2]). `w4_check.py` asserts both.
- The knowledge-test answers re-use the locked hand-examples from L1–L14, reinforcing the traceability chain
  that ran through every lecture/practice pair.

---

## Side change

`capstone/modules/m13_bert_classifier.py`: the `import transformers` guard was widened from `except ImportError`
to `except Exception` (matches m14/m15-deps) so `app.py` imports robustly even on a transient torch DLL load
error. Offline (LogReg) behaviour unchanged; m13's own tests still pass.

---

## Deviation from course_map.yaml

None material. M4 = three async tasks (A/B/C); all produced/verified. Task C reframed from "scaffold" to
"verify existing m15" (built in P15). P16 notebook filed under `course/practices/d16_p16_fastapi.ipynb`
(M4 has no day number; Day-16 prefix used since M4 feeds Day 16). Docker image is shown, not built locally
(kaggle-hardware: Docker optional). app.py serves m13 via offline LogReg (no internet/GPU); production would
`load()` a fine-tuned BERT. final_test stored at `course/final_test.{docx,xlsx}` per `artifact_formats`.

---

## Course completion status

- **Lectures L1–L16**: complete ✅
- **Practices P1–P16** (m01–m15 + app.py): complete ✅
- **Milestones w1–w4**: complete ✅
- **Capstone "O'zbek hujjat yordamchisi"**: complete — m15 agent + SentimentAPI, ready for defense ✅

**The 4-week NLP course material production is fully complete.** 🎓
