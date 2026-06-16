# QA Report: Day 15 Practice (P14)

**Artifacts**: `course/practices/d15_p14_rag.ipynb` + `..._SOLUTIONS.ipynb`
**Capstone module**: `capstone/modules/m14_rag_engine.py` (REAL module — `consumed_by`: m15 agent `retrieve_docs`)
**Bundled data**: `course/practices/d15_checkpoints/uz_kb_mini.txt` (original uz knowledge base, 12 chunks: news + lex.uz-style)
**Date**: 2026-06-16
**Reviewer**: Claude Code (automated gates + local execution)
**Paired lecture**: L14 — RAG va vektor ma'lumotlar bazalari
**Next**: L15 (Day 15 lecture — AI agentlar, ReAct)

---

## Gate Summary

| Gate | Status | Detail |
|------|--------|--------|
| JSON valid — student | **PASS** | `nbformat.validate` OK; nbformat 4.5; 30 cells (14 code, 16 md); all `id` |
| JSON valid — solutions | **PASS** | `nbformat.validate` OK; nbformat 4.5; 30 cells |
| **Executes top-to-bottom (OFFLINE_FALLBACK=True, CPU)** | **PASS (local)** | SOLUTIONS exec'd on Python 3.13.14 — **14/14 asserts passed**, zero exceptions. Offline path: TF-IDF (char n-gram) + numpy cosine + extractive answer. |
| Student stub cells compile | **PASS** | All 14 code cells `compile()` clean |
| **Locked assert (RAG token budget)** | **PASS** | §4A `prompt_tokens(3,200,100) == 700`; `700/128000 ≈ 0.0055 < 1%` (pure-python) |
| Traceability comment cites lecture | **PASS** | `# Ma'ruza L14 [I2]-slayd` (§4A); retrieved_docs=3 asserted in §4C |
| Every blanked region has paired assert | **PASS** | 2 blanked cells (§4B chunk_text, §4C index/answer) → paired asserts; §4D mustaqil + structural assert |
| m14 contract conformance | **PASS** | `index(texts,batch_size)` / `answer(question,k)->dict` / `save_index` / `load_index` exact (contracts.py) |
| **answer returns dict {answer,sources,confidence}** | **PASS** | keys exactly `{answer, sources, confidence}`; answer str; sources list (len=k); confidence float ∈ [0,1] |
| **save_index/load_index test (real module)** | **PASS** | §5 `save_index → load_index → answer` verified; sources len preserved |
| No GPU / VRAM | **PASS** | CPU-only; TF-IDF + numpy; VRAM peak 0 GB |
| Data size | **PASS** | bundled KB ≈ 1.4 KB (≪ 500 MB; real uz_kb 10000 chunks online) |
| Seeds set | **PASS** | `random.seed(42)`, `np.random.seed(42)` |
| Checkpoint cell present | **PASS** | Checkpoint A (§3, reloads knowledge base) |
| Terminology grep | **PASS** | 0 matches for `professor\|talaba\|student\|o'qituvchi` (all 4 artifacts) |
| ASCII apostrophe / no U+2019 / no Cyrillic / no BOM | **PASS** | all artifacts clean; one corpus Cyrillic slip (`Amударyo`) caught & fixed pre-build |

**Overall: ALL GATES PASS**

---

## ST/FAISS/LLM present-or-absent — offline fallback verified

- `sentence-transformers` importable locally **but** the embedding model needs an internet download, and on
  this run `import sentence_transformers` even hit a transient **torch DLL init error** (WinError 1114). The
  module/notebook guards catch **any exception** (not just `ImportError`) → `HAS_ST=False` → graceful fallback.
- `faiss`/`chromadb`/LLM API absent locally.
- Forced offline path (`USE_ST=False`, `USE_LLM=False`): **TF-IDF (char n-gram) embedding + numpy cosine top-k
  + extractive answer** (top-k sources joined). Runs end-to-end, deterministic, no download.

| Path | embedding | search | answer |
|---|---|---|---|
| Kaggle (USE_ST=True, internet) | sentence-transformers (MiniLM) | FAISS IndexFlatIP | LLM API |
| offline (forced, local) | TF-IDF char n-gram (3–5) | numpy cosine | extractive (joined sources) |

**Char n-gram TF-IDF** is a deliberate fallback choice: it gives non-zero similarity for Uzbek
morphological variants ("poytaxt" vs "poytaxti" share n-grams) where word-level TF-IDF scores 0 — the very
morphology problem the course teaches. Local retrieval for "O'zbekiston poytaxti qaysi shahar?" correctly
returns the Toshkent chunk (confidence ≈ 0.40).

---

## Locked / Verified Numbers

| Slide | Quantity | Value | Role |
|-------|----------|-------|------|
| **[I2]** | RAG prompt `3×200+100`; `/128000` | **700 token; ≈0.55% < 1%**; retrieved_docs=3 | **P14 first assert** (course_map lock) |

§4A reproduces the lecture's RAG prompt-token budget — **P14's first assert** — matching course_map Day 15
paired-lecture L14 `hand_example` (`# Ma'ruza L14 [I2]-slayd`). §4C asserts `len(sources) == 3`.

---

## Notebook Structure

| Section | Cells | Type | Status |
|---------|-------|------|--------|
| §0 Header | 1 | MD | OK — objectives from L14 [C], timing, license + offline-design disclosure |
| §1 Muhit | 2 | MD+Code | OK — seeds, OFFLINE_FALLBACK, HAS_ST/FAISS, USE_ST/USE_LLM, module path |
| §2 Yaxlit natija | 2 | MD+Code | OK — load_kb + index + answer demo (answer/sources/confidence) |
| §3 PRIMM periferiya | 3 | Mixed | OK — ST+FAISS+LLM (Kaggle, commented) + char n-gram demo; Bashorat/Tekshiring/O'zgartiring |
| Checkpoint A | 2 | MD+Code | OK |
| §4 core | 9 | Mixed | OK — Namuna (4A locked token budget) + 4B chunk_text + 4C index/answer + 4D mustaqil, each + assert |
| §5 Loyihaga ulash | 3 | MD+Code+MD | OK — m14 contract test, save_index/load_index test, git |
| §6 Tadqiqot + yakun | 3 | MD+Code+MD | OK — k sweep (1/3/5); chunking/embedding reflection; exit ticket |

Total: 30 cells (14 code, 16 markdown). Blanked core cells: 2 (§4B chunk_text, §4C index/answer), each paired with an assert.

---

## ST/FAISS/LLM-Optional Design

m14 branches on `USE_ST` / `HAS_FAISS` / `USE_LLM`:
- **Kaggle path**: `SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")` embeddings + `faiss.IndexFlatIP`
  + LLM API in `_llm_call`.
- **Offline path** (forced): `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5))` + numpy cosine top-k +
  extractive answer (`" ".join(sources)`); confidence = top cosine similarity.
- Import guards catch any exception (torch DLL failures included) → robust degradation.

Result: runs end-to-end with or without sentence-transformers/faiss/LLM/GPU, CPU-only, no download.

---

## Module Conformance (contracts.py)

```
m14 RAGEngine (REAL module, consumed_by: m15 (P15, retrieve_docs)):       provides:
  index(texts: list[str], batch_size=32) -> None                           ✓
  answer(question: str, k=3) -> dict[str, Any]                             ✓  {"answer","sources","confidence"}
  save_index(path: str) -> None  /  load_index(path: str) -> None          ✓  (pickle: docs+emb+vectorizer)
```
Real pipeline module: save_index/load_index present and wired into m15 (`retrieve_docs`). Note the method
names are `save_index`/`load_index` (not `save`/`load`).

---

## Deviation from course_map.yaml

course_map Day 15 `corpus_subset: uz_kb` (news + lex.uz, 10000 chunks, FAISS index, online). The
**OFFLINE_FALLBACK** uses a small **original** uz knowledge base (`uz_kb_mini.txt`, 12 chunks). Local run is
**CPU-only** with TF-IDF char n-gram embeddings + numpy cosine (FAISS Kaggle-only) and an **extractive**
answer (no LLM API locally). `RecursiveCharacterTextSplitter` (langchain) shown as Kaggle reference; a simple
char-window `chunk_text` taught/asserted locally. Answer quality is demo-grade, stated honestly.

---

## Pending

- Full Kaggle kernel run with real uz_kb (10000 chunks) on GPU (sentence-transformers + FAISS + LLM API) —
  confirmed when notebooks are published as a Kaggle Dataset (Day 15, 9-iyul-2026).
- **L15** (Day 15 lecture — AI agentlar, ReAct) is the next chronological artifact.
- m14 will be consumed by m15 (agent `retrieve_docs` tool, P15).
