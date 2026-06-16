# NLP KURSI — LOYIHA HOLAT HISOBOTI

**Sana:** 2026-06-16 · **Branch:** `feat/help_to_improve` · **Oxirgi commit:** `bebcc12`
**Remotelar:** `origin` (datascientistn1/uznlp_course), `rtm` (multi-modal-rtm/uznlp_course).
> Holat: ikkala remote ham **to'liq sinxron** (push qilingan, 0 ortda).

> Bu hisobot mustaqil: uni boshqa Claude (yoki hamkasb) ga berib, **P14 (14-amaliyot:
> m14 RAGEngine — sodda RAG tizimi)** uchun bosqichma-bosqich prompt olish mumkin.

---

## 1. Loyiha nima

O'zbek tilidagi (lotin yozuvi) 4 haftalik intensiv NLP kursining BARCHA o'quv
materiallarini ishlab chiqarish loyihasi (15-iyun – 10-iyul 2026, 16 o'quv kuni +
4 chorshanba milestone). Har bir tinglovchi yagona kapstone loyiha — **"O'zbek
Hujjat Yordamchisi"** — quradi. Kaggle bepul rejim.

## 2. Ish intizomi (qat'iy)

- **Bir kunda bitta:** artefakt → barcha sifat darvozalari → QA → TO'XTA → inson tasdig'i → keyingi.
- **`course/course_map.yaml` — yagona haqiqat manbasi.** Mavzu, maqsadlar, hand_example faqat shundan.
- **`.claude/skills/`** majburiy: `practice-notebook`, `lecture-beamer`, `uzbek-course-style`, `kaggle-hardware`.
- **Kun juftlash:** L(N) ma'ruza P(N) amaliyotiga tayyorlaydi. Har P ning birinchi
  asserti mos L ning [I] hand_example natijasini tekshiradi (kuzatiluvchanlik).
- **Mahalliy vositalar:** Python 3.13 (numpy/sklearn/matplotlib/**torch 2.10 CPU**/**transformers** /
  **sentence-transformers** bor). ⚠️ **faiss YO'Q, chromadb YO'Q, langchain YO'Q, LLM API YO'Q**, datasets YO'Q.
- **Amaliyot uslubi:** `_build_pN.py` builder — offline data, modulni yozish, student+SOLUTIONS
  notebook'larni JSON qurish, SOLUTIONS kataklarini exec qilib assert tekshirish. Builder commit qilinmaydi.

## 3. BAJARILGAN ISHLAR

**Ma'ruzalar:** L1–L14 — …, d13_transfer_learning, **d14_rag** (tex+pdf; d02 faqat tex). Har biriga QA. ✅

**Amaliyotlar:** P1 (m01) … P13 (m13). Har biri +SOLUTIONS +checkpoints +QA, mahalliy bajarilgan. ✅

**Modullar:** m01–m13 (+ m05b). ✅  (m10/m12/m13 — HAQIQIY modullar, save/load).

**Milestonelar:** w1 (m01+m02 ✅), w2 (m01–m05 ✅), w3 (neyron m01–m08, 18/18 ✅).

> Holat: ma'ruzalar **L14 gacha**, amaliyotlar P13 gacha, milestonelar w3 gacha. **Keyingi
> xronologik artefakt — P14** (Day 15 ertalab; L14 ga juft). So'ng L15 (Day 15 ma'ruza — AI agentlar, ReAct).

## 4. KEYINGI QADAM — P14 (so'ralgan)

> P14 = **course_map Day 15, `practice_official_no: 14`**; **L14** (RAG) ma'ruzasiga juft.
> Modul: **m14 RAGEngine**, fayl `capstone/modules/m14_rag_engine.py`.
> Notebook: `course/practices/d15_p14_rag.ipynb` (+ `_SOLUTIONS` + `d15_checkpoints/`).

**P14 spetsifikatsiyasi (course_map Day 15):**
- **Mavzu:** "Sodda RAG tizimini qurish."
- **4 kichik bo'lim (practice_subitems):**
  1. Bir nechta hujjatdan iborat bilimlar bazasini yaratish.
  2. FAISS yoki ChromaDB vektor DB ni o'rnatish va hujjatlarni embedding qilib indekslash.
  3. Foydalanuvchi so'roviga eng mos hujjat qismini vektor qidiruvi orqali topish.
  4. Topilgan kontekstni LLM so'roviga qo'shib, javob generatsiya qilish.
- **Periferiya (to'liq beriladi — PRIMM):**
  - `sentence-transformers` (paraphrase-multilingual-MiniLM-L12-v2) yuklash.
  - FAISS `IndexFlatIP` sozlash + LLM API ulanish.
- **Yadro (tinglovchi yozadi):**
  - Hujjat chunking (`RecursiveCharacterTextSplitter`) va batched embedding.
  - FAISS index qurish, saqlash, yuklash.
  - `retrieve()` + `format_prompt()` + `llm_call()` pipeline.
- **corpus_subset:** uz_kb (yangiliklar + lex.uz, 10000 chunk, FAISS). **OFFLINE_FALLBACK:** `d15_checkpoints/`
  da kichik **original** uz bilim bazasi (bir necha hujjat/chunk).
- **gpu_required:** Kaggle GPU tezlatadi; mahalliy CPU'da fallback bilan ishlaydi.

**QULFLANGAN birinchi assert (L14 [I2] → P14).** Notebook RAG prompt token byudjetini ochib ko'rsatsin
(toza-python) va assert aynan shuni tekshirsin:
- $k=3$ chunk × $T_{\text{chunk}}=200$ + $T_{\text{instr}}=100$ = **700** token; $700/128000<1\%$.
- `assert prompt_tokens(3, 200, 100) == 700  # Ma'ruza L14 [I2]-slayd bilan solishtiring`
- m14 `answer(savol, k=3)` natijasida `assert len(result["sources"]) == 3` (retrieved_docs = 3).

**m14 shartnomasi (capstone/contracts.py — QAT'IY, AYNAN MOS):**
```
class RAGEngine:                               # ⚠️ HAQIQIY modul — consumed_by: m15 (P15, retrieve_docs)
    index(texts: list[str], batch_size=32) -> None
    answer(question: str, k=3) -> dict[str, Any]   # {"answer": str, "sources": list[str], "confidence": float}
    save_index(path: str) -> None    /    load_index(path: str) -> None
```
> ⚠️ DIQQAT: m14 **HAQIQIY** modul (m10/m12/m13 kabi). save_index/load_index BOR. `answer` **dict** qaytaradi
> (`answer`/`sources`/`confidence`). Offline javob ekstraktiv (haqiqiy LLM yo'q) — strukturaviy assert
> (dict 3 kalit; sources len=k; confidence ∈ [0,1]). Aniq/sifatli javob EMAS (kichik data, demo, halol).

## 5. ⚠️ sentence-transformers / faiss / LLM-IXTIYORIY DIZAYN (m12/m13 majburlash naqshi)

> ⚠️ MUHIM: sentence-transformers MAHALLIY BOR, lekin model yuklab olish (internet) talab qiladi;
> **faiss/chromadb/LLM API mahalliy YO'Q**. Shuning uchun **mahalliy tekshirish FALLBACK orqali**.
- **Kaggle yo'li (GPU+internet):** `sentence-transformers` embedding + **FAISS `IndexFlatIP`** + **LLM API**
  bilan javob generatsiya.
- **Offline yo'l (OFFLINE_FALLBACK / kutubxonasiz):**
  - embedding → **TF-IDF** (sklearn) vektorlari (yuklab olishsiz, deterministik);
  - vektor qidiruv → **numpy kosinus** top-k (FAISS o'rniga);
  - LLM javob → **ekstraktiv** (top-k sources ni birlashtirib qaytaradi; haqiqiy LLM API yo'q).
- **`USE_ST` / `HAS_FAISS` / `USE_LLM` bayroqlari** yo'lni tanlaydi; builder mahalliy hammasini False ga
  majburlaydi (m12/m13 naqshi) — internetsiz, tez, deterministik.
- **locked token byudjeti asserti HAR DOIM toza-python** (path-independent).
- Notebook GPU'siz/faiss'siz/LLM'siz uchdan-uchgacha ishlasin; haqiqiy stack kodi KO'RSATILADI,
  mahalliyda TF-IDF + numpy + ekstraktiv BAJARILADI. Javob sifati demo-darajada, halol.
- OFFLINE korpusi: `d15_checkpoints/` da kichik original uz bilim bazasi (yangilik/hujjat chunk'lari).

## 6. PRACTICE-NOTEBOOK TUZILISHI (skill, gold standard P13/P12 naqshi)

§1 Muhit (seeds, OFFLINE_FALLBACK, HAS_ST, HAS_FAISS, USE_LLM) → §2 yaxlit natija (tayyor answer demo:
savol → javob + sources + confidence) → §3 PRIMM periferiya (sentence-transformers + FAISS IndexFlatIP +
LLM API ulanish — to'liq beriladi, guard bilan; offline = TF-IDF) → Checkpoint → §4 yadro: **so'nuvchi tayanch**
(Namuna: locked RAG token byudjeti [700, <1%] toza-python → Birgalikda `# === SIZNING KODINGIZ ===` chunking +
embedding + index qurish + retrieve/format_prompt → Mustaqil: answer() + sources/confidence baholash),
har blank → mos **assert** → §5 loyihaga ulash (m14 ni yozish, import test, **save_index/load_index test**, git)
→ §6 tadqiqot + exit ticket (chunking ta'siri; k tanlash; o'zbek embedding sifati).

## 7. SIFAT DARVOZALARI (MAHALLIY — kechiktirilmaydi)

- **JSON valid** (student + SOLUTIONS): nbformat 4.5; har katak `id`.
- **Uchdan-uchgacha bajariladi** (`OFFLINE_FALLBACK=True`, CPU): SOLUTIONS kataklari mahalliy ishga tushadi,
  **har assert o'tadi**, 0 istisno. Builder orqali exec. Mahalliy = TF-IDF + numpy kosinus + ekstraktiv javob.
- **Student stub kataklar `compile()` toza.**
- **Birinchi (qulflangan) assert:** `prompt_tokens(3,200,100) == 700` — `# Ma'ruza L14 [I2]-slayd`
  (+ `answer(...,k=3)` → `len(sources) == 3`).
- **Har blank region mos assert bilan;** m14 shartnoma mosligi (index/answer/save_index/load_index).
- **answer strukturaviy assert:** dict kalitlari aynan `{answer, sources, confidence}`; `answer` str;
  `sources` list (len = k); `confidence` float ∈ [0,1]. (Aniq/sifatli javob EMAS — kichik data, demo, halol.)
- **save_index/load_index tekshiruvi:** m14 consumed_by m15 — save→load→answer ishlashini sinab ko'r.
- **No GPU** mahalliy; seeds (random/np 42); checkpoint katak(lar)i.
- **Terminologiya grep toza;** ASCII apostrof; U+2019 yo'q; **Kirill 0** (notebook'da ham skan); BOM yo'q.

## 8. SO'RALADIGAN PROMPT

**P14 (14-amaliyot: m14 RAGEngine — sodda RAG tizimi) ni ishlab chiqarish uchun bosqichma-bosqich prompt**
yozib bering —
- course_map Day 15 (practice 14) spetsifikatsiyasiga aniq mos: 4 subitem, periferiya (sentence-transformers +
  FAISS IndexFlatIP + LLM API ulanish), yadro (chunking RecursiveCharacterTextSplitter + batched embedding,
  FAISS build/save/load, retrieve+format_prompt+llm_call pipeline);
- **qulflangan birinchi assert** = L14 [I2]: RAG prompt `3×200+100 = 700` token, `700/128000<1%`, retrieved_docs=3
  (toza-python, `# Ma'ruza L14 [I2]-slayd`);
- **m14 contracts.py imzosiga AYNAN mos** (index(texts,batch_size); answer(question,k)→**dict**
  {answer,sources,confidence}; **save_index/load_index BOR**; ⚠️ HAQIQIY modul — consumed_by m15);
- **sentence-transformers/faiss/LLM-ixtiyoriy**: Kaggle ST embedding + FAISS IndexFlatIP + LLM API; offline
  **TF-IDF + numpy kosinus + ekstraktiv javob** (haqiqiy LLM yo'q); `USE_ST`/`HAS_FAISS`/`USE_LLM` bayroqlari,
  builder mahalliy False ga majburlaydi; GPU'siz/faiss'siz/LLM'siz uchdan-uchgacha; offline = `d15_checkpoints/`
  kichik original uz bilim bazasi;
- answer dict strukturaviy assert (3 kalit; sources len=k; confidence∈[0,1]); javob ekstraktiv/demo KUTILGAN — halol;
- practice-notebook tuzilishi (§1–§6, so'nuvchi tayanch, PRIMM);
- mahalliy darvozalar: JSON valid, SOLUTIONS CPU'da bajariladi (har assert o'tadi), terminologiya toza,
  ASCII, Kirill 0, seeds; **save_index/load_index test**;
- 3 commit: `day15: practice — P14 …` / `day15: capstone — m14 …` / `day15: qa — P14 report`.

---

## Ilova — repozitoriy fayl holati

```
course/lectures/   : d01..d05, d06..d12, d13_transfer_learning, d14_rag  (tex+pdf; d02 tex)
course/practices/  : d02_p1 ... d14_p13_finetune  (+ _SOLUTIONS, + d0N_checkpoints/)
capstone/modules/  : m01..m13 (+ m05b)   (m14 — P14 da quriladi)
course/milestones/ : w1_*, w2_*, w3_*
course/qa/         : d01, L1–L14, P1–P13, w1, w2, w3 (+ skriptlar)
```

So'nggi commitlar:
```
bebcc12 docs: HOLAT_HISOBOT.md — L14 ga yangilandi (RAG ma'ruzasi maqsadi)
356a6e0 day14: lecture — L14 RAG va vektor ma'lumotlar bazalari
7803385 docs: HOLAT_HISOBOT.md — P13 ga yangilandi (L13 yopildi, m13 keyingi)
772c26c day14: qa — P13 report (all gates PASS, 12/12 local asserts; LogReg fallback + toza-torch BCE)
5cfedc5 day14: capstone — m13 FineTunedClassifier (DistilBERT+Trainer + TF-IDF/LogReg fallback)
```
```
origin/rtm = bebcc12 (to'liq sinxron, 0 ortda)
```
