# NLP KURSI — LOYIHA HOLAT HISOBOTI

**Sana:** 2026-06-16 · **Branch:** `feat/help_to_improve` · **Oxirgi commit:** `2c86662`
**Remotelar:** `origin` (datascientistn1/uznlp_course), `rtm` (multi-modal-rtm/uznlp_course).
> Holat: ikkala remote ham **to'liq sinxron** (push qilingan, 0 ortda).

> Bu hisobot mustaqil: uni boshqa Claude (yoki hamkasb) ga berib, **w4 (M4) — 4-hafta milestone
> (deploy + bilim testi + agent scaffold), kursning OXIRGI artefakti** uchun bosqichma-bosqich prompt olish mumkin.

---

## 1. Loyiha nima

O'zbek tilidagi (lotin yozuvi) 4 haftalik intensiv NLP kursining BARCHA o'quv
materiallarini ishlab chiqarish loyihasi (15-iyun – 10-iyul 2026, 16 o'quv kuni +
4 chorshanba milestone). Har bir tinglovchi yagona kapstone loyiha — **"O'zbek
Hujjat Yordamchisi"** — quradi. Kaggle bepul rejim. **w4 — loyihaning yakuniy artefakti.**

## 2. Ish intizomi (qat'iy)

- **Bir kunda bitta:** artefakt → barcha sifat darvozalari → QA → TO'XTA → inson tasdig'i → keyingi.
- **`course/course_map.yaml` — yagona haqiqat manbasi.** Mavzu, vazifalar, formatlar faqat shundan. Day "M4"
  (`type: milestone`, `id: w4`) qatori tasdiqlangan.
- **`.claude/skills/`** majburiy: `practice-notebook` (P16 uchun), `uzbek-course-style`, `kaggle-hardware`.
- **Mahalliy vositalar:** Python 3.13 — **fastapi/httpx/uvicorn BOR** (TestClient ishlaydi),
  **python-docx (`docx`)/openpyxl BOR** (DOCX/XLSX yaratish ishlaydi), numpy/sklearn/torch/transformers bor.
  Demak w4 ning HAR uch vazifasi **mahalliy to'liq tekshiriladi**.
- **Amaliyot uslubi:** milestone — brief (`.md`) + check (`.py`); P16 — builder orqali notebook + app.py + test fayllar.

## 3. BAJARILGAN ISHLAR

**Ma'ruzalar:** L1–L16 — TO'LIQ (d01…d16; tex+pdf; d02 faqat tex). Har biriga QA. ✅
**Amaliyotlar:** P1 (m01) … P15 (m15 — kapstone yakuni). Har biri +SOLUTIONS +QA, mahalliy bajarilgan. ✅
**Modullar:** m01–m15 (+ m05b). ✅  KAPSTONE TO'LIQ.
**Milestonelar:** w1 ✅, w2 ✅, w3 ✅. **w4 (M4) — QOLDI (oxirgi artefakt).**

> Holat: ma'ruzalar **L16 gacha (TO'LIQ)**, amaliyotlar **P15 gacha**, milestonelar **w3 gacha**.
> **Keyingi (va OXIRGI) artefakt — w4 (M4) milestone.** w4 dan keyin loyiha to'liq yakunlanadi. 🎓

## 4. KEYINGI QADAM — w4 (M4) (so'ralgan)

> w4 = **course_map `type: milestone, id: w4`** (sana 2026-07-08, chorshanba; `flipped_classroom: true`).
> Bu **UCH async vazifani** birlashtiruvchi milestone (Day 15 dan oldin topshiriladi). M4 P16 (FastAPI)
> L16 (MLOps) dan OLDIN bajariladi — qasddan (flipped): tinglovchilar avval amalda, keyin nazariya.

**w4 UCH vazifasi (course_map M4):**

**Task A — P16: NLP modelini API sifatida joylashtirish** (`practice_official_no: 16`)
- 4 subitem: (1) o'qitilgan sentiment modelni saqlash; (2) FastAPI veb-server + model yuklash;
  (3) `POST /predict` JSON endpoint; (4) Dockerfile + konteyner (lokal ishga tushirish).
- `notebook_style: fully_worked_primm` — **so'nuvchi tayanch YO'Q, BUTUN kod beriladi** (konsultatsiya seansi,
  baholanadigan blank emas). PRIMM (Bashorat/Tekshiring/O'zgartiring) bilan.
- **Kapstone moduli:** `SentimentAPI` → fayl **`capstone/app.py`**. Interfeys:
  `FastAPI app: POST /predict → {text: str} → {sentiment: str, confidence: float}`. consumed_by: [] (yakuniy mahsulot).
- `modules_available_at_m4: [1..13]` (m13 bor). gpu_required: false.

**Task B — Bilim testi** (knowledge_test)
- Format: **`course/final_test.docx`** (savol varaqasi) + **`course/final_test.xlsx`** (javoblar kaliti).
- Qamrov: **L1–L14** (Day 15 dan oldingi barcha ma'ruzalar). ~30 savol, asinxron, o'z-o'zini vaqtlash.
- DOCX (python-docx) + XLSX (openpyxl) — mahalliy yaratish mumkin.

**Task C — Agent scaffold (PREP)** — ⚠️ **ALLAQACHON TAYYOR**
- course_map: `capstone/modules/m15_langchain_agent.py` scaffold (m13/m14/m12/m04 Tool stub'lari).
- **m15 P15 da TO'LIQ qurilgan** (faqat scaffold emas) — Task C bajarilgan; w4 uni tasdiqlaydi/havola qiladi.

**w4 milestone artefaktlari (w1–w3 formati bo'yicha):**
- **`course/milestones/w4_milestone.md`** — brief (3 vazifa: A/B/C tushuntirilgan, topshirish formati).
- **`course/milestones/w4_check.py`** — tekshiruv skripti (assertlar): app.py TestClient `/predict`,
  final_test fayllar mavjudligi, m15 import.
- **`course/qa/w4_report.md`** — QA hisobot.

**UZVIYLIK / "qulflangan" davomiylik:** `POST /predict` javobi `{sentiment, confidence}` — L15 [I3] API misoli
va **qulflangan yorliqlar `ijobiy`/`salbiy`** (L2 [I2]) bilan mos. w4_check.py shuni tekshiradi:
`/predict {"text": "..."}` → `sentiment ∈ {ijobiy, salbiy}`, `confidence ∈ [0,1]`.

## 5. ⚠️ DIZAYN — barcha vazifalar MAHALLIY tekshiriladi

> ⚠️ MUHIM: w4 ning barcha vositalari mahalliy BOR (fastapi/httpx/uvicorn/docx/openpyxl) → 3 vazifa ham
> mahalliy to'liq bajariladi va tekshiriladi (oldingi P12–P15 dagi "fallback" zarurati YO'Q bu yerda).
- **Task A app.py:** `m13 FineTunedClassifier` (USE_TRANSFORMERS=False → TF-IDF/LogReg) yoki m02 ni startda
  kichik korpusda fit qiladi (yoki saqlangan modelni yuklaydi) — internetsiz. `POST /predict` JSON qaytaradi.
  Mahalliy tekshiruv: `from fastapi.testclient import TestClient` (httpx bor) → `client.post("/predict", ...)`.
  Dockerfile beriladi (ko'rsatiladi, mahalliyda build qilinmaydi — kaggle-hardware: Docker ixtiyoriy).
- **Task B final_test:** python-docx bilan `.docx` (30 savol, L1–L14), openpyxl bilan `.xlsx` (javoblar kaliti).
  Savollar har ma'ruza mavzusidan (BoW/TF-IDF, NB, embedding, n-gram/Viterbi, Word2Vec, RNN, LSTM/GRU,
  generatsiya, NER, seq2seq/attention, Transformer, transfer/BERT, RAG). MCQ + qisqa javob aralash.
- **Task C:** m15 mavjud — w4_check.py `import m15_langchain_agent` + `DocumentAssistantAgent().run(...)` ni sinaydi.
- **Builder:** `_build_w4.py` (commit qilinmaydi) — app.py + P16 notebook (JSON) + final_test.docx/xlsx yozadi,
  so'ng TestClient va docx/xlsx ochilishini exec qilib tekshiradi.

## 6. P16 NOTEBOOK TUZILISHI (fully_worked_primm — so'nuvchi tayanch YO'Q)

§1 Muhit (fastapi/uvicorn, OFFLINE_FALLBACK) → §2 yaxlit natija (TestClient bilan `/predict` demo) →
§3 PRIMM (BUTUN app.py kodi to'liq beriladi: model yuklash global, `POST /predict`, JSON sxema —
Bashorat/Tekshiring/O'zgartiring) → §4 Dockerfile + konteyner (to'liq beriladi; lokal build izohi) →
§5 loyihaga ulash (app.py ni `capstone/app.py` ga yozish, TestClient test, git) → §6 yakun + kapstone himoyasi.
> DIQQAT: P16 — baholanadigan blank YO'Q (fully_worked); lekin TestClient asserti BO'LSIN (mahalliy bajariladi).

## 7. SIFAT DARVOZALARI (MAHALLIY — kechiktirilmaydi)

- **w4_check.py mahalliy ishlaydi, 0 istisno, har assert o'tadi.** Quyidagilar:
  - app.py: `TestClient(app).post("/predict", json={"text": "mahsulot zo'r"})` → 200; javob
    `{"sentiment": ∈{ijobiy,salbiy}, "confidence": float∈[0,1]}`.
  - final_test.docx ochiladi (python-docx) va ≥30 savol/paragraf bor; final_test.xlsx ochiladi (openpyxl),
    javoblar kaliti to'liq.
  - m15 import + `DocumentAssistantAgent` mavjud, `run()` str qaytaradi.
- **P16 notebook:** JSON valid (nbformat 4.5; har katak id); TestClient bilan mahalliy bajariladi; assert o'tadi.
- **Terminologiya grep toza** (`professor\|talaba\|student\|o'qituvchi` = 0); **yorliqlar `ijobiy`/`salbiy`**
  (musbat/manfiy YO'Q) — app.py va final_test da ham; ASCII apostrof; U+2019 yo'q; **Kirill 0** (notebook +
  app.py + w4_*.py + final_test matni skan); BOM yo'q.
- **GPU talab qilinmaydi** (mahalliy CPU); seeds; data <=500 MB.

## 8. SO'RALADIGAN PROMPT

**w4 (M4) — 4-hafta milestone (deploy + bilim testi + agent scaffold), kursning oxirgi artefakti — ni ishlab
chiqarish uchun bosqichma-bosqich prompt** yozib bering —
- course_map M4 (`id: w4`) spetsifikatsiyasiga aniq mos: UCH vazifa (A: P16 FastAPI/Docker; B: bilim testi
  DOCX/XLSX, L1–L14; C: agent scaffold — m15 ALLAQACHON tayyor, faqat tasdiqlanadi);
- **Task A:** `capstone/app.py` (FastAPI `POST /predict` → `{sentiment, confidence}`; m13 LogReg/m02 offline)
  + P16 notebook (`fully_worked_primm`, so'nuvchi tayanch YO'Q) + Dockerfile (ko'rsatiladi); mahalliy TestClient
  asserti (`fastapi.testclient`, httpx bor); javob `sentiment ∈ {ijobiy,salbiy}`, `confidence ∈ [0,1]`
  (L15 [I3] + qulflangan yorliqlar bilan mos);
- **Task B:** `course/final_test.docx` (≥30 savol, L1–L14 qamrovi, MCQ + qisqa javob) + `course/final_test.xlsx`
  (javoblar kaliti); python-docx + openpyxl bilan (mahalliy bor);
- **Task C:** m15 mavjudligini tasdiqla (P15 da qurilgan) — w4_check.py uni import qilib `run()` ni sinaydi;
- **milestone artefaktlari (w1–w3 formati):** `course/milestones/w4_milestone.md` (brief, 3 vazifa),
  `course/milestones/w4_check.py` (assertlar: TestClient /predict + final_test fayllar + m15), `course/qa/w4_report.md`;
- **mahalliy darvozalar:** w4_check.py CPU'da ishlaydi (0 istisno, har assert o'tadi); P16 JSON valid +
  TestClient bajariladi; terminologiya toza; yorliqlar ijobiy/salbiy; ASCII; Kirill 0; BOM yo'q;
- **builder** (`_build_w4.py`, commit qilinmaydi) app.py + P16 notebook + final_test fayllarni yozadi va exec/tekshiradi;
- commitlar (taklif): `m4: milestone — w4 brief + check` / `m4: capstone — app.py SentimentAPI` /
  `m4: practice — P16 fastapi notebook` / `m4: test — final_test docx+xlsx` / `m4: qa — w4 report`.

---

## Ilova — repozitoriy fayl holati

```
course/lectures/   : d01..d16  (tex+pdf; d02 tex)   — TO'LIQ
course/practices/  : d02_p1 ... d16_p15_agent  (+ _SOLUTIONS, + d0N_checkpoints/)   (P16 — w4 da)
capstone/modules/  : m01..m15 (+ m05b)   — KAPSTONE TO'LIQ   (app.py — w4 da quriladi)
course/milestones/ : w1_*, w2_*, w3_*    (w4_* — w4 da)
course/qa/         : d01, L1–L16, P1–P15, w1, w2, w3   (w4 — w4 da)
course/            : final_test.{docx,xlsx} — w4 da
```

So'nggi commitlar:
```
2c86662 docs: HOLAT_HISOBOT.md — L16 ga yangilandi (MLOps yakuniy ma'ruzasi maqsadi)
d51ebf4 day16: lecture — L16 NLP amaliyotida MLOps (kursning yakuniy ma'ruzasi)
5cef4ed docs: HOLAT_HISOBOT.md — P15 ga yangilandi (L15 yopildi, m15 kapstone yakuni)
9e73e3c day16: qa — P15 report (all gates PASS, 14/14 local asserts; rule-based router; capstone finale)
c006d86 day16: capstone — m15 DocumentAssistantAgent (LangChain ReAct + rule-based router fallback)
```
```
origin/rtm = 2c86662 (to'liq sinxron, 0 ortda)
```
