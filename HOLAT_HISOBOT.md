# NLP KURSI — LOYIHA HOLAT HISOBOTI

**Sana:** 2026-06-16 · **Branch:** `feat/help_to_improve` · **Oxirgi commit:** `7125a69`
**Remotelar:** `origin` (datascientistn1/uznlp_course), `rtm` (multi-modal-rtm/uznlp_course).
> Eslatma: P5 (4 commit), L6 (`5a15d49`) **va** w2 (`e70682d`, `7125a69`) — jami 7 commit
> hali remote'larga PUSH QILINMAGAN.

> Bu hisobot mustaqil: uni boshqa Claude (yoki hamkasb) ga berib, **P6 (6-amaliyot:
> m06 CustomWord2Vec — o'zbek korpusida CBOW ni noldan o'qitish)** uchun bosqichma-bosqich
> prompt olish mumkin.

---

## 1. Loyiha nima

O'zbek tilidagi (lotin yozuvi) 4 haftalik intensiv NLP kursining BARCHA o'quv
materiallarini ishlab chiqarish loyihasi (15-iyun – 10-iyul 2026, 16 o'quv kuni +
4 chorshanba milestone). Har bir tinglovchi yagona kapstone loyiha — **"O'zbek
Hujjat Yordamchisi"** — quradi. Kaggle bepul rejim, CPU (1–2 hafta).

## 2. Ish intizomi (qat'iy)

- **Bir kunda bitta:** artefakt → barcha sifat darvozalari → QA → TO'XTA → inson tasdig'i → keyingi.
- **`course/course_map.yaml` — yagona haqiqat manbasi.** Mavzu, maqsadlar, hand_example sonlari faqat shundan.
- **`.claude/skills/`** majburiy: `practice-notebook`, `lecture-beamer`, `uzbek-course-style`, `kaggle-hardware`.
- **Kun juftlash:** L(N) ma'ruza P(N) amaliyotiga tayyorlaydi. Har P ning birinchi
  asserti mos L ning [I] hand_example sonini tekshiradi (kuzatiluvchanlik).
- **Mahalliy vositalar:** MiKTeX (pdflatex — ma'ruzalar) + Python 3.13
  (numpy 2.2.6 / sklearn 1.8.0 / matplotlib). **gensim/datasketch/nltk YO'Q** (modullar ixtiyoriy bog'lansin).
- **Amaliyot uslubi:** `_build_pN.py` builder-skript orqali — offline data yaratish, modulni
  yozish, student + SOLUTIONS notebook'larni JSON sifatida qurish, SOLUTIONS kataklarini
  ishga tushirib assertlarni tekshirish.

## 3. BAJARILGAN ISHLAR

**Ma'ruzalar:** L1 `d01_nlp_asoslari` (tex+pdf), L2 `d02_klassik_tasnif` (tex; **PDF yo'q, Title Case**),
L3 `d03_vektor_embedding` (tex+pdf), L4 `d04_qidiruv_imlo` (tex+pdf, 42 slayd),
L5 `d05_til_modellari` (tex+pdf, 45 slayd), L6 `d06_word2vec` (tex+pdf, 48 slayd). Har biriga QA. ✅

**Amaliyotlar:** P1 (m01), P2 (m02), P3 (m03), P4 (m04), P5 `d06_p5_autocomplete_pos`
(m05 + m05b) — har biri +SOLUTIONS +checkpoints +QA, mahalliy bajarilgan. ✅

**Modullar:** m01 TextPreprocessor, m02 SentimentClassifier, m03 PretrainedEmbedder,
m04 SpellLSHRetriever, m05 Autocomplete, m05b POSTagger. ✅

**Milestonelar:** w1 (21/21 tekshiruv ✅), **w2 (klassik pipeline m01–m05, 40/40 tekshiruv ✅) YANGI**.

> Holat: ma'ruzalar L6 gacha, amaliyotlar P5 gacha, milestonelar w2 gacha. **Keyingi
> xronologik artefakt — P6** (Day 7, 25-iyun ertalab; L6 ga juft). So'ng L7 (Day 7 ma'ruza — RNN).

## 4. KEYINGI QADAM — P6 (so'ralgan)

> P6 = **course_map Day 7, `practice_official_no: 6`**; **L6** (Word2Vec CBOW) ma'ruzasiga juft.
> Modul: **m06 CustomWord2Vec**, fayl `capstone/modules/m06_custom_word2vec.py`.
> Notebook: `course/practices/d07_p6_word2vec.ipynb` (+ `_SOLUTIONS` + `d07_checkpoints/`).

**P6 spetsifikatsiyasi (course_map Day 7):**
- **Mavzu:** "Maxsus korpus uchun Word2Vec modelini noldan o'qitish."
- **4 kichik bo'lim (practice_subitems):**
  1. O'zbek tilida maxsus korpus yig'ish va tozalash.
  2. Gensim kutubxonasi yordamida CBOW arxitekturasini sozlash.
  3. Modelni o'qitish va hosil bo'lgan embeddinglar sifatini tekshirish.
  4. Modelni saqlash va vizualizatsiya (TensorBoard) vositasida ko'rish.
- **Periferiya (to'liq beriladi — PRIMM):**
  - uz_news_full korpusini m01 bilan tozalash va `LineSentence` ga aylantirish.
  - TensorBoard projector metadata fayli va sprite rasm.
- **Yadro (core — tinglovchi yozadi):**
  - `Word2Vec(vector_size=100, window=5, min_count=3, sg=0)` — CBOW o'qitish.
  - `wv.most_similar()` va so'z analogiyalarini sinash.
  - Modelni `.kv` ga saqlash va keyingi sessiyada yuklash.
- **corpus_subset:** uz_news_full (onlayn). **OFFLINE_FALLBACK:** bundle qilingan kichik korpus
  (`d07_checkpoints/` — masalan d05 dagi `uz_news_corpus.txt` uslubida).
- **gpu_required: yo'q** (CPU).

**QULFLANGAN birinchi assert (L6 [I2] → P6).** Notebook CBOW proyeksiya qadamini ochib
ko'rsatsin (kontekst vektorlari o'rtachasi) va assert aynan shuni tekshirsin:
- kontekst = [men, sevaman], emb(men)=[0.5,0.3], emb(sevaman)=[0.1,0.7].
- `cbow_input = np.mean([[0.5,0.3],[0.1,0.7]], axis=0)` → **[0.3, 0.5]**.
- `assert np.allclose(cbow_input, [0.3, 0.5])  # Ma'ruza L6 [I2]-slayd bilan solishtiring`.

**m06 shartnomasi (capstone/contracts.py — QAT'IY, AYNAN MOS):**
```
class CustomWord2Vec:
    train(texts: list[list[str]], vector_size=100, window=5, min_count=3, epochs=10) -> None
    embed(word: str) -> np.ndarray          # OOV uchun nol-vektor
    most_similar(word: str, n=5) -> list[tuple[str, float]]
    save(path: str) -> None
    load(path: str) -> None
```
> DIQQAT: contracts.py da `train` aniq kwargs (vector_size/window/min_count/epochs) bilan,
> va `most_similar` ham bor — promptdagi "train(texts, **kwargs)" emas, AYNAN shu imzo.
> consumed_by: [8, 9] (m08 GRU/LSTM pretrained init, m09 generator).

## 5. GENSIM-IXTIYORIY DIZAYN (P3/m03 namunasi bilan bir xil)

- **Kaggle yo'li:** `gensim.models.Word2Vec(..., sg=0)` bilan CBOW o'qitiladi, `.kv` saqlanadi
  (`wv.most_similar` ishlatiladi). Bu kod **periferiya/izoh** sifatida ko'rsatiladi.
- **Offline yo'l (mahalliy, gensimsiz):** toza-numpy CBOW (kichik korpus, kam epoch) —
  proyeksiya = kontekst o'rtachasi, softmax/negative-sampling soddalashtirilgan; `embed`,
  `most_similar` (kosinus) toza-numpy bilan. `HAS_GENSIM` bayrog'i faqat xabarni almashtiradi.
- Natija: notebook **gensimsiz va uz_news_full siz** uchdan-uchgacha ishlaydi — m03/m04/m05
  ixtiyoriy-kutubxona namunasi.

## 6. PRACTICE-NOTEBOOK TUZILISHI (skill)

§1 Muhit (seeds, OFFLINE_FALLBACK, m01 yo'li, HAS_GENSIM) → §2 yaxlit natija (inline CBOW demo)
→ §3 PRIMM periferiya (m01 tozalash + LineSentence; TensorBoard metadata — to'liq beriladi)
→ Checkpoint → §4 yadro: **so'nuvchi tayanch** (Namuna → Birgalikda `# === SIZNING KODINGIZ ===`
→ Mustaqil), har blank katakka mos **assert**; birinchi assert = L6 [I2] [0.3,0.5]
→ §5 loyihaga ulash (m06 ni yozish, import test, git) → §6 tadqiqot + exit ticket.

## 7. SIFAT DARVOZALARI (MAHALLIY — kechiktirilmaydi)

- **JSON valid** (student + SOLUTIONS): `nbformat` 4.5; har katak `id`.
- **Uchdan-uchgacha bajariladi** (`OFFLINE_FALLBACK=True`): SOLUTIONS kataklari mahalliy
  ishga tushadi (Python 3.13, numpy/sklearn; **gensimsiz**), **har assert o'tadi**, 0 istisno.
- **Student stub kataklar `compile()` toza.**
- **Birinchi (qulflangan) assert:** `np.allclose(cbow_input, [0.3, 0.5])` —
  `# Ma'ruza L6 [I2]-slayd`.
- **Har blank region mos assert bilan;** m06 shartnoma mosligi (train/embed/most_similar/save/load).
- **Kapstone uzviyligi:** m01 dan foydalanadi (korpus tozalash). consumed_by [8,9].
- **No GPU / VRAM;** seeds (`random.seed(42)`, `np.random.seed(42)`); checkpoint katak(lar)i.
- **Terminologiya grep toza:** `professor|talaba|student|o'qituvchi` — 0 mos (barcha artefaktlar).
- **ASCII apostrof;** U+2019 yo'q; Kirill yo'q; BOM yo'q.

## 8. SO'RALADIGAN PROMPT

**P6 (6-amaliyot: m06 CustomWord2Vec — o'zbek korpusida CBOW ni noldan o'qitish) ni ishlab
chiqarish uchun bosqichma-bosqich prompt** yozib bering —
- course_map Day 7 (practice 6) spetsifikatsiyasiga aniq mos: 4 subitem, periferiya
  (uz_news_full→m01→LineSentence; TensorBoard metadata), yadro (`Word2Vec sg=0` CBOW,
  `most_similar`, `.kv` saqlash);
- **qulflangan birinchi assert** = L6 [I2]: `cbow_input = mean([0.5,0.3],[0.1,0.7]) = [0.3,0.5]`
  (`# Ma'ruza L6 [I2]-slayd`);
- **m06 contracts.py imzosiga AYNAN mos** (train kwargs vector_size/window/min_count/epochs;
  embed; most_similar; save; load);
- **gensim-ixtiyoriy** (P3/m03 namunasi): Kaggle'da gensim CBOW, mahalliy toza-numpy fallback;
  notebook gensimsiz + uz_news_full siz uchdan-uchgacha ishlasin;
- practice-notebook tuzilishi (§1–§6, so'nuvchi tayanch, PRIMM periferiya); kapstone uzviyligi (m01);
- mahalliy sifat darvozalari: JSON valid, SOLUTIONS mahalliy bajariladi (gensimsiz, har assert o'tadi),
  terminologiya toza, ASCII apostrof, seeds, no-GPU;
- bitta commit notebook+modul+checkpoints uchun, alohida QA commit
  (`day07: practice — P6 …` / `day07: capstone — m06 …` / `day07: qa — P6 report`).

---

## Ilova — repozitoriy fayl holati

```
course/lectures/   : d01(tex,pdf) d02(tex) d03(tex,pdf) d04(tex,pdf) d05(tex,pdf) d06(tex,pdf)
course/practices/  : d02_p1 d03_p2 d04_p3 d05_p4 d06_p5  (+ _SOLUTIONS, + d0N_checkpoints/)
capstone/modules/  : m01 m02 m03 m04 m05 m05b   (m06 — P6 da quriladi)
course/milestones/ : w1_milestone.md w1_check.py  w2_milestone.md w2_check.py
course/qa/         : d01, L1–L6, P1–P5, w1, w2 (+ skriptlar)
```

So'nggi commitlar:
```
7125a69 w2: qa — w2 report (all gates PASS, 40/40 local checks)
e70682d w2: milestone — w2 brief + check (klassik pipeline m01-m05 integratsiyasi)
5a15d49 day06: lecture — L6 Word2Vec (CBOW): neyron so'z embeddinglari
beee8b9 day06: qa — P5 report (all gates PASS, local execution nltksiz)
5fe86a0 day06: capstone — m05b POSTagger
```
