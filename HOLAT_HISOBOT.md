# NLP KURSI — LOYIHA HOLAT HISOBOTI

**Sana:** 2026-06-15 · **Branch:** `feat/help_to_improve` · **Oxirgi commit:** `681561e`

> Bu hisobot mustaqil: uni boshqa Claude (yoki hamkasb) ga berib, keyingi qadam
> uchun bosqichma-bosqich prompt olish mumkin.

---

## 1. Loyiha nima

O'zbek tilidagi (lotin yozuvi) 4 haftalik intensiv NLP kursining BARCHA o'quv
materiallarini ishlab chiqarish loyihasi (15-iyun – 10-iyul 2026, 16 o'quv kuni).
Har bir tinglovchi 16 kun davomida yagona kapstone loyiha — **"O'zbek Hujjat
Yordamchisi"** — quradi; har amaliyot bitta modul (m01…m15) qo'shadi. Barcha
amaliyotlar Kaggle bepul rejim (1–2 hafta CPU).

## 2. Ish intizomi (qat'iy)

- **Bir kunda bitta:** ma'ruza/amaliyot ishlab chiqariladi → barcha sifat
  darvozalari → QA hisoboti → TO'XTA → inson tasdig'i → keyingi.
- **`course/course_map.yaml` — yagona haqiqat manbasi.** Mavzu, o'lchanadigan
  maqsadlar, hand_example sonlari faqat shundan olinadi.
- **`.claude/skills/`** majburiy: `lecture-beamer`, `practice-notebook`,
  `uzbek-course-style`, `kaggle-hardware`.
- **Kun juftlash:** L(N) ma'ruza ertangi P(N) amaliyotiga tayyorlaydi. Har P ning
  birinchi asserti mos L ning [I] hand_example sonini tekshiradi (kuzatiluvchanlik).
- **Commit formati:** `dayNN: lecture|practice|qa|capstone — qisqa tavsif`.

## 3. BAJARILGAN ISHLAR

**Infratuzilma:** barcha governance hujjatlari, 4 skill, `course_map.yaml`
(16 kun to'liq rejalashtirilgan), `capstone/SPEC.md`, `capstone/contracts.py`
(16 modul imzosi).

**1-kun (orientatsiya):** SPEC, contracts, `d01_orientatsiya.ipynb`,
`d01_kirish.tex/pdf`, `HISOB_YARATISH.md`, `pre_course.{docx,xlsx}` —
QA: `d01_qa.md` ✅

**Ma'ruzalar (lectures):**

| Ma'ruza | Mavzu | Holat |
|---|---|---|
| L1 `d01_nlp_asoslari` | NLP asoslari, preprocessing, BoW/TF-IDF | ✅ tex+**pdf**, QA L1 (~48 slayd) |
| L2 `d02_klassik_tasnif` | LR, Naive Bayes, metrikalar, etika | ✅ tex, QA L2 (~43 slayd) — **PDF hali yo'q** |
| L3 `d03_vektor_embedding` | Vektor fazo, embedding, kosinus, PCA | ✅ tex+**pdf**, QA L3 (40 slayd) |
| L4 `d04_qidiruv_imlo` | LSH qidiruv, noisy channel, Levenshtein | ✅ tex+**pdf**, QA L4 (**42 slayd**, L1–L3 darajasiga kengaytirilgan) |

**Amaliyotlar (practices):**

| Amaliyot | Mavzu | Modul | Holat |
|---|---|---|---|
| P1 `d02_p1_preprocessing` | Preprocessing pipeline | m01 TextPreprocessor | ✅ +SOLUTIONS +checkpoints, QA P1 |
| P2 `d03_p2_sentiment` | Sentiment tasnif (LR/NB) | m02 SentimentClassifier | ✅ +SOLUTIONS +checkpoints, QA P2 (mahalliy bajarildi) |

**Qurilgan kapstone modullar:** `m01_text_preprocessor.py`,
`m02_sentiment_classifier.py`.

**Muhim:** mahalliy MiKTeX (pdflatex) va Python 3.13 + sklearn mavjud —
ma'ruzalar mahalliy kompilyatsiya qilinadi, amaliyotlar mahalliy bajariladi
(bajarish darvozasi endi kechiktirilmaydi).

## 4. KEYINGI QADAM (eng mantiqiy tartib)

Ma'ruzalar Day 4 (L4) gacha tayyor, amaliyotlar esa Day 3 (P2) gacha —
**amaliyotlar ortda**. 1-haftani yakunlash uchun:

1. **P3 — `d04_p3_embeddings.ipynb`** (m03 **PretrainedEmbedder**) — L3 bilan
   juftlashadi. course_map Day 4 / practice_official_no 3. Birinchi assert:
   `abs(cos_val - 0.667) < 1e-3` (L3 [I2]). Periferiya: `cc_uz_100k.kv` yuklash,
   PCA scatter. O'zak: `cosine_similarity`, `most_similar`, analogiya, `oov_rate`.
   **← Eng aniq keyingi nomzod.**
2. **P4 — `d05_p4_spell_lsh.ipynb`** (m04 **SpellLSHRetriever**) — L4 bilan
   juftlashadi. Birinchi assert: `edit_distance("qo'l","ko'l")==1` (L4 [I3]).
   O'zak: edit_distance DP, noisy channel `correct()`, MinHash LSH.
3. **w1 milestone** — `course/milestones/w1_milestone.md` + `w1_check.py`
   (m01–m04 ni birlashtirish + contracts tekshiruvi).
4. Keyin **2-hafta** (L5/P5…): L5 mavzusi N-gram, perplexity, HMM/Viterbi
   (course_map Day 5da, hand_example tayyor).

## 5. KEYINGI PROMPT UCHUN MAJBURIY KONVENTSIYALAR

- **Auditoriya:** "tinglovchi/tinglovchilar". TAQIQLANGAN: `professor, talaba,
  student, o'qituvchi` (oxirgisi faqat lingvistik misol so'z sifatida mumkin).
  Grep darvozasi.
- **Yorliqlar qulflangan:** `ijobiy`/`salbiy` (musbat/manfiy EMAS).
- **Uslub etaloni:** `d01_nlp_asoslari.tex` (ma'ruza) va
  `d02_p1_preprocessing.ipynb` (amaliyot) — sentence-case, tabiiy o'zbekcha,
  birinchi shaxs ko'plik ("o'rganamiz"). ASCII apostrof (`'`).
- **Amaliyot tuzilmasi (practice-notebook skill):** §1 muhit → §2 yaxlit natija
  → §3 PRIMM (periferiya, Bashorat/Tekshiring/O'zgartiring) → §4 so'nuvchi tayanch
  (Namuna → Birgalikda `# === SIZNING KODINGIZ ===` → Mustaqil) → §5 kapstone
  modul yozish + git → §6 tadqiqot + chiqish chiptasi. Har bo'sh joy uchun juft
  assert. Checkpoint kataklar. OFFLINE_FALLBACK=True. CPU-only, seedlar o'rnatilgan.
- **Modul:** `capstone/contracts.py` imzolariga ANIQ mos; oldingi modullar ustiga
  quriladi (P3 → m01 import).

## 6. OCHIQ MASALALAR (inson e'tibori kerak)

1. **course_map Day 3 korpusi:** P2 da course_map `uz_news_mini` o'rniga
   `uz_sentiment_uzum` (Uzum sharhlari, MIT) ishlatildi (inson ko'rsatmasi bilan).
   course_map Day 3 `corpus_subset` ni yangilash tavsiya etiladi
   (`P2_report.md` da hujjatlangan).
2. **L2 PDF yo'q:** `d02_klassik_tasnif.tex` kompilyatsiya qilinib PDF commit
   qilinmagan (endi mahalliy MiKTeX bilan qilish mumkin).
3. **L2 Title Case'da:** L2 sarlavhalari eski uslubda; L1/L3/L4 sentence-case.
   Izchillik uchun L2 ni sentence-case'ga keltirish mumkin (ixtiyoriy).

## 7. SO'RALADIGAN PROMPT

Yuqoridagilarga asoslanib, **P3 (m03 PretrainedEmbedder) amaliyotini ishlab
chiqarish uchun bosqichma-bosqich prompt** yozib bering — gold standard (P1/P2)
tuzilmasini nusxalaydigan, course_map Day 4 spetsifikatsiyasiga va L3 [I2]
hand_example'iga (`cos = 2/3 ≈ 0.667`) mos, mahalliy bajarish darvozasi bilan.

---

## Ilova — repozitoriy fayl holati

```
course/lectures/   : d01.(tex,pdf)  d02.tex  d03.(tex,pdf)  d04.(tex,pdf)
course/practices/  : d02_p1(.ipynb,_SOLUTIONS)  d03_p2(.ipynb,_SOLUTIONS)
                     d02_checkpoints/  d03_checkpoints/
capstone/modules/  : m01_text_preprocessor.py  m02_sentiment_classifier.py
course/qa/         : d01, L1, L2, L3, L4, P1, P2 (+ tekshiruv skriptlari)
course/milestones/ : (hali yo'q)
```

So'nggi commitlar:
```
681561e day04: lecture — L4 ni L1-L3 darajasiga kengaytirish (27 -> 42 slayd)
5cedcd3 day04: lecture — L4 masofaga asoslangan qidiruv va imlo tuzatish
a85a2af day03: qa — P2 report (all gates PASS, local execution)
1d9bbb1 day03: capstone — m02 SentimentClassifier
2cc3800 day03: practice — P2 sentiment notebook + SOLUTIONS
685e641 day03: lecture — L3 PDF ko'rigi: 4 kamchilik tuzatildi + compiled PDF
81a5d1b day03: lecture — L3 vektor fazo modellari va semantik munosabatlar
```
