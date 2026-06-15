# NLP KURSI — LOYIHA HOLAT HISOBOTI

**Sana:** 2026-06-15 · **Branch:** `feat/help_to_improve` · **Oxirgi commit:** `48e338b`
**Remotelar:** `origin` (datascientistn1/uznlp_course), `rtm` (multi-modal-rtm/uznlp_course).

> Bu hisobot mustaqil: uni boshqa Claude (yoki hamkasb) ga berib, **P5 amaliyoti**
> (m05 Autocomplete + m05b POSTagger) uchun bosqichma-bosqich prompt olish mumkin.

---

## 1. Loyiha nima

O'zbek tilidagi (lotin yozuvi) 4 haftalik intensiv NLP kursining BARCHA o'quv
materiallarini ishlab chiqarish loyihasi (15-iyun – 10-iyul 2026, 16 o'quv kuni +
4 chorshanba milestone). Har bir tinglovchi yagona kapstone loyiha — **"O'zbek
Hujjat Yordamchisi"** — quradi; har amaliyot bitta (yoki ikki) modul qo'shadi.
Kaggle bepul rejim, CPU (1–2 hafta).

## 2. Ish intizomi (qat'iy)

- **Bir kunda bitta:** artefakt → barcha sifat darvozalari → QA → TO'XTA → inson tasdig'i → keyingi.
- **`course/course_map.yaml` — yagona haqiqat manbasi.**
- **`.claude/skills/`** majburiy: `lecture-beamer`, `practice-notebook`,
  `uzbek-course-style`, `kaggle-hardware`.
- **Kun juftlash:** L(N) ma'ruza P(N) amaliyotiga tayyorlaydi. Har P ning birinchi
  asserti mos L ning [I] hand_example sonini tekshiradi (kuzatiluvchanlik).
- **Mahalliy vositalar:** MiKTeX + Python 3.13 (numpy/sklearn/matplotlib).
  **gensim/datasketch/nltk YO'Q** — modullar shularga ixtiyoriy bog'lansin
  (offline toza-python fallback bilan).

## 3. BAJARILGAN ISHLAR (1-hafta TO'LIQ yopildi)

**Ma'ruzalar:** L1 (tex+pdf), L2 (tex; **PDF yo'q, Title Case**), L3 (tex+pdf),
L4 (tex+pdf), L5 `d05_til_modellari` (tex+pdf). Har biriga QA. ✅

**Amaliyotlar:** P1 (m01), P2 (m02), P3 (m03), P4 (m04) — har biri +SOLUTIONS
+checkpoints +QA, mahalliy bajarilgan. ✅

**Modullar:** m01 TextPreprocessor, m02 SentimentClassifier, m03 PretrainedEmbedder,
m04 SpellLSHRetriever. ✅

**Milestone:** **w1** (`milestones/w1_milestone.md` + `w1_check.py`) — 21/21 tekshiruv
mahalliy o'tdi, QA `w1_report.md`. ✅

> Holat: ma'ruzalar L5 gacha, amaliyotlar P4 gacha. **Keyingi xronologik artefakt — P5**
> (Day 6, sesh 23-iyun). w2 milestone (24-iyun) m05 ga bog'liq, shuning uchun P5 dan KEYIN.

## 4. KEYINGI QADAM — P5 (so'ralgan)

> P5 = **course_map Day 6, practice_official_no 5**; ma'ruza **L5** (`d05_til_modellari`,
> N-gram/perplexity/POS/HMM/Viterbi) ga juftlashadi. Fayl: `d06_p5_autocomplete_pos.ipynb`.
> **Ikki kapstone modul** quriladi (P5 odatdagidan farqli — ikkita modul).

**P5 spetsifikatsiyasi (course_map Day 6):**
- **Mavzu:** "Autocomplete tizimi va so'z turkumini teglash dasturini yaratish."
- **4 kichik bo'lim:**
  1. N-gramma (bi/trigram) ehtimolliklarini korpusdan hisoblash
  2. Keyingi so'zni bashorat qiluvchi autocomplete prototipi
  3. HMM parametrlarini (π, A, B) hisoblash
  4. Viterbi algoritmi bilan so'z turkumlarini (POS) teglash
- **Periferiya (to'liq beriladi, PRIMM):** `nltk.ngrams()` + `defaultdict` chastota
  jadvali; HMM emission/transition matritsalarini CSV dan yuklash.
- **O'zak (so'nuvchi tayanch):** Bigram MLE + Laplace `P(w_i|w_{i-1})`; autocomplete
  top-3 keyingi so'z; Viterbi DP (δ, ψ matritsalar).
- **Kapstone modullar (IKKITA):**
  - **m05 Autocomplete** — `capstone/modules/m05_autocomplete.py`. Shartnoma:
    `train(texts, n) / complete(prefix: str, k: int) -> list[str] / perplexity(text: str) -> float`. consumed_by [16].
  - **m05b POSTagger** (pedagogik) — `capstone/modules/m05b_pos_tagger.py`. Shartnoma:
    `train(tagged_sents) / tag(tokens: list[str]) -> list[tuple[str,str]]`. consumed_by [] (yakuniy pipelineda emas).
- **Korpus:** `uz_news_full` (online). Offline: kichik bundled korpus + HMM CSV
  (`d06_checkpoints/`). Litsenziya CONFIRMED (LICENSES.md).
- **QULFLANGAN birinchi assert (L5 [I4] → P5):** Viterbi
  `δ(VB, t=2) = 0.3402` va teg ketma-ketligi `[NN, VB]`. m05b POSTagger / Viterbi
  Namunasida aynan shu HMM bilan (NN/VB; nlp/yozdi; π(NN)=0.7, π(VB)=0.3;
  B(nlp|NN)=0.9, B(nlp|VB)=0.1, B(yozdi|NN)=0.1, B(yozdi|VB)=0.9;
  A(VB|NN)=0.6, A(NN|VB)=0.3, A(NN|NN)=0.4, A(VB|VB)=0.7).
  `# Ma'ruza L5 [I4]-slayd bilan solishtiring`
  (Qo'shimcha kuzatiluvchanlik: m05 bigram uchun L5 [I1] `P(kitob|men)=2/3` ishlatish mumkin.)
- **Kapstone uzviyligi:** m01 TextPreprocessor (tokenizatsiya) ustiga quriladi.

**MUHIM — offline:** `nltk` mahalliy YO'Q bo'lishi mumkin. m05/notebook **nltk-ixtiyoriy**
bo'lsin — Kaggle da `nltk.ngrams`, offline da toza-python n-gram (zip/slicing) fallback.
Viterbi va bigram toza-python (tashqi kutubxonasiz). HMM matritsalari kichik CSV dan
(`d06_checkpoints/`). Notebook nltk va 240MBsiz TO'LIQ ishlasin.

## 5. KEYINGI PROMPT UCHUN MAJBURIY KONVENTSIYALAR

- **Auditoriya:** "tinglovchi/tinglovchilar". TAQIQLANGAN: `professor, talaba,
  student, o'qituvchi` (oxirgisi faqat lingvistik misol). Grep darvozasi.
- **Yorliqlar qulflangan:** `ijobiy`/`salbiy`. **POS teglari:** NN/VB (ot/fe'l) — L5 dagidek.
- **GOLD STANDARD (aynan nusxa):** `d02_p1`, `d03_p2`, `d04_p3`, `d05_p4` (+ _SOLUTIONS).
- **Amaliyot tuzilmasi (practice-notebook skill):** §1 muhit (seedlar,
  OFFLINE_FALLBACK=True, HAS_NLTK bayrog'i) → §2 yaxlit natija → §3 PRIMM periferiya
  (Bashorat/Tekshiring/O'zgartiring) → §4 so'nuvchi tayanch (Namuna → Birgalikda
  `# === SIZNING KODINGIZ ===` → Mustaqil) → §5 ikkala kapstone modul (m05, m05b)
  yozish + import test + git → §6 tadqiqot + chiqish chiptasi. Har bo'sh joy uchun JUFT assert.
- **Uslub:** sentence-case, tabiiy o'zbekcha, birinchi shaxs ko'plik, ASCII apostrof.
- **Modullar:** `contracts.py` imzolariga ANIQ mos; m01 ustiga quriladi.

## 6. SIFAT DARVOZALARI (MAHALLIY — kechiktirilmaydi)

- nbformat: yaroqli JSON (ikkala notebook).
- **MAHALLIY BAJARISH:** SOLUTIONS ni OFFLINE_FALLBACK=True bilan (nltk/gensim/datasketchsiz)
  ketma-ket bajar — HAR assert o'tsin. P2/P3/P4 dagidek builder-skript orqali `exec` qilib tekshir.
- Terminologiya grep toza; CPU-only; data <=500 MB; seedlar.

## 7. OCHIQ MASALALAR (inson e'tibori)

1. **course_map Day 3 korpusi:** P2 da `uz_sentiment_uzum` ishlatildi (map'da `uz_news_mini`)
   — map'ni yangilash tavsiya etiladi (P2_report.md da hujjatlangan).
2. **L2 PDF yo'q + Title Case:** mahalliy MiKTeX bilan tuzatish mumkin (ixtiyoriy).
3. **Keyingi ketma-ketlik:** P5 → (L6, Day 6 ma'ruzasi) → **w2 milestone** (m01–m05,
   24-iyun). w2 m05 ga bog'liq, shuning uchun P5 birinchi.

## 8. SO'RALADIGAN PROMPT

Yuqoridagilarga asoslanib, **P5 amaliyotini (m05 Autocomplete + m05b POSTagger:
bigram MLE + Laplace + perplexity, top-3 autocomplete, Viterbi POS teglash) ishlab
chiqarish uchun bosqichma-bosqich prompt** yozib bering —
- gold standard P1–P4 tuzilmasini nusxalaydigan;
- course_map Day 6 (practice 5) spetsifikatsiyasiga va L5 [I4] qulflangan
  hand_example'iga (`Viterbi δ(VB,t=2)=0.3402`, ketma-ketlik `[NN,VB]`) aniq mos;
- **nltk-ixtiyoriy** (offline toza-python n-gram) + bundled kichik korpus va HMM CSV
  (`d06_checkpoints/`) bilan — mahalliy bajarish darvozasi nltksiz o'tadigan;
- **ikkala** modul (m05, m05b) ni contracts.py ga mos qiladigan; m01 ustiga quriladigan;
- 4 alohida commit: practice / capstone m05 / capstone m05b / qa (yoki m05+m05b birga).

---

## Ilova — repozitoriy fayl holati

```
course/lectures/   : d01(tex,pdf) d02(tex) d03(tex,pdf) d04(tex,pdf) d05(tex,pdf)
course/practices/  : d02_p1(+SOL) d03_p2(+SOL) d04_p3(+SOL) d05_p4(+SOL)
                     d02..d05_checkpoints/
capstone/modules/  : m01 m02 m03 m04
course/milestones/ : w1_milestone.md  w1_check.py
course/qa/         : d01, L1–L5, P1–P4, w1 (+ skriptlar)
```

So'nggi commitlar:
```
48e338b w1: qa — w1 report (all gates PASS, 21/21 local checks)
95e505e w1: milestone — brief + check
34dd90b day05: qa — P4 report
d4e6881 day05: capstone — m04 SpellLSHRetriever
77561b3 day05: practice — P4 spell_lsh notebook + SOLUTIONS
```
