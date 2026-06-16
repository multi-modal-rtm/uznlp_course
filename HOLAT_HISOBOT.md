# NLP KURSI — LOYIHA HOLAT HISOBOTI

**Sana:** 2026-06-16 · **Branch:** `feat/help_to_improve` · **Oxirgi commit:** `96e01fc`
**Remotelar:** `origin` (datascientistn1/uznlp_course), `rtm` (multi-modal-rtm/uznlp_course).
> Holat: ikkala remote ham **to'liq sinxron** (push qilingan, 0 ortda).

> Bu hisobot mustaqil: uni boshqa Claude (yoki hamkasb) ga berib, **P9 (9-amaliyot:
> m09 TextGenerator — char-darajali LSTM bilan matn generatsiya)** uchun bosqichma-bosqich prompt olish mumkin.

---

## 1. Loyiha nima

O'zbek tilidagi (lotin yozuvi) 4 haftalik intensiv NLP kursining BARCHA o'quv
materiallarini ishlab chiqarish loyihasi (15-iyun – 10-iyul 2026, 16 o'quv kuni +
4 chorshanba milestone). Har bir tinglovchi yagona kapstone loyiha — **"O'zbek
Hujjat Yordamchisi"** — quradi. Kaggle bepul rejim.

## 2. Ish intizomi (qat'iy)

- **Bir kunda bitta:** artefakt → barcha sifat darvozalari → QA → TO'XTA → inson tasdig'i → keyingi.
- **`course/course_map.yaml` — yagona haqiqat manbasi.** Mavzu, maqsadlar, hand_example sonlari faqat shundan.
- **`.claude/skills/`** majburiy: `practice-notebook`, `lecture-beamer`, `uzbek-course-style`, `kaggle-hardware`.
- **Kun juftlash:** L(N) ma'ruza P(N) amaliyotiga tayyorlaydi. Har P ning birinchi
  asserti mos L ning [I] hand_example sonini tekshiradi (kuzatiluvchanlik).
- **Mahalliy vositalar:** Python 3.13 (numpy/sklearn/matplotlib/**torch 2.10 CPU**). gensim/datasketch/nltk YO'Q.
- **Amaliyot uslubi:** `_build_pN.py` builder — offline data, modulni yozish, student+SOLUTIONS
  notebook'larni JSON qurish, SOLUTIONS kataklarini exec qilib assert tekshirish. Builder commit qilinmaydi.

## 3. BAJARILGAN ISHLAR

**Ma'ruzalar:** L1 `d01` (tex+pdf), L2 `d02` (tex; **PDF yo'q, Title Case**), L3 `d03`,
L4 `d04`, L5 `d05`, L6 `d06_word2vec`, L7 `d07_rnn`, L8 `d08_gru_lstm`,
**L9 `d09_matn_generatsiya` (47 sl.) YANGI**. Har biriga QA. ✅

**Amaliyotlar:** P1 (m01), P2 (m02), P3 (m03), P4 (m04), P5 (m05+m05b), P6 (m06), P7 (m07),
P8 `d09_p8_gru_lstm` (m08). Har biri +SOLUTIONS +checkpoints +QA, mahalliy bajarilgan. ✅

**Modullar:** m01–m08 (+ m05b). ✅

**Milestonelar:** w1 (21/21 ✅), w2 (klassik pipeline m01–m05, 40/40 ✅).

> Holat: ma'ruzalar **L9 gacha**, amaliyotlar P8 gacha, milestonelar w2 gacha. **Keyingi
> xronologik artefakt — P9** (Day 10 ertalab; L9 ga juft). So'ng L10 (Day 10 ma'ruza — NER),
> keyin w3 milestone (1-iyul, m01–m08).

## 4. KEYINGI QADAM — P9 (so'ralgan)

> P9 = **course_map Day 10, `practice_official_no: 9`**; **L9** (generatsiya/temperature/Bi-RNN)
> ma'ruzasiga juft. Modul: **m09 TextGenerator**, fayl `capstone/modules/m09_text_generator.py`.
> Notebook: `course/practices/d10_p9_textgen.ipynb` (+ `_SOLUTIONS` + `d10_checkpoints/`).

**P9 spetsifikatsiyasi (course_map Day 10):**
- **Mavzu:** "LSTM va Bi-LSTM modellarni matn generatsiya qilishda sinab ko'rish."
- **4 kichik bo'lim (practice_subitems):**
  1. LSTM bilan badiiy asar uslubida matn generatsiya qiluvchi modelni o'qitish.
  2. Generatsiyada temperature parametrini o'zgartirib natijani kuzatish.
  3. Bi-LSTM qatlamini qo'shib, kontekstni o'rganish samaradorligini oshirish.
  4. Oddiy va ikki tomonlama modellar natijalarini solishtirish.
- **Periferiya (to'liq beriladi — PRIMM):**
  - Cho'lpon / Navoiy she'rlar korpusini yuklash va **char-level** tokenizatsiya.
  - next-char prediction training loop.
- **Yadro (tinglovchi yozadi):**
  - char-level LSTM generativ model (hidden=128, 2 qatlam).
  - temperature=0.3/0.7/1.2 da generatsiyalarni taqqoslash.
  - Bi-LSTM qo'shish va perplexity o'lchash.
- **corpus_subset:** uz_news_full yoki badiiy asar parchasi (kichik). **OFFLINE_FALLBACK:**
  `d10_checkpoints/` da bundle qilingan kichik o'zbek badiiy/char korpusi.
- **gpu_required: true** (Day 10). LEKIN mahalliy GPU yo'q: kaggle-hardware bo'yicha GPU — tezlatgich,
  talab emas; mahalliy CPU'da kichik korpus + kam epoch + qisqa generatsiya bilan ishlasin.

**QULFLANGAN birinchi assert (L9 [I2] → P9).** Notebook temperature softmax qadamini ochib ko'rsatsin
(toza-numpy, torch'siz) va assert aynan shuni tekshirsin:
- logitlar $z=[3.0, 1.0, 2.0]$, vocab $=[\text{nlp}, \text{foydali}, \text{qiziq}]$.
- $T=1$: $p(\text{nlp}) = e^3/(e^3+e^1+e^2) \approx 0.665$; $T=0.5$: $\approx 0.867$.
- **KUTILGAN: $T{=}1$ da $p(\text{nlp})\approx 0.665$**.
- `assert abs(p_nlp_T1 - 0.665) < 1e-3  # Ma'ruza L9 [I2]-slayd bilan solishtiring`.

**m09 shartnomasi (capstone/contracts.py — QAT'IY, AYNAN MOS):**
```
class TextGenerator:                       # pedagogik demo (consumed_by: [])
    train(text: str, epochs: int = 20, hidden_size: int = 128) -> None
    generate(seed: str, length: int = 200, temperature: float = 0.7) -> str
```
> DIQQAT: m09 **char-darajali** (train xom MATN qatorini oladi, token ro'yxati EMAS).
> save/load shartnomada YO'Q (pedagogik). generate temperature bilan autoregressiv sample qiladi.

## 5. ⚠️ TORCH-IXTIYORIY + Bi-LSTM NUANSI

- **Kaggle yo'li (`HAS_TORCH=True`):** char-LSTM (nn.Embedding/one-hot + `nn.LSTM(num_layers=2)` + Linear),
  next-char CrossEntropyLoss + Adam. generate autoregressiv + temperature.
- **Offline yo'l (torch'siz):** char-darajali **n-gram** model (m05 g'oyasi) + temperature sampling —
  generatsiya qiladi, temperature qo'llab-quvvatlaydi, mo'rt BPTT'siz, ishonchli. (Yoki numpy char-LSTM,
  lekin n-gram zaxira osonroq va barqaror.)
- **`HAS_TORCH` bayrog'i** yo'lni tanlaydi; locked temperature-softmax asserti har doim toza-numpy.
- Notebook **GPU'siz va uz_news_full siz** uchdan-uchgacha ishlasin; kam epoch, qisqa generatsiya (CPU).
- **Bi-LSTM nuansi (L9 da o'rgatildi!):** bidirectional model **generatsiya qila olmaydi** (kelajak kerak).
  Shuning uchun subitem 3–4 (Bi-LSTM) generatsiya emas, **tushunish/perplexity** taqqoslashi sifatida
  ko'rsatilsin (oddiy LSTM generatsiya qiladi; Bi-LSTM perplexity/representation uchun). Buni notebook'da
  aniq tushuntir. m09 moduli faqat generatsiyaga (train/generate) javobgar; Bi-LSTM taqqoslash notebook-darajali.

## 6. PRACTICE-NOTEBOOK TUZILISHI (skill, gold standard P8/P7 naqshi)

§1 Muhit (seeds, OFFLINE_FALLBACK, m01 yo'li ixtiyoriy, HAS_TORCH) → §2 yaxlit natija (tayyor generate demo)
→ §3 PRIMM periferiya (char korpus yuklash + char tokenizatsiya; next-char training loop — to'liq beriladi)
→ Checkpoint → §4 yadro: **so'nuvchi tayanch** (Namuna: locked temperature softmax 0.665 → Birgalikda
`# === SIZNING KODINGIZ ===` char-LSTM train + generate → Mustaqil: temperature 0.3/0.7/1.2 taqqoslash
+ Bi-LSTM perplexity), har blank → mos **assert** → §5 loyihaga ulash (m09 ni yozish, import test, git)
→ §6 tadqiqot + exit ticket.

## 7. SIFAT DARVOZALARI (MAHALLIY — kechiktirilmaydi)

- **JSON valid** (student + SOLUTIONS): nbformat 4.5; har katak `id`.
- **Uchdan-uchgacha bajariladi** (`OFFLINE_FALLBACK=True`, CPU): SOLUTIONS kataklari mahalliy ishga tushadi,
  **har assert o'tadi**, 0 istisno. Builder orqali exec.
- **Student stub kataklar `compile()` toza.**
- **Birinchi (qulflangan) assert:** `abs(p_nlp_T1 - 0.665) < 1e-3` — `# Ma'ruza L9 [I2]-slayd`.
- **Har blank region mos assert bilan;** m09 shartnoma mosligi (train/generate).
- **generate strukturaviy assert:** so'ralgan uzunlikdagi `str` qaytaradi; belgilar korpus lug'atidan;
  temperature 0.3/0.7/1.2 da ishlaydi (aniq matn EMAS — kichik korpusda generatsiya tasodifiy).
- **No GPU** mahalliy; seeds (random/np/torch 42); checkpoint katak(lar)i.
- **Terminologiya grep toza;** ASCII apostrof; U+2019 yo'q; **Kirill 0** (lotincha k/a/g/o o'rniga
  kirill tushib qolmasin — L9 da bo'lgan, kompilyatsiya/skan bilan ushlandi); BOM yo'q.

## 8. SO'RALADIGAN PROMPT

**P9 (9-amaliyot: m09 TextGenerator — char-darajali LSTM bilan matn generatsiya) ni ishlab chiqarish uchun
bosqichma-bosqich prompt** yozib bering —
- course_map Day 10 (practice 9) spetsifikatsiyasiga aniq mos: 4 subitem, periferiya (char korpus +
  char tokenizatsiya, next-char training loop), yadro (char-LSTM hidden=128 2 qatlam,
  temperature 0.3/0.7/1.2, Bi-LSTM perplexity);
- **qulflangan birinchi assert** = L9 [I2]: temperature softmax `T=1 p(nlp)=0.665` (toza-numpy,
  `# Ma'ruza L9 [I2]-slayd`);
- **m09 contracts.py imzosiga AYNAN mos** (train(text, epochs, hidden_size); generate(seed, length,
  temperature)→str; char-darajali; save/load YO'Q; pedagogik consumed_by []);
- **torch-ixtiyoriy**: Kaggle char-LSTM; offline char n-gram + temperature (mo'rt BPTT'siz);
  GPU'siz/uz_news_fullsiz uchdan-uchgacha; offline korpus `d10_checkpoints/`;
- **Bi-LSTM nuansi**: bidirectional generatsiya qila olmaydi (L9) → tushunish/perplexity taqqoslashi sifatida;
- practice-notebook tuzilishi (§1–§6, so'nuvchi tayanch, PRIMM);
- mahalliy darvozalar: JSON valid, SOLUTIONS CPU'da bajariladi (har assert o'tadi), terminologiya toza,
  ASCII, Kirill 0, seeds; generate strukturaviy assert (uzunlik/belgilar);
- 3 commit: `day10: practice — P9 …` / `day10: capstone — m09 …` / `day10: qa — P9 report`.

---

## Ilova — repozitoriy fayl holati

```
course/lectures/   : d01..d05, d06_word2vec, d07_rnn, d08_gru_lstm, d09_matn_generatsiya  (tex+pdf; d02 tex)
course/practices/  : d02_p1 ... d08_p7 d09_p8_gru_lstm  (+ _SOLUTIONS, + d0N_checkpoints/)
capstone/modules/  : m01..m08 (+ m05b)   (m09 — P9 da quriladi)
course/milestones/ : w1_*, w2_*
course/qa/         : d01, L1–L9, P1–P8, w1, w2 (+ skriptlar)
```

So'nggi commitlar:
```
96e01fc day09: lecture — L9 Matn generatsiya va ikki tomonlama RNN
57c1d08 day09: qa — P8 report (all gates PASS, 8/8 local asserts; torch+numpy)
fa4692d day09: capstone — m08 GRULSTMClassifier (nn.LSTM/nn.GRU, torch-ixtiyoriy)
706e1ac day09: practice — P8 gru_lstm notebook + SOLUTIONS
21f22ce day08: lecture — L8 Ilg'or RNN arxitekturalari (GRU va LSTM)
```
