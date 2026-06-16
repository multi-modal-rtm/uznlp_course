# NLP KURSI — LOYIHA HOLAT HISOBOTI

**Sana:** 2026-06-16 · **Branch:** `feat/help_to_improve` · **Oxirgi commit:** `b30ab3f`
**Remotelar:** `origin` (datascientistn1/uznlp_course), `rtm` (multi-modal-rtm/uznlp_course).
> Holat: ikkala remote ham **to'liq sinxron** (push qilingan, 0 ortda).

> Bu hisobot mustaqil: uni boshqa Claude (yoki hamkasb) ga berib, **P13 (13-amaliyot:
> m13 FineTunedClassifier — Hugging Face bilan BERT nozik sozlash)** uchun bosqichma-bosqich prompt olish mumkin.

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
- **Mahalliy vositalar:** Python 3.13 (numpy/sklearn/matplotlib/**torch 2.10 CPU**/**transformers BOR**).
  ⚠️ **datasets YO'Q**, sentencepiece YO'Q, gensim/datasketch/nltk/seqeval YO'Q.
- **Amaliyot uslubi:** `_build_pN.py` builder — offline data, modulni yozish, student+SOLUTIONS
  notebook'larni JSON qurish, SOLUTIONS kataklarini exec qilib assert tekshirish. Builder commit qilinmaydi.

## 3. BAJARILGAN ISHLAR

**Ma'ruzalar:** L1–L13 — …, d11_seq2seq_attention, d12_transformer, **d13_transfer_learning** (tex+pdf;
d02 faqat tex). Har biriga QA. ✅

**Amaliyotlar:** P1 (m01) … P12 (m12). Har biri +SOLUTIONS +checkpoints +QA, mahalliy bajarilgan. ✅

**Modullar:** m01–m12 (+ m05b). ✅  (m10/m12 — HAQIQIY modullar, save/load).

**Milestonelar:** w1 (m01+m02 ✅), w2 (m01–m05 ✅), w3 (neyron m01–m08, 18/18 ✅).

> Holat: ma'ruzalar **L13 gacha**, amaliyotlar P12 gacha, milestonelar w3 gacha. **Keyingi
> xronologik artefakt — P13** (Day 14 ertalab; L13 ga juft). So'ng L14 (Day 14 ma'ruza).

## 4. KEYINGI QADAM — P13 (so'ralgan)

> P13 = **course_map Day 14, `practice_official_no: 13`**; **L13** (Transfer Learning) ma'ruzasiga juft.
> Modul: **m13 FineTunedClassifier**, fayl `capstone/modules/m13_bert_classifier.py`.
> Notebook: `course/practices/d14_p13_finetune.ipynb` (+ `_SOLUTIONS` + `d14_checkpoints/`).

**P13 spetsifikatsiyasi (course_map Day 14):**
- **Mavzu:** "Hugging Face bilan nozik sozlash (fine-tuning) amaliyoti."
- **4 kichik bo'lim (practice_subitems):**
  1. HF `transformers` va `datasets` kutubxonalarini o'rnatish va tanishish.
  2. Oldindan o'qitilgan BERT modelini sentiment tahlili uchun fine-tune qilish.
  3. `Trainer API` yordamida o'qitish jarayonini soddalashtirish.
  4. Nozik sozlangan modelni baholash va pipeline bilan ishlatish.
- **Periferiya (to'liq beriladi — PRIMM):**
  - Uzum Market sharhlarini yuklash va binarizatsiya (reyting {4,5}→ijobiy, {1,2}→salbiy, 3 tashlanadi).
  - `AutoTokenizer` tokenizatsiya + `DataCollatorWithPadding`.
- **Yadro (tinglovchi yozadi):**
  - `AutoModelForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=2)`.
  - `TrainingArguments`: lr=2e-5, batch=16, epochs=3, warmup_steps=100.
  - `Trainer.train()` + `evaluate()`; `classification_report`.
- **VERIFIED korpus:** `risqaliyevds/uzbek-sentiment-analysis` (MIT ✓, LICENSES.md CONFIRMED).
  352k qator; amaliyotda 5000 subsample (2500 ijobiy + 2500 salbiy). **OFFLINE_FALLBACK:** `d14_checkpoints/`
  da kichik **original** uz sentiment korpus (ijobiy/salbiy yorliqli).
- **gpu_required: true** (Day 14). Mahalliy GPU yo'q: kaggle-hardware — GPU tezlatgich; mahalliy CPU'da fallback.

**QULFLANGAN birinchi assert (L13 [I1] → P13).** Notebook BCE yo'qotishini ochib ko'rsatsin (toza-torch,
HF SHART EMAS) va assert aynan shuni tekshirsin:
- logit $z=2.0$, yorliq $y=1$ (ijobiy). $p=\sigma(2.0)=1/(1+e^{-2.0})\approx 0.880$; $\mathrm{BCE}=-\ln(0.880)\approx 0.128$.
- **KUTILGAN: $\sigma(2.0)\approx 0.880$; BCE $\approx 0.128$.**
- `loss = nn.BCEWithLogitsLoss()(torch.tensor([2.0]), torch.tensor([1.0]))`
  `assert abs(float(loss) - 0.128) < 1e-2  # Ma'ruza L13 [I1]-slayd bilan solishtiring`
  ($\sigma(2.0)\approx 0.880$ ham tekshirilsin).

**m13 shartnomasi (capstone/contracts.py — QAT'IY, AYNAN MOS):**
```
class FineTunedClassifier:                     # ⚠️ HAQIQIY modul — consumed_by: m15 (P15), app.py (M4 P16)
    fit(texts: list[str], labels: list[str], model_name="distilbert-base-multilingual-cased",
        epochs=3, batch_size=16, lr=2e-5) -> None
    predict(text: str) -> str                  # "ijobiy" yoki "salbiy"
    predict_proba(text: str) -> dict[str, float]   # {"ijobiy": 0.87, "salbiy": 0.13}
    save(path: str) -> None    /    load(path: str) -> None
```
> ⚠️ DIQQAT: m13 **HAQIQIY** modul (m10/m12 kabi). save/load BOR. Yorliqlar **QULFLANGAN** `ijobiy`/`salbiy`
> (musbat/manfiy EMAS — L2 [I2] bilan bog'langan). predict_proba dict kalitlari aynan `ijobiy`/`salbiy`.
> predict aniqligi past bo'lishi mumkin (kichik data) — strukturaviy assert ({ijobiy,salbiy}, [0,1] dict).

## 5. ⚠️ transformers-IXTIYORIY + datasets-IXTIYORIY DIZAYN (m02 LogReg fallback)

> ⚠️ MUHIM: transformers MAHALLIY BOR, LEKIN haqiqiy DistilBERT fine-tuning model yuklab olish (internet,
> ~500 MB) + sekin CPU o'qitish talab qiladi. Shuning uchun **mahalliy tekshirish FALLBACK orqali**.
- **Kaggle yo'li (`HAS_TRANSFORMERS=True`, GPU+internet):** `AutoModelForSequenceClassification` DistilBERT
  + `Trainer` API bilan fine-tune; `predict`/`predict_proba` HF pipeline orqali.
- **Offline yo'l (transformers'siz YOKI OFFLINE_FALLBACK):** **TF-IDF + sklearn LogisticRegression** sentiment
  klassifikatori (m02 naqshi, isbotlangan, tez, yuklab olishsiz). predict→ijobiy/salbiy, predict_proba→dict.
- **`HAS_TRANSFORMERS`/`USE_TRANSFORMERS` bayrog'i** yo'lni tanlaydi; builder mahalliy `USE_TRANSFORMERS=False`
  ga majburlaydi (m12 da `HAS_TORCH=False` ga o'xshab) — internetsiz, tez, deterministik.
- **datasets YO'Q** → Uzum korpusini `datasets.load_dataset` o'rniga offline CSV/txt dan yuklash (fallback).
- **locked BCE asserti HAR DOIM toza-torch** (path-independent; torch mahalliy bor).
- Notebook GPU'siz/transformers'siz/datasets'siz uchdan-uchgacha ishlasin; fine-tuning kodi ko'rsatiladi,
  mahalliyda LogReg fallback bajariladi. Aniqlik demo-sifatli, halol.

## 6. PRACTICE-NOTEBOOK TUZILISHI (skill, gold standard P12/P10 naqshi)

§1 Muhit (seeds, OFFLINE_FALLBACK, HAS_TORCH, HAS_TRANSFORMERS, HAS_DATASETS) → §2 yaxlit natija (tayyor
predict + ehtimol demosi) → §3 PRIMM periferiya (Uzum sharhlar yuklash + binarizatsiya {4,5}→ijobiy/{1,2}→salbiy;
AutoTokenizer + DataCollator — to'liq beriladi, HF guard bilan) → Checkpoint → §4 yadro: **so'nuvchi tayanch**
(Namuna: locked BCE [0.880, 0.128] toza-torch → Birgalikda `# === SIZNING KODINGIZ ===` AutoModelForSeqClass +
TrainingArguments + Trainer qurish (HF; mahalliyda fallback) → Mustaqil: predict + predict_proba + baholash),
har blank → mos **assert** → §5 loyihaga ulash (m13 ni yozish, import test, save/load test, git)
→ §6 tadqiqot + exit ticket (mBERT WordPiece vs natija; ijobiy/salbiy chegara).

## 7. SIFAT DARVOZALARI (MAHALLIY — kechiktirilmaydi)

- **JSON valid** (student + SOLUTIONS): nbformat 4.5; har katak `id`.
- **Uchdan-uchgacha bajariladi** (`OFFLINE_FALLBACK=True`, CPU): SOLUTIONS kataklari mahalliy ishga tushadi,
  **har assert o'tadi**, 0 istisno. Builder orqali exec. Mahalliy = LogReg fallback + toza-torch BCE.
- **Student stub kataklar `compile()` toza.**
- **Birinchi (qulflangan) assert:** `abs(BCE - 0.128) < 1e-2` — `# Ma'ruza L13 [I1]-slayd` (+ σ(2.0)≈0.880).
- **Har blank region mos assert bilan;** m13 shartnoma mosligi (fit/predict/predict_proba/save/load).
- **predict/predict_proba strukturaviy assert:** predict ∈ {`ijobiy`,`salbiy`}; predict_proba dict
  (kalitlar aynan `ijobiy`/`salbiy`, har biri `float` $\in[0,1]$, yig'indi $\approx 1$).
  (Aniq tasnif/yuqori F1 EMAS — kichik data, demo-sifatli, halol.)
- **save/load tekshiruvi:** m13 consumed_by m15/app.py — save→load→predict ishlashini sinab ko'r.
- **No GPU** mahalliy; seeds (random/np/torch 42); checkpoint katak(lar)i.
- **Terminologiya grep toza;** **yorliqlar `ijobiy`/`salbiy`** (musbat/manfiy YO'Q); ASCII apostrof;
  U+2019 yo'q; **Kirill 0** (notebook'da ham skan); BOM yo'q.

## 8. SO'RALADIGAN PROMPT

**P13 (13-amaliyot: m13 FineTunedClassifier — Hugging Face bilan BERT nozik sozlash) ni ishlab chiqarish uchun
bosqichma-bosqich prompt** yozib bering —
- course_map Day 14 (practice 13) spetsifikatsiyasiga aniq mos: 4 subitem, periferiya (Uzum sharhlar yuklash +
  binarizatsiya, AutoTokenizer + DataCollator), yadro (AutoModelForSequenceClassification + TrainingArguments +
  Trainer.train()/evaluate(), classification_report);
- **qulflangan birinchi assert** = L13 [I1]: BCE — `σ(2.0)≈0.880`, `BCE≈0.128` (toza-torch, `# Ma'ruza L13 [I1]-slayd`);
- **m13 contracts.py imzosiga AYNAN mos** (fit(texts,labels,model_name,epochs,batch_size,lr); predict→str
  `ijobiy`/`salbiy`; predict_proba→dict {`ijobiy`,`salbiy`}; **save/load BOR**; ⚠️ HAQIQIY modul — consumed_by m15/app.py);
- **transformers-ixtiyoriy**: Kaggle DistilBERT + Trainer fine-tune; offline **TF-IDF + sklearn LogisticRegression**
  fallback (m02 naqshi; mahalliy = fallback, chunki HF model yuklab olish + sekin CPU); **datasets-ixtiyoriy**
  (offline CSV/txt); GPU'siz/transformers'siz uchdan-uchgacha; offline = `d14_checkpoints/` kichik original uz sentiment;
- **yorliqlar QULFLANGAN `ijobiy`/`salbiy`** (musbat/manfiy EMAS); predict aniqligi past KUTILGAN (kichik data) —
  halol; strukturaviy assert (predict ∈ {ijobiy,salbiy}; predict_proba dict, [0,1], yig'indi≈1);
- practice-notebook tuzilishi (§1–§6, so'nuvchi tayanch, PRIMM);
- mahalliy darvozalar: JSON valid, SOLUTIONS CPU'da bajariladi (har assert o'tadi), terminologiya toza,
  ASCII, Kirill 0, seeds; **save/load test**; predict/predict_proba strukturaviy assert;
- 3 commit: `day14: practice — P13 …` / `day14: capstone — m13 …` / `day14: qa — P13 report`.

---

## Ilova — repozitoriy fayl holati

```
course/lectures/   : d01..d05, d06..d12, d13_transfer_learning  (tex+pdf; d02 tex)
course/practices/  : d02_p1 ... d13_p12_transformer  (+ _SOLUTIONS, + d0N_checkpoints/)
capstone/modules/  : m01..m12 (+ m05b)   (m13 — P13 da quriladi)
course/milestones/ : w1_*, w2_*, w3_*
course/qa/         : d01, L1–L13, P1–P12, w1, w2, w3 (+ skriptlar)
```

So'nggi commitlar:
```
b30ab3f day13: lecture — L13 Transfer Learning va oldindan o'qitilgan modellar (BERT, T5)
ce1bf2a docs: HOLAT_HISOBOT.md — P12 ga yangilandi (L12 yopildi, m12 keyingi)
10e43f3 day13: qa — P12 report (all gates PASS, 13/13 local asserts; torch+ekstraktiv)
58c2a21 day13: capstone — m12 TransformerSummarizer (nn.Transformer + ekstraktiv fallback, torch-ixtiyoriy)
6d213c6 day13: practice — P12 transformer notebook + SOLUTIONS
```
```
origin/rtm = b30ab3f (to'liq sinxron, 0 ortda)
```
