# NLP KURSI — LOYIHA HOLAT HISOBOTI

**Sana:** 2026-06-16 · **Branch:** `feat/help_to_improve` · **Oxirgi commit:** `f5ee38b`
**Remotelar:** `origin` (datascientistn1/uznlp_course), `rtm` (multi-modal-rtm/uznlp_course).
> Holat: ikkala remote ham **to'liq sinxron** (push qilingan, 0 ortda).

> Bu hisobot mustaqil: uni boshqa Claude (yoki hamkasb) ga berib, **P12 (12-amaliyot:
> m12 TransformerSummarizer — Transformer bilan matn umumlashtirgich)** uchun bosqichma-bosqich prompt olish mumkin.

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
- **Mahalliy vositalar:** Python 3.13 (numpy/sklearn/matplotlib/**torch 2.10 CPU**). gensim/datasketch/nltk/seqeval/sentencepiece YO'Q.
- **Amaliyot uslubi:** `_build_pN.py` builder — offline data, modulni yozish, student+SOLUTIONS
  notebook'larni JSON qurish, SOLUTIONS kataklarini exec qilib assert tekshirish. Builder commit qilinmaydi.

## 3. BAJARILGAN ISHLAR

**Ma'ruzalar:** L1–L12 — d01…d05, d06_word2vec, d07_rnn, d08_gru_lstm, d09_matn_generatsiya,
d10_ner, d11_seq2seq_attention, **d12_transformer YANGI** (tex+pdf; d02 faqat tex). Har biriga QA. ✅

**Amaliyotlar:** P1 (m01), P2 (m02), P3 (m03), P4 (m04), P5 (m05+m05b), P6 (m06), P7 (m07),
P8 (m08), P9 (m09), P10 (m10), P11 (m11). Har biri +SOLUTIONS +checkpoints +QA, mahalliy bajarilgan. ✅

**Modullar:** m01–m11 (+ m05b). ✅

**Milestonelar:** w1 (m01+m02 ✅), w2 (m01–m05 ✅), w3 (neyron m01–m08, 18/18 ✅).

> Holat: ma'ruzalar **L12 gacha**, amaliyotlar P11 gacha, milestonelar w3 gacha. **Keyingi
> xronologik artefakt — P12** (Day 13 ertalab; L12 ga juft). So'ng L13 (Day 13 ma'ruza — Transfer Learning, BERT/T5).

## 4. KEYINGI QADAM — P12 (so'ralgan)

> P12 = **course_map Day 13, `practice_official_no: 12`**; **L12** (Transformer) ma'ruzasiga juft.
> Modul: **m12 TransformerSummarizer**, fayl `capstone/modules/m12_transformer_summarizer.py`.
> Notebook: `course/practices/d13_p12_transformer.ipynb` (+ `_SOLUTIONS` + `d13_checkpoints/`).

**P12 spetsifikatsiyasi (course_map Day 13):**
- **Mavzu:** "Transformer arxitekturasi yordamida matnlardan xulosalar chiqarish."
- **4 kichik bo'lim (practice_subitems):**
  1. Matn va uning qisqa mazmuni juftliklaridan iborat ma'lumotlar to'plamini tayyorlash.
  2. Transformerning Enkoder-Dekoder arxitekturasini qurish (PyTorch).
  3. Self-Attention va Positional Encoding qatlamlarini implementatsiya qilish.
  4. Modelni o'qitish va yangi matnlar uchun qisqa mazmun generatsiya qilish.
- **Periferiya (to'liq beriladi — PRIMM):**
  - Wikipedia uz lead-sentence juftlarini yuklash.
  - Custom BPE tokenizer (sentencepiece, vocab=5000) — sentencepiece YO'Q bo'lsa fallback (so'z/char).
  - Encoder-Decoder scaffold: layer norm, FFN, residual.
- **Yadro (tinglovchi yozadi):**
  - `ScaledDotProductAttention` + `MultiHeadAttention` implementatsiyasi.
  - `PositionalEncoding` (sinusoidal) implementatsiyasi.
  - Mini-training loop (10 epoch); ROUGE-1 baholash.
- **Kaggle-hardware:** d_model=128, 4 head, 2 layer, vocab=5000, T4 16GB. **gpu_required: true** (Day 13).
  LEKIN mahalliy GPU yo'q: kaggle-hardware bo'yicha GPU — tezlatgich; mahalliy CPU'da kichik korpus + kam epoch.
- **corpus_subset:** uz_wiki_summ (Wikipedia uz lead-paragraph, CC-BY-SA, ~2000 juft). **OFFLINE_FALLBACK:**
  `d13_checkpoints/` da kichik **original** uz maqola-xulosa parallel korpus.

**QULFLANGAN birinchi assert (L12 [I4] → P12).** Notebook ROUGE-1 ni ochib ko'rsatsin (toza-python,
torch'siz) va assert aynan shuni tekshirsin:
- Reference = `nlp juda qiziq va foydali` (5 token); Hypothesis = `nlp juda foydali` (3 token).
- Mos unigramlar: nlp, juda, foydali (3 ta). **P=3/3=1.000; R=3/5=0.600; F1=2·1·0.6/1.6=0.750.**
- **KUTILGAN: ROUGE-1 P=1.000, R=0.600, F1=0.750.**
- `r = sum.rouge1(["nlp juda qiziq va foydali"], ["nlp juda foydali"])`
  `assert abs(r["f1"] - 0.750) < 1e-3  # Ma'ruza L12 [I4]-slayd bilan solishtiring`
  (P=1.000, R=0.600 ham tekshirilsin).

**m12 shartnomasi (capstone/contracts.py — QAT'IY, AYNAN MOS):**
```
class TransformerSummarizer:                  # ⚠️ PEDAGOGIK EMAS — consumed_by: [15, 16]
    train(src_texts: list[str], tgt_texts: list[str], epochs=10, d_model=128, nhead=4) -> None
    summarize(text: str, max_length=60) -> str
    rouge1(references: list[str], hypotheses: list[str]) -> dict[str, float]   # {"precision","recall","f1"}
    save(path: str) -> None
    load(path: str) -> None
```
> ⚠️ DIQQAT: m12 **PEDAGOGIK EMAS** (m09/m11 dan farqli). consumed_by [15, 16] — m15 agent (summarize_text
> tool) va Day 16 pipeline ishlatadi. **save/load BOR.** `rouge1` **dict** qaytaradi ({"precision","recall","f1"}),
> float EMAS. summarize gibberish bo'lishi mumkin (kichik data) — strukturaviy assert (str qaytaradi), aniq
> xulosa EMAS; rouge1 dict kalitlari va [0,1] diapazoni tekshiriladi.

## 5. ⚠️ TORCH-IXTIYORIY + sentencepiece-IXTIYORIY DIZAYN

- **Kaggle yo'li (`HAS_TORCH=True`):** mini Transformer enkoder-dekoder — `MultiheadAttention` (yoki qo'lda
  ScaledDotProduct + MultiHead) + sinusoidal `PositionalEncoding` + residual/LayerNorm/FFN, teacher forcing + Adam;
  greedy decode bilan summarize.
- **Offline yo'l (torch'siz):** to'liq Transformer numpy BPTT JUDA og'ir/mo'rt — buning o'rniga
  **soddalashtirilgan EKSTRAKTIV xulosalagich** (lead-sentence / chastota-asosli gap tanlash, max_length gacha).
  Self-attention/PE matematikasi alohida **numpy forward demosi** sifatida ko'rsatiladi (o'qitishsiz: shakl +
  softmax yig'indisi=1 tekshiriladi). summarize ekstraktiv yo'l bilan ishlaydi.
- **`HAS_TORCH` bayrog'i** yo'lni tanlaydi; locked ROUGE-1 asserti har doim toza-python (path-independent).
- **sentencepiece YO'Q** (mahalliy) → BPE fallback: oddiy so'z (whitespace) yoki char tokenizer (`HAS_SP` bayrog'i).
- **rouge1() — toza-python** (unigram Counter clipping + P/R/F1 dict). Notebook GPU'siz/torch'siz/sentencepiece'siz
  uchdan-uchgacha ishlasin; kam epoch (CPU). ROUGE demo-sifatli, halol.

## 6. PRACTICE-NOTEBOOK TUZILISHI (skill, gold standard P11/P10 naqshi)

§1 Muhit (seeds, OFFLINE_FALLBACK, HAS_TORCH, HAS_SP) → §2 yaxlit natija (tayyor summarize + ROUGE demo)
→ §3 PRIMM periferiya (uz maqola-xulosa juftlari yuklash + tokenizatsiya; encoder-decoder scaffold — to'liq beriladi)
→ Checkpoint → §4 yadro: **so'nuvchi tayanch** (Namuna: locked ROUGE-1 [1.000,0.600,0.750] + PE/attention numpy
demosi → Birgalikda `# === SIZNING KODINGIZ ===` ScaledDotProduct/MultiHead/PositionalEncoding qurish va
o'qitish → Mustaqil: summarize + ROUGE-1 baholash), har blank → mos **assert** → §5 loyihaga ulash (m12 ni yozish,
import test, save/load test, git) → §6 tadqiqot + exit ticket.

## 7. SIFAT DARVOZALARI (MAHALLIY — kechiktirilmaydi)

- **JSON valid** (student + SOLUTIONS): nbformat 4.5; har katak `id`.
- **Uchdan-uchgacha bajariladi** (`OFFLINE_FALLBACK=True`, CPU): SOLUTIONS kataklari mahalliy ishga tushadi,
  **har assert o'tadi**, 0 istisno. Builder orqali exec. torch BOR → haqiqiy Transformer yo'li ham tekshiriladi.
- **Student stub kataklar `compile()` toza.**
- **Birinchi (qulflangan) assert:** `abs(rouge1(...)["f1"] - 0.750) < 1e-3` — `# Ma'ruza L12 [I4]-slayd`
  (+ precision=1.000, recall=0.600).
- **Har blank region mos assert bilan;** m12 shartnoma mosligi (train/summarize/rouge1/save/load).
- **summarize/rouge1 strukturaviy assert:** summarize `str` qaytaradi; rouge1 `dict` ({"precision","recall","f1"},
  har biri `float` $\in[0,1]$). PE/attention numpy: shakl + softmax yig'indisi $\approx 1$.
  (Aniq xulosa/yuqori ROUGE EMAS — kichik data, demo-sifatli, halol.)
- **save/load tekshiruvi:** m12 consumed_by [15,16] — save→load→summarize ishlashini sinab ko'r.
- **No GPU** mahalliy; seeds (random/np/torch 42); checkpoint katak(lar)i.
- **Terminologiya grep toza;** ASCII apostrof; U+2019 yo'q; **Kirill 0** (notebook'da ham skan); BOM yo'q.

## 8. SO'RALADIGAN PROMPT

**P12 (12-amaliyot: m12 TransformerSummarizer — Transformer bilan matn umumlashtirgich) ni ishlab chiqarish uchun
bosqichma-bosqich prompt** yozib bering —
- course_map Day 13 (practice 12) spetsifikatsiyasiga aniq mos: 4 subitem, periferiya (Wikipedia uz/offline
  maqola-xulosa yuklash + tokenizatsiya, encoder-decoder scaffold), yadro (ScaledDotProductAttention,
  MultiHeadAttention, PositionalEncoding, mini-training loop, summarize + ROUGE-1);
- **qulflangan birinchi assert** = L12 [I4]: ROUGE-1 `P=1.000, R=0.600, F1=0.750` (Ref `nlp juda qiziq va foydali`,
  Hyp `nlp juda foydali`), toza-python, `# Ma'ruza L12 [I4]-slayd`;
- **m12 contracts.py imzosiga AYNAN mos** (train(src,tgt,epochs,d_model,nhead); summarize(text,max_length)→str;
  rouge1(refs,hyps)→**dict** {"precision","recall","f1"}; **save/load BOR**; ⚠️ PEDAGOGIK EMAS — consumed_by [15,16]);
- **torch-ixtiyoriy**: Kaggle mini Transformer enkoder-dekoder (MultiHead+PE+residual/LayerNorm/FFN); offline
  soddalashtirilgan ekstraktiv xulosalagich + self-attention/PE numpy forward demosi; **sentencepiece-ixtiyoriy**
  BPE (so'z/char fallback); GPU'siz/torch'siz/sentencepiece'siz uchdan-uchgacha; offline = `d13_checkpoints/`
  kichik original uz maqola-xulosa korpus;
- summarize gibberish/ROUGE past KUTILGAN (kichik data, demo-sifatli) — halol; strukturaviy assert; PE/attention
  numpy demosi shakl + softmax yig'indisi=1 tekshiradi;
- practice-notebook tuzilishi (§1–§6, so'nuvchi tayanch, PRIMM);
- mahalliy darvozalar: JSON valid, SOLUTIONS CPU'da bajariladi (har assert o'tadi), terminologiya toza,
  ASCII, Kirill 0, seeds; summarize/rouge1 strukturaviy assert; **save/load test** (consumed_by [15,16]);
- 3 commit: `day13: practice — P12 …` / `day13: capstone — m12 …` / `day13: qa — P12 report`.

---

## Ilova — repozitoriy fayl holati

```
course/lectures/   : d01..d05, d06_word2vec, d07_rnn, d08_gru_lstm, d09_matn_generatsiya, d10_ner, d11_seq2seq_attention, d12_transformer  (tex+pdf; d02 tex)
course/practices/  : d02_p1 ... d12_p11_seq2seq  (+ _SOLUTIONS, + d0N_checkpoints/)
capstone/modules/  : m01..m11 (+ m05b)   (m12 — P12 da quriladi)
course/milestones/ : w1_*, w2_*, w3_*
course/qa/         : d01, L1–L12, P1–P11, w1, w2, w3 (+ skriptlar)
```

So'nggi commitlar:
```
f5ee38b docs: HOLAT_HISOBOT.md — L12 ga yangilandi (Transformer ma'ruzasi maqsadi)
02aa30f day12: lecture — L12 Transformer arxitekturasi va matnni umumlashtirish
674efa1 docs: HOLAT_HISOBOT.md — P11 ga yangilandi (L11 yopildi, m11 keyingi)
b467b72 day12: qa — P11 report (all gates PASS, 10/10 local asserts; torch+lug'at)
2bbe0e6 day12: capstone — m11 Seq2SeqTranslator (LSTM+Bahdanau attention, torch-ixtiyoriy)
```
