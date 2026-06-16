# NLP KURSI — LOYIHA HOLAT HISOBOTI

**Sana:** 2026-06-16 · **Branch:** `feat/help_to_improve` · **Oxirgi commit:** `0e9da4b`
**Remotelar:** `origin` (datascientistn1/uznlp_course), `rtm` (multi-modal-rtm/uznlp_course).
> Holat: ikkala remote ham **to'liq sinxron** (push qilingan, 0 ortda).

> Bu hisobot mustaqil: uni boshqa Claude (yoki hamkasb) ga berib, **P11 (11-amaliyot:
> m11 Seq2SeqTranslator — attention'li neyro-tarjimon)** uchun bosqichma-bosqich prompt olish mumkin.

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
- **Mahalliy vositalar:** Python 3.13 (numpy/sklearn/matplotlib/**torch 2.10 CPU**). gensim/datasketch/nltk/seqeval YO'Q.
- **Amaliyot uslubi:** `_build_pN.py` builder — offline data, modulni yozish, student+SOLUTIONS
  notebook'larni JSON qurish, SOLUTIONS kataklarini exec qilib assert tekshirish. Builder commit qilinmaydi.

## 3. BAJARILGAN ISHLAR

**Ma'ruzalar:** L1–L11 — d01…d05, d06_word2vec, d07_rnn, d08_gru_lstm, d09_matn_generatsiya,
d10_ner, **d11_seq2seq_attention YANGI** (tex+pdf; d02 faqat tex). Har biriga QA. ✅

**Amaliyotlar:** P1 (m01), P2 (m02), P3 (m03), P4 (m04), P5 (m05+m05b), P6 (m06), P7 (m07),
P8 (m08), P9 (m09), P10 (m10). Har biri +SOLUTIONS +checkpoints +QA, mahalliy bajarilgan. ✅

**Modullar:** m01–m10 (+ m05b). ✅

**Milestonelar:** w1 (m01+m02 ✅), w2 (m01–m05 ✅), w3 (neyron m01–m08, 18/18 ✅).

> Holat: ma'ruzalar **L11 gacha**, amaliyotlar P10 gacha, milestonelar w3 gacha. **Keyingi
> xronologik artefakt — P11** (Day 12 ertalab; L11 ga juft). So'ng L12 (Day 12 ma'ruza — Transformer).

## 4. KEYINGI QADAM — P11 (so'ralgan)

> P11 = **course_map Day 12, `practice_official_no: 11`**; **L11** (Seq2Seq/Attention) ma'ruzasiga juft.
> Modul: **m11 Seq2SeqTranslator**, fayl `capstone/modules/m11_seq2seq_translator.py`.
> Notebook: `course/practices/d12_p11_seq2seq.ipynb` (+ `_SOLUTIONS` + `d12_checkpoints/`).

**P11 spetsifikatsiyasi (course_map Day 12):**
- **Mavzu:** "Attention mexanizmli neyro-tarjimon."
- **4 kichik bo'lim (practice_subitems):**
  1. O'zbekcha-inglizcha parallel korpus tayyorlash.
  2. Enkoder (LSTM) va Dekoder (LSTM) dan iborat Seq2Seq modelini qurish.
  3. Attention qatlami bilan attention mexanizmini modelga qo'shish.
  4. Modelni o'qitish va sodda gaplarni tarjima qilib, BLEU bilan baholash.
- **Periferiya (to'liq beriladi — PRIMM):**
  - OPUS-100 uz-en subset (20k juft) yuklash va tokenizatsiya.
  - BLEU hisoblash: `nltk.translate.bleu_score`.
- **Yadro (tinglovchi yozadi):**
  - Encoder LSTM + Bahdanau Attention + Decoder LSTM.
  - Teacher forcing bilan o'qitish.
  - Oddiy gaplarni tarjima + attention heatmap vizualizatsiyasi.
- **corpus_subset:** uz_en_opus100 (onlayn, 20k juft, demo-sifatli BLEU). **OFFLINE_FALLBACK:**
  `d12_checkpoints/` da kichik **original** uz-en parallel korpus.
- **gpu_required: true** (Day 12). LEKIN mahalliy GPU yo'q: kaggle-hardware bo'yicha GPU — tezlatgich,
  talab emas; mahalliy CPU'da kichik korpus + kam epoch bilan ishlasin.

**QULFLANGAN birinchi assert (L11 [I3] → P11).** Notebook attention softmax ni ochib ko'rsatsin
(toza-numpy, torch'siz) va assert aynan shuni tekshirsin:
- energiyalar $e=[2.0, 1.0, 3.0]$; $\alpha = \mathrm{softmax}(e)$.
- **KUTILGAN: $\alpha \approx [0.245, 0.090, 0.665]$** (eng katta energiya, 3-pozitsiya, eng katta vazn).
- `assert np.allclose(alpha, [0.245, 0.090, 0.665], atol=1e-3)  # Ma'ruza L11 [I3]-slayd`.
- DIQQAT: bu attention softmax (alignment vaznlari), L9 temperature emas — bir xil matematik, boshqa ma'no.

**m11 shartnomasi (capstone/contracts.py — QAT'IY, AYNAN MOS):**
```
class Seq2SeqTranslator:                       # pedagogik demo (consumed_by: [])
    train(src_texts: list[str], tgt_texts: list[str], epochs=10, max_len=50) -> None
    translate(text: str) -> str
    bleu(references: list[list[str]], hypotheses: list[str]) -> float
```
> DIQQAT: m11 **pedagogik** (m09 kabi) — save/load YO'Q; consumed_by []. translate gibberish bo'lishi
> mumkin (kichik data) — strukturaviy assert (str qaytaradi), aniq tarjima EMAS.

## 5. ⚠️ TORCH-IXTIYORIY + nltk-IXTIYORIY DIZAYN

- **Kaggle yo'li (`HAS_TORCH=True`):** LSTM enkoder + **Bahdanau attention** + LSTM dekoder,
  teacher forcing + Adam; greedy decode bilan translate.
- **Offline yo'l (torch'siz):** to'liq attention-seq2seq numpy BPTT JUDA og'ir/mo'rt — buning o'rniga
  **soddalashtirilgan lug'at-asosli tarjimon** (so'z-tekislash: manba↔maqsad ko'p uchragan juftlik) +
  attention matematikasi alohida numpy demosi sifatida. translate so'zma-so'z lug'at orqali ishlaydi.
- **`HAS_TORCH` bayrog'i** yo'lni tanlaydi; locked attention-softmax asserti har doim toza-numpy.
- **bleu() — toza-python** (n-gram aniqlik + brevity penalty); `nltk` bor bo'lsa undan, aks holda o'zimizniki
  (`HAS_NLTK` bayrog'i). Mahalliyda nltk YO'Q → toza-python BLEU.
- Notebook **GPU'siz/torch'siz/OPUS-100siz** uchdan-uchgacha ishlasin; kam epoch (CPU). BLEU demo-sifatli, halol.

## 6. PRACTICE-NOTEBOOK TUZILISHI (skill, gold standard P10/P9 naqshi)

§1 Muhit (seeds, OFFLINE_FALLBACK, HAS_TORCH, HAS_NLTK) → §2 yaxlit natija (tayyor translate + BLEU demo)
→ §3 PRIMM periferiya (uz-en parallel korpus yuklash + tokenizatsiya; BLEU hisoblash — to'liq beriladi)
→ Checkpoint → §4 yadro: **so'nuvchi tayanch** (Namuna: locked attention softmax [0.245,0.090,0.665]
→ Birgalikda `# === SIZNING KODINGIZ ===` Seq2Seq+attention qurish va o'qitish → Mustaqil: translate +
BLEU + attention heatmap), har blank → mos **assert** → §5 loyihaga ulash (m11 ni yozish, import test, git)
→ §6 tadqiqot + exit ticket.

## 7. SIFAT DARVOZALARI (MAHALLIY — kechiktirilmaydi)

- **JSON valid** (student + SOLUTIONS): nbformat 4.5; har katak `id`.
- **Uchdan-uchgacha bajariladi** (`OFFLINE_FALLBACK=True`, CPU): SOLUTIONS kataklari mahalliy ishga tushadi,
  **har assert o'tadi**, 0 istisno. Builder orqali exec.
- **Student stub kataklar `compile()` toza.**
- **Birinchi (qulflangan) assert:** `np.allclose(alpha, [0.245,0.090,0.665], atol=1e-3)` — `# Ma'ruza L11 [I3]-slayd`.
- **Har blank region mos assert bilan;** m11 shartnoma mosligi (train/translate/bleu).
- **translate/bleu strukturaviy assert:** translate `str` qaytaradi; bleu `float` $\in[0,1]$.
  (Aniq tarjima/yuqori BLEU EMAS — kichik data, demo-sifatli, halol.)
- **No GPU** mahalliy; seeds (random/np/torch 42); checkpoint katak(lar)i.
- **Terminologiya grep toza;** ASCII apostrof; U+2019 yo'q; **Kirill 0** (notebook'da ham skan); BOM yo'q.

## 8. SO'RALADIGAN PROMPT

**P11 (11-amaliyot: m11 Seq2SeqTranslator — attention'li neyro-tarjimon) ni ishlab chiqarish uchun
bosqichma-bosqich prompt** yozib bering —
- course_map Day 12 (practice 11) spetsifikatsiyasiga aniq mos: 4 subitem, periferiya (OPUS-100/offline
  uz-en yuklash + tokenizatsiya, BLEU), yadro (LSTM enkoder + Bahdanau attention + LSTM dekoder, teacher
  forcing, tarjima + attention heatmap);
- **qulflangan birinchi assert** = L11 [I3]: attention `alpha = softmax([2,1,3]) = [0.245,0.090,0.665]`
  (toza-numpy, `# Ma'ruza L11 [I3]-slayd`);
- **m11 contracts.py imzosiga AYNAN mos** (train(src,tgt,epochs,max_len); translate→str; bleu→float;
  pedagogik consumed_by []; save/load YO'Q);
- **torch-ixtiyoriy**: Kaggle LSTM+attention seq2seq; offline soddalashtirilgan lug'at-asosli tarjimon +
  attention numpy demosi; **nltk-ixtiyoriy** BLEU (toza-python fallback); GPU'siz/torch'siz uchdan-uchgacha;
  offline = `d12_checkpoints/` kichik original uz-en parallel korpus;
- translate gibberish/BLEU past KUTILGAN (kichik data, demo-sifatli) — halol; strukturaviy assert;
- practice-notebook tuzilishi (§1–§6, so'nuvchi tayanch, PRIMM);
- mahalliy darvozalar: JSON valid, SOLUTIONS CPU'da bajariladi (har assert o'tadi), terminologiya toza,
  ASCII, Kirill 0, seeds; translate/bleu strukturaviy assert;
- 3 commit: `day12: practice — P11 …` / `day12: capstone — m11 …` / `day12: qa — P11 report`.

---

## Ilova — repozitoriy fayl holati

```
course/lectures/   : d01..d05, d06_word2vec, d07_rnn, d08_gru_lstm, d09_matn_generatsiya, d10_ner, d11_seq2seq_attention  (tex+pdf; d02 tex)
course/practices/  : d02_p1 ... d11_p10_ner  (+ _SOLUTIONS, + d0N_checkpoints/)
capstone/modules/  : m01..m10 (+ m05b)   (m11 — P11 da quriladi)
course/milestones/ : w1_*, w2_*, w3_*
course/qa/         : d01, L1–L11, P1–P10, w1, w2, w3 (+ skriptlar)
```

So'nggi commitlar:
```
0e9da4b day11: lecture — L11 Neyron mashina tarjimasi (Seq2Seq va Attention)
7b9abc6 day11: qa — P10 report (all gates PASS, 10/10 local asserts; torch+reservoir)
6d44903 day11: capstone — m10 NERTagger (Bi-LSTM IOB2, torch-ixtiyoriy)
4ff2ad0 day11: practice — P10 ner notebook + SOLUTIONS
146f9a6 w3: qa — w3 report (all gates PASS, 18/18 local checks)
```
