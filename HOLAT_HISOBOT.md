# NLP KURSI — LOYIHA HOLAT HISOBOTI

**Sana:** 2026-06-16 · **Branch:** `feat/help_to_improve` · **Oxirgi commit:** `674efa1`
**Remotelar:** `origin` (datascientistn1/uznlp_course), `rtm` (multi-modal-rtm/uznlp_course).
> Holat: ikkala remote ham **to'liq sinxron** (push qilingan, 0 ortda).

> Bu hisobot mustaqil: uni boshqa Claude (yoki hamkasb) ga berib, **L12 (12-ma'ruza:
> Transformer arxitekturasi va matnni umumlashtirish)** uchun bosqichma-bosqich prompt olish mumkin.

---

## 1. Loyiha nima

O'zbek tilidagi (lotin yozuvi) 4 haftalik intensiv NLP kursining BARCHA o'quv
materiallarini ishlab chiqarish loyihasi (15-iyun – 10-iyul 2026, 16 o'quv kuni +
4 chorshanba milestone). Har bir tinglovchi yagona kapstone loyiha — **"O'zbek
Hujjat Yordamchisi"** — quradi. Kaggle bepul rejim.

## 2. Ish intizomi (qat'iy)

- **Bir kunda bitta:** artefakt → barcha sifat darvozalari → QA → TO'XTA → inson tasdig'i → keyingi.
- **`course/course_map.yaml` — yagona haqiqat manbasi.** Mavzu, subitemlar, seminal_paper,
  uzbek_angle, hand_example faqat shundan. Day 12 `lecture_official_no: 12` qatori tasdiqlangan.
- **`.claude/skills/`** majburiy: ma'ruza uchun **`lecture-beamer`** + `uzbek-course-style` + `kaggle-hardware`.
- **Kun juftlash:** L(N) ma'ruza P(N) amaliyotiga tayyorlaydi. L12 ning [I] hand_example
  (ROUGE-1) **P12** (Day 13 ertalab — m12 TransformerSummarizer) ning birinchi assertiga ulanadi.
- **Mahalliy vositalar:** MiKTeX (`pdflatex`), Python 3.13 (vizual render uchun). torch 2.10 CPU bor.

## 3. BAJARILGAN ISHLAR

**Ma'ruzalar:** L1–L11 — d01…d05, d06_word2vec, d07_rnn, d08_gru_lstm, d09_matn_generatsiya,
d10_ner, d11_seq2seq_attention (tex+pdf; d02 faqat tex). Har biriga QA. ✅

**Amaliyotlar:** P1 (m01) … P11 (m11). Har biri +SOLUTIONS +checkpoints +QA, mahalliy bajarilgan. ✅

**Modullar:** m01–m11 (+ m05b). ✅

**Milestonelar:** w1 (m01+m02 ✅), w2 (m01–m05 ✅), w3 (neyron m01–m08, 18/18 ✅).

> Holat: ma'ruzalar **L11 gacha**, amaliyotlar **P11 gacha**, milestonelar w3 gacha. **Keyingi
> xronologik artefakt — L12** (Day 12 ma'ruza — Transformer). So'ng P12 (Day 13 ertalab) ga juft bo'ladi.

## 4. KEYINGI QADAM — L12 (so'ralgan)

> L12 = **course_map Day 12, `lecture_official_no: 12`**; P11 (Seq2Seq/Attention amaliyoti) dan keyingi ma'ruza.
> Fayl: `course/lectures/d12_transformer.tex` → `d12_transformer.pdf`.
> Juft amaliyot: **P12** (Day 13 ertalab — m12 TransformerSummarizer), L12 [I] ROUGE ni iste'mol qiladi.

**L12 spetsifikatsiyasi (course_map Day 12):**
- **Mavzu:** "Transformer arxitekturasi va matnni umumlashtirish."
- **4 kichik bo'lim (lecture_subitems) → 4 to'liq sikl:**
  1. Rekurrentlikdan voz kechish: **Self-Attention** bilan parallellashtirish.
  2. Transformerning asosiy komponentlari: **Multi-Head Attention** va **Positional Encodings**.
  3. Transformer arxitekturasini **abstrakt matnni umumlashtirishda** qo'llash (enkoder-dekoder).
  4. Umumlashtirish natijalarini **ROUGE** metrikasi bilan baholash.
- **Seminal maqola:** Vaswani, A., Shazeer, N., Parmar, N., et al. (2017).
  *Attention Is All You Need.* NeurIPS 30.
- **uzbek_angle (majburiy [M] slayd):** Transformer va o'zbek erkin so'z tartibi — positional
  encoding erkin tartib muammosini qanchalik hal qiladi? Self-attention ga SOV va SVO bir xilmi?

**QULFLANGAN [I] hand_example (L12[I] → P12 birinchi assert) — course_map dan AYNAN:**
- **ROUGE-1 F1.** Reference = `nlp juda qiziq va foydali` (5 token); Hypothesis = `nlp juda foydali` (3 token).
- Precision = 3/3 = **1.000** (barcha hypothesis tokenlari reference da bor).
- Recall = 3/5 = **0.600** (reference tokenlarining 3/5 i topildi).
- F1 = 2·1.0·0.6 / (1.0+0.6) = 1.2/1.6 = **0.750**.
- **KUTILGAN: ROUGE-1 P=1.000, R=0.600, F1=0.750.** Qo'lda hisob [I4] slaydda ko'rsatiladi;
  traceability izohi: `# Ma'ruza L12 [I4]-slayd bilan solishtiring` (P12 ning birinchi asserti shu).

**Sikllarning tabiiy taqsimoti (taklif — lecture-beamer):**
- **Sikl 1 — Self-Attention:** scaled dot-product `Attention(Q,K,V)=softmax(QKᵀ/√d_k)V`; rekurrentlik yo'q,
  to'liq parallel, har pozitsiya barcha pozitsiyalarga "qaraydi". [I1] kichik qo'lda hisob (masalan
  bitta so'rov uchun 3-pozitsiyali softmax — L11 attention bilan bog'lash mumkin).
- **Sikl 2 — Multi-Head + Positional Encoding:** bir nechta head (har biri alohida Q/K/V proyeksiya),
  konkatenatsiya + chiziqli; sinusoidal `PE(pos,2i)=sin(pos/10000^{2i/d})`. [I2] bitta pozitsiya uchun PE qiymati.
- **Sikl 3 — Transformer bloki / enkoder-dekoder:** residual + LayerNorm + FFN; encoder self-attn,
  decoder masked self-attn + cross-attn; umumlashtirish (summarization) uchun seq2seq sifatida.
- **Sikl 4 — ROUGE:** ROUGE-1/ROUGE-L; **[I4] = QULFLANGAN** ROUGE-1 P/R/F1 misoli (yuqorida).

> DIQQAT: L12 da **abstraktiv umumlashtirish** (Wikipedia lead-paragraph) konteksti — P12/m12 ga ulanish.
> ROUGE BLEU dan farqi (recall-yo'naltirilgan) [F4]/[L4] da ta'kidlanadi.

## 5. MA'RUZA TUZILISHI (lecture-beamer skill — qat'iy skelet)

Arxetiplar [A]–[S], **4 to'liq sikl**, ~47 frame (`\appendix` dan oldin), footer `/47`.
- **[A]–[E] Kirish:** sarlavha, **[B] recap** (L11 Seq2Seq/Attention + bu ertalabki P11),
  reja, motivatsiya. **[E] muammo-avval:** RNN ketma-ket ishlaydi → parallellashmaydi + uzoq
  bog'liqlik sust → "rekurrentlikni butunlay olib tashlasak-chi?".
- **4 sikl, har biri:** [F] intuitsiya → [G] ta'rif + **`\bunda{}`** kalit → [H] hosil qilish →
  [I] qo'lda misol → [J] vazifa → [K] kod↔formula (`[fragile]` `lstlisting`) → [L] tipik xato.
- **[M] majburiy o'zbek-burchak slaydi:** positional encoding va erkin so'z tartibi (yuqoridagi uzbek_angle).
- **[N]–[R] Xulosa:** taqqoslash jadvali (RNN/LSTM vs Transformer), [O] seminal (Vaswani 2017 + muhokama),
  [Q] ko'prik (P12/m12 TransformerSummarizer — birinchi assert = qulflangan ROUGE), [R] yakun.
- **[S] appendix** (`\appendix` dan keyin, frame hisobiga kirmaydi): masalan to'liq MHA matematikasi
  yoki ROUGE-L LCS hisobi.

## 6. ⚠️ KOMPILYATSIYA — TAKRORLANUVCHI XATOLAR (oldini ol)

L7–L11 da uchragan va OLDINI OLISH kerak bo'lgan nuqtalar:
- **Kirill harflari:** kompilyatsiyadan OLDIN butun `.tex` ni Kirill (U+0400–U+04FF) ga skan qil → **0** bo'lsin
  (L9 "kontekстга", L10 "shунга" hodisalari). Lotin ko'rinishidagi `с/а/о/р/е/х/г` ayniqsa xavfli.
- **tcolorbox/frame sarlavhasidagi matematik:** `h_0`, `W_h`, `x_1` kabi pastki indekslar sarlavhada
  `$...$` ichida bo'lsin (L7 "Missing $ inserted").
- **Sarlavhadagi vergul:** `title={precision, recall, F1}` — vergulli sarlavha `title={...}` qavs ichida
  bo'lsin (L10 pgfkeys xatosi).
- **Yuklanmagan paketlar:** `\ding{55}` (pifont YO'Q) → `($\times$)`; `\psmallmatrix` (mathtools YO'Q) →
  `\left(\begin{smallmatrix}...\end{smallmatrix}\right)` (L7/L11). `\checkmark` amssymb dan — ruxsat.
- **Preambula d11 dan BAYT-AYNAN ko'chiriladi** (ranglar, Boadilla footer, `lstset`, `tcbset`, tikz stillari).

## 7. SIFAT DARVOZALARI (MAHALLIY — kechiktirilmaydi)

- **`pdflatex` ×2:** chiqish 0; `grep "^!"` bo'sh (0 xato).
- **0 ta `Overfull \hbox` > 10pt:** `grep "Overfull \hbox ([1-9][0-9]"` bo'sh.
- **Frame soni 4 to'liq sikl bilan** (L1–L11 pariteti, ~47 frame `\appendix` dan oldin); 22–28 ga QISQARTIRILMAYDI.
- **Barcha arxetiplar [A]–[S] mavjud;** [H1]–[H4] har siklda hosil-qilish slaydi; [M] o'zbek slaydi.
- **Har formulada `\bunda{}` kaliti;** har `lstlisting` frame `[fragile]`.
- **QULFLANGAN [I4]:** ROUGE-1 P=1.000, R=0.600, F1=0.750 (course_map dan aynan) +
  `# Ma'ruza L12 [I4]-slayd` traceability izohi; [Q] ko'prik P12 birinchi assertini ko'rsatadi.
- **Vizual ko'rik:** PNG render (@95–150dpi) — qulflangan [I4], MHA/PE kod, taqqoslash jadvali, [Q] ko'prik
  — toshib ketish / kesilish / ustma-ust tushish YO'Q.
- **Terminologiya grep toza** (`professor\|talaba\|student\|o'qituvchi` = 0); ASCII apostrof; U+2019 yo'q;
  **Kirill 0**; BOM yo'q.
- **Aux/PNG avtomatik tozalash:** kompilyatsiyadan keyin `.aux/.log/.nav/.out/.snm/.toc/.vrb` + render PNG
  papkasi o'chiriladi; repozitoriyda faqat `.tex` + `.pdf` qoladi (doimiy ko'rsatma — [[clean-latex-aux-after-compile]]).

## 8. SO'RALADIGAN PROMPT

**L12 (12-ma'ruza: Transformer arxitekturasi va matnni umumlashtirish) ni ishlab chiqarish uchun
bosqichma-bosqich prompt** yozib bering —
- course_map Day 12 (lecture 12) spetsifikatsiyasiga aniq mos: 4 subitem (Self-Attention bilan
  parallellashtirish; Multi-Head Attention + Positional Encodings; abstrakt umumlashtirish;
  ROUGE baholash) → 4 to'liq sikl;
- seminal maqola **Vaswani et al. (2017) "Attention Is All You Need"** [O] slaydda; muhokama bilan;
- **[M] o'zbek-burchak slaydi**: positional encoding va erkin so'z tartibi (SOV/SVO self-attention uchun bir xilmi);
- **QULFLANGAN [I4] hand_example** = ROUGE-1 P=1.000, R=0.600, F1=0.750 (Ref `nlp juda qiziq va foydali`,
  Hyp `nlp juda foydali`); qo'lda hisob slaydda; `# Ma'ruza L12 [I4]-slayd`; [Q] ko'prik P12/m12 ga;
- **lecture-beamer skeleti**: arxetiplar [A]–[S], 4 sikl ([F][G][H][I][J][K][L]), [H1]–[H4],
  har formulada `\bunda{}`, har `lstlisting` `[fragile]`, ~47 frame, footer `/47`;
- **preambula d11_seq2seq_attention.tex dan BAYT-AYNAN**; recap [B] = L11 + P11; [E] muammo-avval
  (RNN parallellashmaydi / uzoq bog'liqlik);
- L11 attention bilan bog'lash: self-attention QKV — L11 dagi Bahdanau attention ning umumlashmasi;
  BLEU (L11) vs ROUGE (L12) farqini ta'kidlash;
- **mahalliy darvozalar**: `pdflatex` ×2 (0 xato), 0 Overfull >10pt, vizual render ko'rik, terminologiya
  toza, ASCII, **Kirill 0** (kompilyatsiyadan oldin skan), BOM yo'q; aux/PNG avtomatik tozalash;
- **takrorlanuvchi xatolarni oldini ol** (§6: sarlavhadagi `$...$`/vergul, yuklanmagan paketlar, Kirill);
- 1 commit: `day12: lecture — L12 Transformer arxitekturasi va matnni umumlashtirish`.

---

## Ilova — repozitoriy fayl holati

```
course/lectures/   : d01..d05, d06_word2vec, d07_rnn, d08_gru_lstm, d09_matn_generatsiya, d10_ner, d11_seq2seq_attention  (tex+pdf; d02 tex)
course/practices/  : d02_p1 ... d12_p11_seq2seq  (+ _SOLUTIONS, + d0N_checkpoints/)
capstone/modules/  : m01..m11 (+ m05b)   (m12 — P12 da quriladi)
course/milestones/ : w1_*, w2_*, w3_*
course/qa/         : d01, L1–L11, P1–P11, w1, w2, w3 (+ skriptlar)
```

So'nggi commitlar:
```
674efa1 docs: HOLAT_HISOBOT.md — P11 ga yangilandi (L11 yopildi, m11 keyingi)
b467b72 day12: qa — P11 report (all gates PASS, 10/10 local asserts; torch+lug'at)
2bbe0e6 day12: capstone — m11 Seq2SeqTranslator (LSTM+Bahdanau attention, torch-ixtiyoriy)
f88f047 day12: practice — P11 seq2seq notebook + SOLUTIONS
0e9da4b day11: lecture — L11 Neyron mashina tarjimasi (Seq2Seq va Attention)
```
