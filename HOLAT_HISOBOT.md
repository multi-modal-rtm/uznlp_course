# NLP KURSI — LOYIHA HOLAT HISOBOTI

**Sana:** 2026-06-16 · **Branch:** `feat/help_to_improve` · **Oxirgi commit:** `7803385`
**Remotelar:** `origin` (datascientistn1/uznlp_course), `rtm` (multi-modal-rtm/uznlp_course).
> Holat: ikkala remote ham **to'liq sinxron** (push qilingan, 0 ortda).

> Bu hisobot mustaqil: uni boshqa Claude (yoki hamkasb) ga berib, **L14 (14-ma'ruza:
> RAG va vektor ma'lumotlar bazalari)** uchun bosqichma-bosqich prompt olish mumkin.

---

## 1. Loyiha nima

O'zbek tilidagi (lotin yozuvi) 4 haftalik intensiv NLP kursining BARCHA o'quv
materiallarini ishlab chiqarish loyihasi (15-iyun – 10-iyul 2026, 16 o'quv kuni +
4 chorshanba milestone). Har bir tinglovchi yagona kapstone loyiha — **"O'zbek
Hujjat Yordamchisi"** — quradi. Kaggle bepul rejim.

## 2. Ish intizomi (qat'iy)

- **Bir kunda bitta:** artefakt → barcha sifat darvozalari → QA → TO'XTA → inson tasdig'i → keyingi.
- **`course/course_map.yaml` — yagona haqiqat manbasi.** Mavzu, subitemlar, seminal_paper,
  uzbek_angle, hand_example faqat shundan. Day 14 `lecture_official_no: 14` qatori tasdiqlangan.
- **`.claude/skills/`** majburiy: ma'ruza uchun **`lecture-beamer`** + `uzbek-course-style` + `kaggle-hardware`.
- **Kun juftlash:** L(N) ma'ruza P(N) amaliyotiga tayyorlaydi. L14 ning [I] hand_example (RAG prompt token
  hisobi) **P14** (Day 15 ertalab — m14 RAGEngine) ning birinchi assertiga ulanadi.
- **Mahalliy vositalar:** MiKTeX (`pdflatex`), Python 3.13 (vizual render uchun). torch 2.10 CPU + transformers bor.

## 3. BAJARILGAN ISHLAR

**Ma'ruzalar:** L1–L13 — …, d12_transformer, d13_transfer_learning (tex+pdf; d02 faqat tex). Har biriga QA. ✅

**Amaliyotlar:** P1 (m01) … P13 (m13). Har biri +SOLUTIONS +checkpoints +QA, mahalliy bajarilgan. ✅

**Modullar:** m01–m13 (+ m05b). ✅  (m10/m12/m13 — HAQIQIY modullar, save/load).

**Milestonelar:** w1 (m01+m02 ✅), w2 (m01–m05 ✅), w3 (neyron m01–m08, 18/18 ✅).

> Holat: ma'ruzalar **L13 gacha**, amaliyotlar **P13 gacha**, milestonelar w3 gacha. **Keyingi
> xronologik artefakt — L14** (Day 14 ma'ruza — RAG). So'ng P14 (Day 15 ertalab — m14 RAGEngine) ga juft bo'ladi.

## 4. KEYINGI QADAM — L14 (so'ralgan)

> L14 = **course_map Day 14, `lecture_official_no: 14`**; P13 (BERT fine-tuning amaliyoti) dan keyingi ma'ruza.
> Fayl: `course/lectures/d14_rag.tex` → `d14_rag.pdf`.
> Juft amaliyot: **P14** (Day 15 ertalab — m14 RAGEngine), L14 [I] RAG token hisobi ni iste'mol qiladi.

**L14 spetsifikatsiyasi (course_map Day 14):**
- **Mavzu:** "RAG va vektor ma'lumotlar bazalari."
- **4 kichik bo'lim (lecture_subitems) → 4 to'liq sikl:**
  1. LLM cheklovlari (**gallyutsinatsiyalar**) va RAG yondashuvi.
  2. **RAG jarayoni**: tashqi manbalardan ma'lumot qidirish va javob yaratish.
  3. Zamonaviy qidiruvda **vektor embeddinglar**ining o'rni.
  4. **Vektor ma'lumotlar bazalari** (Pinecone, Weaviate) va ularning RAG dagi roli.
- **Seminal maqola:** Lewis, P., Perez, E., Piktus, A., et al. (2020).
  *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 33.
- **uzbek_angle (majburiy [M] slayd):** RAG va o'zbek huquqiy matnlar — lex.uz qonunchilik hujjatlari;
  hallucination xavfi yuqori domenlarda RAG ayniqsa muhim; o'zbek tilida LLM javoblari qanchalik ishonchli?

**QULFLANGAN [I2] hand_example (L14[I] → P14 birinchi assert) — course_map dan AYNAN:**
- **RAG prompt token hisobi.** 3 ta retrieved chunk (har biri ≈200 token) + instruction template (≈100 token).
- jami $= 3\times 200 + 100 = \mathbf{700}$ token. LLM kontekst oynasiga nisbatan: $700/128000 < 1\%$.
- **KUTILGAN: prompt ≈ 700 token; retrieved_docs = 3; narx ≈ minimal.** Javob hallyusinatsiya o'rniga
  kontekstga tayanadi.
- Qo'lda hisob RAG-jarayoni siklida (2-sikl, [I2]) ko'rsatiladi; traceability izohi:
  `# Ma'ruza L14 [I2]-slayd bilan solishtiring` (P14 ning birinchi asserti shu).

**Sikllarning tabiiy taqsimoti (taklif — lecture-beamer):**
- **Sikl 1 — LLM cheklovlari va RAG g'oyasi:** hallucination, bilim eskirishi (knowledge cutoff);
  fine-tuning yangi faktlarni arzon qo'sha olmaydi → tashqi manbadan **qidirib** olamiz. [I1] aqlda misol
  (masalan bilim kesilishi sanasi → noto'g'ri javob ehtimoli).
- **Sikl 2 — RAG jarayoni:** retrieve top-k → prompt yig'ish (chunk'lar + instruction) → generate.
  [I2] = QULFLANGAN token hisobi (700 token, <1%).
- **Sikl 3 — Vektor embeddinglar va qidiruv:** so'rov va hujjatlarni embed qilish, **kosinus o'xshashlik**,
  top-k tanlash. [I3] kosinus o'xshashlik (L3 dagi 2/3 ni qayta ishlatish mumkin).
- **Sikl 4 — Vektor ma'lumotlar bazalari:** Pinecone/Weaviate; ANN indeks; upsert/query. [I4] aqlda misol
  (masalan 10000 chunk, top-3 retrieve; brute-force vs ANN taqqoslash).

> DIQQAT: L14 — 4-haftaning RAG/agent bosqichi. L13 dan tabiiy davom: "fine-tuning bilimni o'zgartiradi,
> ammo yangi/tashqi faktlarni qo'shmaydi → RAG ularni qidirib keltiradi". Faqat [I2] token hisobi course_map'da qulflangan.

## 5. MA'RUZA TUZILISHI (lecture-beamer skill — qat'iy skelet)

Arxetiplar [A]–[S], **4 to'liq sikl**, ~47 frame (`\appendix` dan oldin), footer `/47`.
- **[A]–[E] Kirish:** sarlavha, **[B] recap** (L13 transfer learning / BERT / T5 / fine-tuning + bu ertalabki P13),
  reja, motivatsiya. **[E] muammo-avval:** LLM gallyutsinatsiya qiladi va bilimi eskiradi; fine-tuning yangi
  faktni arzon qo'shmaydi → "tashqi manbadan qidirib, javobni unga tayantirsak-chi?".
- **4 sikl, har biri:** [F] intuitsiya → [G] ta'rif + **`\bunda{}`** kalit → [H] hosil qilish (≤4 qadam) →
  [I] qo'lda misol → [J] vazifa (savol→`\pause`→javob) → [K] kod↔formula (`[fragile]` `lstlisting`) → [L] tipik xato.
- **[M] majburiy o'zbek-burchak slaydi:** RAG va lex.uz huquqiy matnlar (hallucination-xavfli domen).
- **[N]–[R] Xulosa:** taqqoslash jadvali (faqat LLM vs RAG; fine-tuning vs RAG), [O] seminal (Lewis 2020 + muhokama),
  [Q] ko'prik (P14/m14 RAGEngine — birinchi assert = qulflangan RAG token hisobi), [R] yakun.
- **[S] appendix** (`\appendix` dan keyin, frame hisobiga kirmaydi): masalan chunking strategiyalari yoki ANN (HNSW) batafsil.

## 6. ⚠️ KOMPILYATSIYA — TAKRORLANUVCHI XATOLAR (oldini ol)

L7–L13 da uchragan va OLDINI OLISH kerak bo'lgan nuqtalar:
- **Kirill harflari:** kompilyatsiyadan OLDIN butun `.tex` ni Kirill (U+0400–U+04FF) ga skan qil → **0** bo'lsin
  (L9 "kontekстга", L10 "shунга", L12 "modelга" — har safar bittadan tushgan; L13 toza chiqdi). Lotin
  ko'rinishidagi `с/а/о/р/е/х/г/к` ayniqsa xavfli.
- **tcolorbox/frame sarlavhasidagi matematik:** pastki indekslar/kasr/`%` sarlavhada `$...$` ichida (L7).
  DIQQAT: `700/128000` kabi kasr va `<1\%` — matnda `\%` (qochirilgan), formulada math rejim.
- **Sarlavhadagi vergul:** vergulli sarlavha `title={...}` qavs ichida bo'lsin (L10 pgfkeys xatosi).
- **Yuklanmagan paketlar:** `\ding` (pifont YO'Q) → `($\times$)`; `\psmallmatrix` (mathtools YO'Q) →
  `\left(\begin{smallmatrix}...\end{smallmatrix}\right)`. `\checkmark` amssymb dan — ruxsat.
- **lstlisting ichida U+2019 (jingalak ') YO'Q — faqat ASCII '.**
- **`[K]` kod listinglarida uzun qatorlar/qo'lda chuqur indent toshib ketmasin** (L13 [K3]/[K4] dars'i):
  uzun import/identifikatorlarni qisqa qatorlarga bo'l yoki ustun kengligini moslang.
- **Preambula d13 dan BAYT-AYNAN ko'chiriladi** (ranglar, Boadilla footer, `lstset`, `tcbset`, tikz stillari).

## 7. SIFAT DARVOZALARI (MAHALLIY — kechiktirilmaydi)

- **`pdflatex` ×2:** chiqish 0; `grep "^!"` bo'sh (0 xato).
- **0 ta `Overfull \hbox` > 10pt:** `grep "Overfull \hbox ([1-9][0-9]"` bo'sh.
- **Frame soni 4 to'liq sikl bilan** (L1–L13 pariteti, ~47 frame `\appendix` dan oldin); 22–28 ga QISQARTIRILMAYDI.
- **Barcha arxetiplar [A]–[S] mavjud;** [H1]–[H4] har siklda hosil-qilish slaydi; [M] o'zbek slaydi.
- **Har formulada `\bunda{}` kaliti;** har `lstlisting` frame `[fragile]`.
- **QULFLANGAN [I2]:** prompt $≈700$ token, $700/128000<1\%$, retrieved_docs$=3$ (course_map dan aynan) +
  `# Ma'ruza L14 [I2]-slayd` traceability izohi; [Q] ko'prik P14 birinchi assertini ko'rsatadi.
- **Vizual ko'rik:** PNG render (@95–150dpi) — qulflangan [I2], RAG/vektor-DB kod, taqqoslash jadvali, [Q] ko'prik
  — toshib ketish / kesilish / ustma-ust tushish YO'Q.
- **Terminologiya grep toza** (`professor\|talaba\|student\|o'qituvchi` = 0); ASCII apostrof; U+2019 yo'q;
  **Kirill 0**; BOM yo'q.
- **Aux/PNG avtomatik tozalash:** kompilyatsiyadan keyin `.aux/.log/.nav/.out/.snm/.toc/.vrb` + render PNG
  papkasi o'chiriladi; repozitoriyda faqat `.tex` + `.pdf` qoladi (doimiy ko'rsatma — [[clean-latex-aux-after-compile]]).

## 8. SO'RALADIGAN PROMPT

**L14 (14-ma'ruza: RAG va vektor ma'lumotlar bazalari) ni ishlab chiqarish uchun bosqichma-bosqich prompt**
yozib bering —
- course_map Day 14 (lecture 14) spetsifikatsiyasiga aniq mos: 4 subitem (LLM cheklovlari/gallyutsinatsiya + RAG;
  RAG jarayoni; vektor embeddinglar qidiruvda; vektor DB Pinecone/Weaviate) → 4 to'liq sikl;
- seminal maqola **Lewis et al. (2020) "Retrieval-Augmented Generation"** [O] slaydda; muhokama bilan;
- **[M] o'zbek-burchak slaydi**: RAG va lex.uz huquqiy matnlar; hallucination-xavfli domenlarda RAG;
- **QULFLANGAN [I2] hand_example** = RAG prompt token hisobi: 3 chunk × ≈200 + ≈100 instruction = **700 token**,
  $700/128000<1\%$, retrieved_docs=3; qo'lda hisob slaydda; `# Ma'ruza L14 [I2]-slayd`; [Q] ko'prik P14/m14 RAGEngine ga;
- **lecture-beamer skeleti**: arxetiplar [A]–[S], 4 sikl ([F][G][H][I][J][K][L]), [H1]–[H4],
  har formulada `\bunda{}`, har `lstlisting` `[fragile]`, ~47 frame, footer `/47`;
- **preambula d13_transfer_learning.tex dan BAYT-AYNAN**; recap [B] = L13 + P13; [E] muammo-avval
  (LLM gallyutsinatsiya + bilim eskirishi; fine-tuning yangi faktni qo'shmaydi → tashqi qidiruv);
- L13 bilan bog'lash: fine-tuning bilimni sozlaydi, ammo yangi/tashqi faktlarni qo'sha olmaydi → RAG ularni
  qidirib keltiradi va javobni kontekstga tayantiradi; embedding (L3/L6) qidiruvda qayta ishlatiladi;
- **mahalliy darvozalar**: `pdflatex` ×2 (0 xato), 0 Overfull >10pt, vizual render ko'rik, terminologiya
  toza, ASCII, **Kirill 0** (kompilyatsiyadan oldin skan), BOM yo'q; aux/PNG avtomatik tozalash;
- **takrorlanuvchi xatolarni oldini ol** (§6: sarlavhadagi `$...$`/`\%`/vergul, yuklanmagan paketlar,
  lstlisting uzun qator, Kirill);
- 1 commit: `day14: lecture — L14 RAG va vektor ma'lumotlar bazalari`.

---

## Ilova — repozitoriy fayl holati

```
course/lectures/   : d01..d05, d06..d12, d13_transfer_learning  (tex+pdf; d02 tex)
course/practices/  : d02_p1 ... d14_p13_finetune  (+ _SOLUTIONS, + d0N_checkpoints/)
capstone/modules/  : m01..m13 (+ m05b)   (m14 — P14 da quriladi)
course/milestones/ : w1_*, w2_*, w3_*
course/qa/         : d01, L1–L13, P1–P13, w1, w2, w3 (+ skriptlar)
```

So'nggi commitlar:
```
7803385 docs: HOLAT_HISOBOT.md — P13 ga yangilandi (L13 yopildi, m13 keyingi)
772c26c day14: qa — P13 report (all gates PASS, 12/12 local asserts; LogReg fallback + toza-torch BCE)
5cfedc5 day14: capstone — m13 FineTunedClassifier (DistilBERT+Trainer + TF-IDF/LogReg fallback, transformers-ixtiyoriy)
2e72578 day14: practice — P13 finetune notebook + SOLUTIONS
b30ab3f day13: lecture — L13 Transfer Learning va oldindan o'qitilgan modellar (BERT, T5)
```
```
origin/rtm = 7803385 (to'liq sinxron, 0 ortda)
```
