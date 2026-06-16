# NLP KURSI — LOYIHA HOLAT HISOBOTI

**Sana:** 2026-06-16 · **Branch:** `feat/help_to_improve` · **Oxirgi commit:** `99054d4`
**Remotelar:** `origin` (datascientistn1/uznlp_course), `rtm` (multi-modal-rtm/uznlp_course).
> ⚠️ Holat: mahalliy HEAD remotelardan **4 commit oldinda** (P14 ning 4 commiti). Ikkala remote `bebcc12` da.
> L15 dan oldin yoki keyin push qilish kerak.

> Bu hisobot mustaqil: uni boshqa Claude (yoki hamkasb) ga berib, **L15 (15-ma'ruza:
> Sun'iy intellekt agentlarini yaratish — ReAct, LangChain)** uchun bosqichma-bosqich prompt olish mumkin.

---

## 1. Loyiha nima

O'zbek tilidagi (lotin yozuvi) 4 haftalik intensiv NLP kursining BARCHA o'quv
materiallarini ishlab chiqarish loyihasi (15-iyun – 10-iyul 2026, 16 o'quv kuni +
4 chorshanba milestone). Har bir tinglovchi yagona kapstone loyiha — **"O'zbek
Hujjat Yordamchisi"** — quradi. Kaggle bepul rejim.

## 2. Ish intizomi (qat'iy)

- **Bir kunda bitta:** artefakt → barcha sifat darvozalari → QA → TO'XTA → inson tasdig'i → keyingi.
- **`course/course_map.yaml` — yagona haqiqat manbasi.** Mavzu, subitemlar, seminal_paper,
  uzbek_angle, hand_example faqat shundan. Day 15 `lecture_official_no: 15` qatori tasdiqlangan.
- **`.claude/skills/`** majburiy: ma'ruza uchun **`lecture-beamer`** + `uzbek-course-style` + `kaggle-hardware`.
- **Kun juftlash:** L(N) ma'ruza P(N) amaliyotiga tayyorlaydi. L15 ning [I] hand_example (ReAct tool tanlash)
  **P15** (Day 16 ertalab — m15 DocumentAssistantAgent) ning birinchi assertiga ulanadi.
- **Mahalliy vositalar:** MiKTeX (`pdflatex`), Python 3.13 (vizual render uchun).

## 3. BAJARILGAN ISHLAR

**Ma'ruzalar:** L1–L14 — …, d13_transfer_learning, d14_rag (tex+pdf; d02 faqat tex). Har biriga QA. ✅

**Amaliyotlar:** P1 (m01) … P14 (m14). Har biri +SOLUTIONS +checkpoints +QA, mahalliy bajarilgan. ✅

**Modullar:** m01–m14 (+ m05b). ✅  (m10/m12/m13/m14 — HAQIQIY modullar, save/load yoki save_index).

**Milestonelar:** w1 (m01+m02 ✅), w2 (m01–m05 ✅), w3 (neyron m01–m08, 18/18 ✅).

> Holat: ma'ruzalar **L14 gacha**, amaliyotlar **P14 gacha**, milestonelar w3 gacha. **Keyingi
> xronologik artefakt — L15** (Day 15 ma'ruza — AI agentlar). So'ng P15 (Day 16 ertalab — m15 agent) ga juft bo'ladi.

## 4. KEYINGI QADAM — L15 (so'ralgan)

> L15 = **course_map Day 15, `lecture_official_no: 15`**; P14 (RAG amaliyoti) dan keyingi ma'ruza.
> Fayl: `course/lectures/d15_agentlar.tex` → `d15_agentlar.pdf`.
> Juft amaliyot: **P15** (Day 16 ertalab — m15 DocumentAssistantAgent), L15 [I1] ReAct tool tanlash ni iste'mol qiladi.

**L15 spetsifikatsiyasi (course_map Day 15):**
- **Mavzu:** "Sun'iy intellekt agentlarini yaratish."
- **4 kichik bo'lim (lecture_subitems) → 4 to'liq sikl:**
  1. Agent tizimlari: **ReAct** (Fikrlash + Harakat) namunasi.
  2. LLM imkoniyatlarini kengaytirish: **funksiya chaqirish** va vositalardan (tools) foydalanish.
  3. Tashqi APIlar bilan ishlovchi agentlar arxitekturasi va **NLP modellarini API sifatida joylashtirish**.
  4. **LangChain** va agentlik tizimlari arxitekturasi.
- **Seminal maqola:** Yao, S., Zhao, J., Yu, D., et al. (2022).
  *ReAct: Synergizing Reasoning and Acting in Language Models.* ICLR 2023.
- **uzbek_angle (majburiy [M] slayd):** LLM agent va o'zbek — ReAct tizimida o'zbek tilida fikrlash
  qanchalik ishonchli? Tool chaqiruv o'zbek tilida berilgan vazifani tushuna oladimi?

**QULFLANGAN [I1] hand_example (L15[I] → P15 birinchi assert) — course_map dan AYNAN:**
- **LangChain ReAct 1-qadam.** query = `"Uzum Market ilovasi yaxshimi?"`;
  Tools = `[sentiment_classify, retrieve_docs, summarize]`.
- Thought: "Hissiyot tahlili kerak → sentiment_classify ni chaqiraman."
  Action: `sentiment_classify(text="Uzum Market ilovasi yaxshimi?")`.
  Observation: `{sentiment: ijobiy, confidence: 0.82}`.
- **KUTILGAN: correct_tool = sentiment_classify; action_confidence > 0.7** (agent summarize ni emas,
  sentiment_classify ni tanlashi kerak).
- Qo'lda hisob ReAct siklida (1-sikl, [I1]) ko'rsatiladi; traceability izohi:
  `# Ma'ruza L15 [I1]-slayd bilan solishtiring` (P15 ning birinchi asserti shu).

**Sikllarning tabiiy taqsimoti (taklif — lecture-beamer):**
- **Sikl 1 — ReAct (Thought → Action → Observation):** fikrlash + harakat sikli; agent vositani tanlaydi,
  natijani kuzatadi, takrorlaydi. [I1] = QULFLANGAN ReAct trace (sentiment_classify, conf 0.82).
- **Sikl 2 — Funksiya chaqirish / tools:** LLM ga vositalar (nom, tavsif, argumentlar sxemasi) beriladi;
  model qaysi toolни qanday argument bilan chaqirishni tanlaydi. [I2] aqlda misol (tool tanlash/sxema).
- **Sikl 3 — Agent arxitekturasi + NLP modelni API:** tashqi API (masalan FastAPI endpoint) bilan ishlash;
  POST /predict {text} → {sentiment, confidence}. [I3] API so'rov/javob misoli.
- **Sikl 4 — LangChain:** `Tool` obyektlari, `AgentExecutor`, ReAct prompt shabloni. [I4] aqlda misol
  (masalan 5 ta tool: sentiment/retrieve/summarize/spell/entities ulanishi).

> DIQQAT: L15 — kapstone YAKUNI sari. Avvalgi modullar (m13 sentiment, m14 RAG, m12 summarize, m04 spell,
> m10 entities) agent **vositalariga** aylanadi. L14 (RAG) dan davom: RAG — bitta tool; agent ko'p toolни
> rejalashtirib chaqiradi. Faqat [I1] (ReAct tool tanlash) course_map'da qulflangan. **gpu_required: false.**

## 5. MA'RUZA TUZILISHI (lecture-beamer skill — qat'iy skelet)

Arxetiplar [A]–[S], **4 to'liq sikl**, ~47 frame (`\appendix` dan oldin), footer `/47`.
- **[A]–[E] Kirish:** sarlavha, **[B] recap** (L14 RAG + bu ertalabki P14), reja, motivatsiya.
  **[E] muammo-avval:** LLM o'zi tashqi dunyoga ta'sir qila olmaydi (qidiruv, hisob, API) va ko'p qadamli
  vazifani rejalashtira olmaydi → "unga vositalar berib, fikrlab-harakat qildirsak-chi (agent)?".
- **4 sikl, har biri:** [F] intuitsiya → [G] ta'rif + **`\bunda{}`** kalit → [H] hosil qilish (≤4 qadam) →
  [I] qo'lda misol → [J] vazifa (savol→`\pause`→javob) → [K] kod↔formula (`[fragile]` `lstlisting`) → [L] tipik xato.
- **[M] majburiy o'zbek-burchak slaydi:** ReAct o'zbek tilida fikrlash; tool chaqiruv o'zbek vazifani tushunadimi.
- **[N]–[R] Xulosa:** taqqoslash jadvali (oddiy LLM vs RAG vs agent; yoki tool'siz vs tool'li), [O] seminal
  (Yao 2022 ReAct + muhokama), [Q] ko'prik (P15/m15 DocumentAssistantAgent — birinchi assert = qulflangan ReAct tool tanlash),
  [R] yakun.
- **[S] appendix** (`\appendix` dan keyin, frame hisobiga kirmaydi): masalan ReAct prompt shabloni to'liq yoki
  agent xotira (memory) / multi-step trace.

## 6. ⚠️ KOMPILYATSIYA — TAKRORLANUVCHI XATOLAR (oldini ol)

L7–L14 da uchragan va OLDINI OLISH kerak bo'lgan nuqtalar:
- **⚠️ KIRILL HARFLARI (ENG O'JAR):** kompilyatsiyadan OLDIN butun `.tex` ni Kirill (U+0400–U+04FF) ga skan
  qil → **0** bo'lsin. L14 da **8 ta** homoglif tushgan edi (modelга, manbани, uzunlikка/дan, ...)! Lotin
  ko'rinishidagi `г/к/д/н/с/а/о/р/е/х` qo'shimchalarga (-ga, -ka, -dan, -ни) aralashib ketadi. JUDA ehtiyot bo'l.
- **tcolorbox/frame sarlavhasidagi matematik/%:** indeks/kasr/`%` sarlavhada `$...$` ichida; matnda `\%` (L7/L14).
- **Sarlavhadagi vergul:** vergulli sarlavha `title={...}` qavs ichida (L10). DIQQAT: Tools=[a, b, c] kabi
  vergulли ro'yxatlar sarlavhada bo'lsa `title={...}` da.
- **Yuklanmagan paketlar:** `\ding` (pifont YO'Q) → `($\times$)`; `\psmallmatrix` (mathtools YO'Q) →
  `\left(\begin{smallmatrix}...\end{smallmatrix}\right)`. `\checkmark` amssymb dan — ruxsat.
- **lstlisting ichida U+2019 (jingalak ') YO'Q — faqat ASCII '.** Uzun qatorlar/chuqur indent toshmasin (L13/L14):
  uzun import (langchain) / JSON ni qisqa qatorlarga bo'l.
- **JSON/`{}` lstlisting tashqarisida:** matnда `{sentiment: ijobiy}` ko'rsatilsa `\{...\}` qochiriladi yoki
  `\texttt{}`/verbatim ichida bo'lsin (LaTeX `{}` guruh belgisi).
- **Preambula d14 dan BAYT-AYNAN ko'chiriladi** (ranglar, Boadilla footer, `lstset`, `tcbset`, tikz stillari).

## 7. SIFAT DARVOZALARI (MAHALLIY — kechiktirilmaydi)

- **`pdflatex` ×2:** chiqish 0; `grep "^!"` bo'sh (0 xato).
- **0 ta `Overfull \hbox` > 10pt:** `grep "Overfull \hbox ([1-9][0-9]"` bo'sh.
- **Frame soni 4 to'liq sikl bilan** (L1–L14 pariteti, ~47 frame `\appendix` dan oldin); 22–28 ga QISQARTIRILMAYDI.
- **Barcha arxetiplar [A]–[S] mavjud;** [H1]–[H4] har siklda hosil-qilish slaydi; [M] o'zbek slaydi.
- **Har formulada `\bunda{}` kaliti;** har `lstlisting` frame `[fragile]`.
- **QULFLANGAN [I1]:** correct_tool=sentiment_classify, action_confidence>0.7 (Observation conf=0.82) (course_map dan aynan) +
  `# Ma'ruza L15 [I1]-slayd` traceability izohi; [Q] ko'prik P15 birinchi assertini ko'rsatadi.
- **Vizual ko'rik:** PNG render (@95–150dpi) — qulflangan [I1] ReAct trace, agent/LangChain kod, taqqoslash jadvali,
  [Q] ko'prik — toshib ketish / kesilish / ustma-ust tushish YO'Q.
- **Terminologiya grep toza** (`professor\|talaba\|student\|o'qituvchi` = 0); yorliqlar `ijobiy`/`salbiy`;
  ASCII apostrof; U+2019 yo'q; **Kirill 0**; BOM yo'q.
- **Aux/PNG avtomatik tozalash:** kompilyatsiyadan keyin `.aux/.log/.nav/.out/.snm/.toc/.vrb` + render PNG
  papkasi o'chiriladi; repozitoriyda faqat `.tex` + `.pdf` qoladi (doimiy ko'rsatma — [[clean-latex-aux-after-compile]]).

## 8. SO'RALADIGAN PROMPT

**L15 (15-ma'ruza: Sun'iy intellekt agentlarini yaratish — ReAct, LangChain) ni ishlab chiqarish uchun
bosqichma-bosqich prompt** yozib bering —
- course_map Day 15 (lecture 15) spetsifikatsiyasiga aniq mos: 4 subitem (ReAct; funksiya chaqirish/tools;
  agent arxitekturasi + NLP modelni API; LangChain) → 4 to'liq sikl;
- seminal maqola **Yao et al. (2022) "ReAct"** [O] slaydda; muhokama bilan;
- **[M] o'zbek-burchak slaydi**: ReAct o'zbek tilida fikrlash ishonchliligi; tool chaqiruv o'zbek vazifani tushunishi;
- **QULFLANGAN [I1] hand_example** = ReAct 1-qadam: query "Uzum Market ilovasi yaxshimi?", Tools=[sentiment_classify,
  retrieve_docs, summarize], Thought→sentiment_classify, Observation {sentiment: ijobiy, confidence: 0.82};
  KUTILGAN correct_tool=sentiment_classify, action_confidence>0.7; `# Ma'ruza L15 [I1]-slayd`; [Q] ko'prik P15/m15 ga;
- **lecture-beamer skeleti**: arxetiplar [A]–[S], 4 sikl ([F][G][H][I][J][K][L]), [H1]–[H4],
  har formulada `\bunda{}`, har `lstlisting` `[fragile]`, ~47 frame, footer `/47`;
- **preambula d14_rag.tex dan BAYT-AYNAN**; recap [B] = L14 + P14; [E] muammo-avval (LLM o'zi tashqi dunyoga
  ta'sir qila olmaydi / ko'p qadamli rejani bajara olmaydi → vositalar berib agent qilamiz);
- L14 bilan bog'lash: RAG — bitta vosita (retrieve); agent ko'p vositani (m13/m14/m12/m04/m10) rejalashtirib
  chaqiradi; avvalgi modullar agent tool'lariga aylanadi;
- **mahalliy darvozalar**: `pdflatex` ×2 (0 xato), 0 Overfull >10pt, vizual render ko'rik, terminologiya
  toza, ASCII, **Kirill 0** (kompilyatsiyadan oldin skan — L14 da 8 ta tushgan!), BOM yo'q; aux/PNG avtomatik tozalash;
- **takrorlanuvchi xatolarni oldini ol** (§6: Kirill, sarlavhadagi `$...$`/`\%`/vergul, yuklanmagan paketlar,
  lstlisting uzun qator, matndagi `{}` ni `\{\}` qochirish);
- 1 commit: `day15: lecture — L15 Sun'iy intellekt agentlarini yaratish`.

---

## Ilova — repozitoriy fayl holati

```
course/lectures/   : d01..d05, d06..d12, d13_transfer_learning, d14_rag  (tex+pdf; d02 tex)
course/practices/  : d02_p1 ... d15_p14_rag  (+ _SOLUTIONS, + d0N_checkpoints/)
capstone/modules/  : m01..m14 (+ m05b)   (m15 — P15 da quriladi)
course/milestones/ : w1_*, w2_*, w3_*
course/qa/         : d01, L1–L14, P1–P14, w1, w2, w3 (+ skriptlar)
```

So'nggi commitlar:
```
99054d4 docs: HOLAT_HISOBOT.md — P14 ga yangilandi (L14 yopildi, m14 keyingi)
a572aa4 day15: qa — P14 report (all gates PASS, 14/14 local asserts; TF-IDF char-ngram fallback)
e3a6075 day15: capstone — m14 RAGEngine (ST+FAISS+LLM + TF-IDF/numpy/ekstraktiv fallback)
696dc47 day15: practice — P14 rag notebook + SOLUTIONS
bebcc12 docs: HOLAT_HISOBOT.md — L14 ga yangilandi (RAG ma'ruzasi maqsadi)
```
```
⚠️ origin/rtm = bebcc12 (P14 ning 4 commiti push qilinmagan — 4 ortda)
```
