# NLP KURSI — LOYIHA HOLAT HISOBOTI

**Sana:** 2026-06-16 · **Branch:** `feat/help_to_improve` · **Oxirgi commit:** `5cef4ed`
**Remotelar:** `origin` (datascientistn1/uznlp_course), `rtm` (multi-modal-rtm/uznlp_course).
> ⚠️ Holat: mahalliy HEAD remotelardan **4 commit oldinda** (P15 ning 4 commiti). Ikkala remote `4f40b14` da.
> L16 dan oldin yoki keyin push qilish kerak.

> Bu hisobot mustaqil: uni boshqa Claude (yoki hamkasb) ga berib, **L16 (16-ma'ruza:
> NLP amaliyotida MLOps — kursning YAKUNIY ma'ruzasi)** uchun bosqichma-bosqich prompt olish mumkin.

---

## 1. Loyiha nima

O'zbek tilidagi (lotin yozuvi) 4 haftalik intensiv NLP kursining BARCHA o'quv
materiallarini ishlab chiqarish loyihasi (15-iyun – 10-iyul 2026, 16 o'quv kuni +
4 chorshanba milestone). Har bir tinglovchi yagona kapstone loyiha — **"O'zbek
Hujjat Yordamchisi"** — quradi. Kaggle bepul rejim.

## 2. Ish intizomi (qat'iy)

- **Bir kunda bitta:** artefakt → barcha sifat darvozalari → QA → TO'XTA → inson tasdig'i → keyingi.
- **`course/course_map.yaml` — yagona haqiqat manbasi.** Mavzu, subitemlar, seminal_paper,
  uzbek_angle, hand_example faqat shundan. Day 16 `lecture_official_no: 16` qatori tasdiqlangan.
- **`.claude/skills/`** majburiy: ma'ruza uchun **`lecture-beamer`** + `uzbek-course-style` + `kaggle-hardware`.
- **Kun juftlash:** odatda L(N) → P(N). LEKIN **L16 — kursning OXIRGI ma'ruzasi: keyingi amaliyot YO'Q**
  (`hand_example: null`). Demak **QULFLANGAN [I] assert YO'Q** — [I] slaydlar faqat illyustrativ (aqlda hisob).
- **Mahalliy vositalar:** MiKTeX (`pdflatex`), Python 3.13 (vizual render uchun).

## 3. BAJARILGAN ISHLAR

**Ma'ruzalar:** L1–L15 — …, d14_rag, d15_agentlar (tex+pdf; d02 faqat tex). Har biriga QA. ✅

**Amaliyotlar:** P1 (m01) … P15 (m15 — kapstone yakuni). Har biri +SOLUTIONS +QA, mahalliy bajarilgan. ✅

**Modullar:** m01–m15 (+ m05b). ✅  KAPSTONE TO'LIQ: barcha modullar m15 agentга ulangan.

**Milestonelar:** w1 (m01+m02 ✅), w2 (m01–m05 ✅), w3 (neyron m01–m08, 18/18 ✅). w4 (M4) — qoldi.

> Holat: ma'ruzalar **L15 gacha**, amaliyotlar **P15 gacha** (kapstone yakunlandi), milestonelar w3 gacha.
> **Keyingi xronologik artefakt — L16** (Day 16 ma'ruza — MLOps, kursning yakuniy ma'ruzasi).
> L16 dan keyin: **w4 (M4) milestone** (P16 FastAPI/Docker, bilim testi, agent scaffold) — kurs to'liq yopiladi.

## 4. KEYINGI QADAM — L16 (so'ralgan)

> L16 = **course_map Day 16, `lecture_official_no: 16`**; P15 (kapstone agent amaliyoti) dan keyingi ma'ruza.
> Fayl: `course/lectures/d16_mlops.tex` → `d16_mlops.pdf`.
> ⚠️ **Juft amaliyot YO'Q** — bu kursning OXIRGI ma'ruzasi; `hand_example: null`. [Q] ko'prik kapstone
> **himoyasiga (defense)** olib boradi, keyingi P ga emas.

**L16 spetsifikatsiyasi (course_map Day 16):**
- **Mavzu:** "NLP amaliyotida MLOps amaliyotlari."
- **4 kichik bo'lim (lecture_subitems) → 4 to'liq sikl:**
  1. Modelni **ishlab chiqarishga uzatish** bosqichlari.
  2. **API yaratish va dokerlashtirish** (FastAPI + Docker).
  3. **Model monitoringi va versiyalash** (drift, model registry).
  4. **CI/CD payplaynlari.**
- **Seminal maqola:** Sculley, D., et al. (2015). *Hidden Technical Debt in Machine Learning Systems.*
  NeurIPS 28. (MLOps, model degradatsiyasi, monitoring.)
- **uzbek_angle (majburiy [M] slayd):** O'zbek NLP ekotizimi — mavjud open-source modellar (UzBERT,
  Tahrirchi modellari), ochiq muammolar (labeled data, morfologik tahlilchi, domain-specific LLM);
  tinglovchilar hissa qo'sha oladigan yo'nalishlar.

**⚠️ QULFLANGAN [I] hand_example YO'Q (course_map: `hand_example: null`).**
- L16 — kursning oxirgi ma'ruzasi; keyingi P (amaliyot) yo'q → kuzatiluvchanlik (traceability) asserti YO'Q.
- [I] slaydlar baribir bo'lsin (lecture-beamer talab qiladi), lekin ular **illyustrativ** (aqlda hisob,
  P-assertga ulanmaydi). Misol: [I3] monitoring chegarasi — F1 0.85 → 0.70, $\Delta=0.15>0.10$ → qayta o'qitish signali.
- Hech qanday `# Ma'ruza L16 [I]-slayd` traceability izohi P ga ulanmaydi (P yo'q).

**Sikllarning tabiiy taqsimoti (taklif — lecture-beamer):**
- **Sikl 1 — Deploy bosqichlari:** o'qitish → paketlash → serve → monitor; "noutbukda ishlaydi" ≠ "ishlab chiqarish".
  [I1] aqlda misol (masalan model hajmi / yuklash bir marta vs har so'rov).
- **Sikl 2 — API + Docker:** FastAPI endpoint (M4 P16) + Dockerfile (base + deps + model qatlamlari).
  [I2] aqlda misol (image qatlamlari / port).
- **Sikl 3 — Monitoring + versiyalash:** model drift, metrikalar, model registry (v1/v2). [I3] drift chegarasi
  (F1 pasayishi → alert/retrain).
- **Sikl 4 — CI/CD:** test → build → deploy payplayni; avtomatlashtirish. [I4] aqlda misol (bosqichlar soni).

> DIQQAT: L16 — KURS YAKUNI. M4 P16 (FastAPI/Docker, flipped) da tinglovchilar buni amalda bajargan; L16
> tushunchalarni RASMIYLASHTIRADI. [B] recap = L15 (agentlar) + bu ertalabki P15 (kapstone agent). [Q] ko'prik
> = kapstone himoyasi (m15 DocumentAssistantAgent + SentimentAPI demo) + tinglovchining keyingi yo'li. gpu_required: false.

## 5. MA'RUZA TUZILISHI (lecture-beamer skill — qat'iy skelet)

Arxetiplar [A]–[S], **4 to'liq sikl**, ~47 frame (`\appendix` dan oldin), footer `/47`.
- **[A]–[E] Kirish:** sarlavha, **[B] recap** (L15 agentlar + bu ertalabki P15 kapstone agent), reja, motivatsiya.
  **[E] muammo-avval:** model noutbukda/Kaggle'da ishlaydi, ammo ishlab chiqarishda — masshtab, kechikish,
  monitoring, degradatsiya, yangilash muammolari → "modelni qanday ishonchli xizmatga aylantiramiz (MLOps)?".
- **4 sikl, har biri:** [F] intuitsiya → [G] ta'rif + **`\bunda{}`** kalit → [H] hosil qilish (≤4 qadam) →
  [I] qo'lda misol (ILLYUSTRATIV — locked emas) → [J] vazifa (savol→`\pause`→javob) → [K] kod↔formula
  (`[fragile]` `lstlisting` — Dockerfile/FastAPI/CI yaml) → [L] tipik xato.
- **[M] majburiy o'zbek-burchak slaydi:** O'zbek NLP ekotizimi, ochiq muammolar, hissa qo'shish yo'nalishlari.
- **[N]–[R] Xulosa:** taqqoslash jadvali (tadqiqot/notebook vs ishlab chiqarish; yoki deploy variantlari),
  [O] seminal (Sculley 2015 + muhokama), **[Q] ko'prik = kapstone himoyasi** (m15 + SentimentAPI demo; keyingi
  P emas!), [R] yakun + butun kurs xulosasi.
- **[S] appendix** (`\appendix` dan keyin, frame hisobiga kirmaydi): masalan model registry / A-B test yoki
  to'liq CI/CD yaml.

## 6. ⚠️ KOMPILYATSIYA — TAKRORLANUVCHI XATOLAR (oldini ol)

L7–L15 da uchragan va OLDINI OLISH kerak bo'lgan nuqtalar:
- **⚠️ KIRILL HARFLARI (ENG O'JAR):** kompilyatsiyadan OLDIN butun `.tex` ni Kirill (U+0400–U+04FF) ga skan
  qil → **0** bo'lsin. L14 da **8 ta**, L15 da **5 ta** homoglif tushgan (modelга, agentга, endpointга, ...)!
  Lotin ko'rinishidagi `г/к/д/н/с/а/о/р/е/х` ayniqsa qo'shimchalarga (-ga, -ka, -dan) aralashadi. JIDDIY tekshir.
- **tcolorbox/frame sarlavhasidagi matematik/%:** indeks/kasr/`%` sarlavhada `$...$` ichida; matnda `\%` (L7/L14).
- **Sarlavhadagi vergul:** vergulli sarlavha `title={...}` qavs ichida (L10).
- **⚠️ MATNDAGI {} (figurali qavs):** JSON / Docker / yaml matnda ko'rsatilsa `\{...\}` qochiriladi YOKI
  `\texttt{}`/`lstlisting` ichida (LaTeX `{}` guruh belgisi) (L15 darsi).
- **Yuklanmagan paketlar:** `\ding` (pifont YO'Q) → `($\times$)`; `\psmallmatrix` (mathtools YO'Q) →
  `\left(\begin{smallmatrix}...\end{smallmatrix}\right)`. `\checkmark` amssymb dan — ruxsat.
- **lstlisting (Dockerfile/yaml/bash) `[fragile]`; ichida U+2019 YO'Q — faqat ASCII '.** Uzun qator toshmasin.
- **Preambula d15 dan BAYT-AYNAN ko'chiriladi** (ranglar, Boadilla footer, `lstset`, `tcbset`, tikz stillari).

## 7. SIFAT DARVOZALARI (MAHALLIY — kechiktirilmaydi)

- **`pdflatex` ×2:** chiqish 0; `grep "^!"` bo'sh (0 xato).
- **0 ta `Overfull \hbox` > 10pt:** `grep "Overfull \hbox ([1-9][0-9]"` bo'sh.
- **Frame soni 4 to'liq sikl bilan** (L1–L15 pariteti, ~47 frame `\appendix` dan oldin); 22–28 ga QISQARTIRILMAYDI.
- **Barcha arxetiplar [A]–[S] mavjud;** [H1]–[H4] har siklda hosil-qilish slaydi; [M] o'zbek slaydi.
- **Har formulada `\bunda{}` kaliti;** har `lstlisting` frame `[fragile]`.
- **[I] hand_example QULFLANGAN EMAS** (L16: hand_example null) — illyustrativ; **`# Ma'ruza ...` traceability
  izohi yo'q** (P yo'q). [Q] ko'prik kapstone himoyasiga (keyingi P ga emas).
- **Vizual ko'rik:** PNG render (@95–150dpi) — Dockerfile/FastAPI/CI kod, taqqoslash jadvali, [M], [Q] himoya
  ko'prigi — toshib ketish / kesilish / ustma-ust tushish YO'Q.
- **Terminologiya grep toza** (`professor\|talaba\|student\|o'qituvchi` = 0); ASCII apostrof; U+2019 yo'q;
  **Kirill 0**; BOM yo'q.
- **Aux/PNG avtomatik tozalash:** kompilyatsiyadan keyin `.aux/.log/.nav/.out/.snm/.toc/.vrb` + render PNG
  papkasi o'chiriladi; repozitoriyda faqat `.tex` + `.pdf` qoladi (doimiy ko'rsatma — [[clean-latex-aux-after-compile]]).

## 8. SO'RALADIGAN PROMPT

**L16 (16-ma'ruza: NLP amaliyotida MLOps — kursning YAKUNIY ma'ruzasi) ni ishlab chiqarish uchun
bosqichma-bosqich prompt** yozib bering —
- course_map Day 16 (lecture 16) spetsifikatsiyasiga aniq mos: 4 subitem (deploy bosqichlari; API+Docker;
  monitoring+versiyalash; CI/CD) → 4 to'liq sikl;
- seminal maqola **Sculley et al. (2015) "Hidden Technical Debt in ML Systems"** [O] slaydda; muhokama bilan;
- **[M] o'zbek-burchak slaydi**: O'zbek NLP ekotizimi (UzBERT, Tahrirchi), ochiq muammolar, hissa qo'shish yo'nalishlari;
- **⚠️ QULFLANGAN [I] assert YO'Q** (course_map `hand_example: null`) — [I] slaydlar illyustrativ (aqlda hisob,
  P ga ulanmaydi); hech qanday traceability izohi qo'shilmaydi;
- **lecture-beamer skeleti**: arxetiplar [A]–[S], 4 sikl ([F][G][H][I][J][K][L]), [H1]–[H4],
  har formulada `\bunda{}`, har `lstlisting` `[fragile]`, ~47 frame, footer `/47`;
- **preambula d15_agentlar.tex dan BAYT-AYNAN**; recap [B] = L15 + P15; [E] muammo-avval (model noutbukda
  ishlaydi ≠ ishlab chiqarishda; masshtab/monitoring/degradatsiya → MLOps);
- M4 P16 (FastAPI/Docker, flipped) bilan bog'lash: tinglovchilar buni amalda qilgan, L16 rasmiylashtiradi;
  m13/m15 (SentimentAPI, agent) deploy konteksti; **[Q] ko'prik = kapstone himoyasi** (m15 + SentimentAPI demo,
  keyingi P emas) + butun kurs xulosasi;
- **mahalliy darvozalar**: `pdflatex` ×2 (0 xato), 0 Overfull >10pt, vizual render ko'rik, terminologiya
  toza, ASCII, **Kirill 0** (kompilyatsiyadan oldin skan — L14 da 8, L15 da 5 ta tushgan!), BOM yo'q; aux/PNG avtomatik tozalash;
- **takrorlanuvchi xatolarni oldini ol** (§6: Kirill, sarlavhadagi `$...$`/`\%`/vergul, matndagi `{}` ni
  `\{\}` qochirish, yuklanmagan paketlar, lstlisting uzun qator);
- 1 commit: `day16: lecture — L16 NLP amaliyotida MLOps`.

---

## Ilova — repozitoriy fayl holati

```
course/lectures/   : d01..d05, d06..d12, d13_transfer_learning, d14_rag, d15_agentlar  (tex+pdf; d02 tex)
course/practices/  : d02_p1 ... d16_p15_agent  (+ _SOLUTIONS, + d0N_checkpoints/)
capstone/modules/  : m01..m15 (+ m05b)   — KAPSTONE TO'LIQ
course/milestones/ : w1_*, w2_*, w3_*    (w4/M4 — qoldi)
course/qa/         : d01, L1–L15, P1–P15, w1, w2, w3 (+ skriptlar)
```

So'nggi commitlar:
```
5cef4ed docs: HOLAT_HISOBOT.md — P15 ga yangilandi (L15 yopildi, m15 kapstone yakuni)
9e73e3c day16: qa — P15 report (all gates PASS, 14/14 local asserts; rule-based router; capstone finale)
c006d86 day16: capstone — m15 DocumentAssistantAgent (LangChain ReAct + rule-based router fallback)
81ac8a4 day16: practice — P15 agent notebook + SOLUTIONS (kapstone yakuni)
4f40b14 docs: HOLAT_HISOBOT.md — L15 ga yangilandi (AI agentlar ma'ruzasi maqsadi)
```
```
⚠️ origin/rtm = 4f40b14 (P15 ning 4 commiti push qilinmagan — 4 ortda)
```
