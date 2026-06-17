# NLP KURSI — LOYIHA HOLAT HISOBOTI

**Sana:** 2026-06-16 · **Branch:** `feat/help_to_improve` · **Oxirgi commit:** `4f40b14`
**Remotelar:** `origin` (datascientistn1/uznlp_course), `rtm` (multi-modal-rtm/uznlp_course).
> Holat: ikkala remote ham **to'liq sinxron** (push qilingan, 0 ortda).

> Bu hisobot mustaqil: uni boshqa Claude (yoki hamkasb) ga berib, **P15 (15-amaliyot:
> m15 DocumentAssistantAgent — LangChain ReAct agent, kapstone yakuni)** uchun bosqichma-bosqich prompt olish mumkin.

---

## 1. Loyiha nima

O'zbek tilidagi (lotin yozuvi) 4 haftalik intensiv NLP kursining BARCHA o'quv
materiallarini ishlab chiqarish loyihasi (15-iyun – 10-iyul 2026, 16 o'quv kuni +
4 chorshanba milestone). Har bir tinglovchi yagona kapstone loyiha — **"O'zbek
Hujjat Yordamchisi"** — quradi. Kaggle bepul rejim. **P15 — kapstone YAKUNI:** barcha modullar bitta agentga jamlanadi.

## 2. Ish intizomi (qat'iy)

- **Bir kunda bitta:** artefakt → barcha sifat darvozalari → QA → TO'XTA → inson tasdig'i → keyingi.
- **`course/course_map.yaml` — yagona haqiqat manbasi.** Mavzu, maqsadlar, hand_example faqat shundan.
- **`.claude/skills/`** majburiy: `practice-notebook`, `lecture-beamer`, `uzbek-course-style`, `kaggle-hardware`.
- **Kun juftlash:** L(N) ma'ruza P(N) amaliyotiga tayyorlaydi. Har P ning birinchi
  asserti mos L ning [I] hand_example natijasini tekshiradi (kuzatiluvchanlik).
- **Mahalliy vositalar:** Python 3.13 (numpy/sklearn/matplotlib/torch/transformers/sentence-transformers).
  ⚠️ **langchain YO'Q, faiss YO'Q, LLM API YO'Q**, datasets YO'Q.
- **Amaliyot uslubi:** `_build_pN.py` builder — offline data, modulni yozish, student+SOLUTIONS
  notebook'larni JSON qurish, SOLUTIONS kataklarini exec qilib assert tekshirish. Builder commit qilinmaydi.

## 3. BAJARILGAN ISHLAR

**Ma'ruzalar:** L1–L15 — …, d14_rag, **d15_agentlar** (tex+pdf; d02 faqat tex). Har biriga QA. ✅

**Amaliyotlar:** P1 (m01) … P14 (m14). Har biri +SOLUTIONS +checkpoints +QA, mahalliy bajarilgan. ✅

**Modullar:** m01–m14 (+ m05b). ✅  (m10/m12/m13/m14 — HAQIQIY modullar, save/load yoki save_index).

**Milestonelar:** w1 (m01+m02 ✅), w2 (m01–m05 ✅), w3 (neyron m01–m08, 18/18 ✅).

> Holat: ma'ruzalar **L15 gacha**, amaliyotlar P14 gacha, milestonelar w3 gacha. **Keyingi
> xronologik artefakt — P15** (Day 16 ertalab; L15 ga juft). So'ng L16 (Day 16 ma'ruza — MLOps) va w4 milestone.

## 4. KEYINGI QADAM — P15 (so'ralgan)

> P15 = **course_map Day 16, `practice_official_no: 15`**; **L15** (AI agentlar, ReAct) ma'ruzasiga juft.
> Modul: **m15 DocumentAssistantAgent**, fayl `capstone/modules/m15_langchain_agent.py`.
> Notebook: `course/practices/d16_p15_agent.ipynb` (+ `_SOLUTIONS` + `d16_checkpoints/`).
> ⚠️ **session_role: wiring_and_polish** — M4 Task C scaffold + Day 15 qisman ulashdan davom; noldan emas.

**P15 spetsifikatsiyasi (course_map Day 16):**
- **Mavzu:** "LangChain yordamida AI agentini yaratish."
- **4 kichik bo'lim (practice_subitems):**
  1. LangChain LLM/so'rov/vositalar bilan tanishish.
  2. Agentga integratsiya uchun oddiy vositalarni aniqlash.
  3. ReAct agentini yaratish va unga vositalardan foydalanish imkonini berish.
  4. Agentga murakkabroq vazifa berib, ishlashini kuzatish.
- **Periferiya (to'liq beriladi — PRIMM):**
  - LangChain `Tool`, `AgentExecutor`, `create_react_agent`.
  - m12 (TransformerSummarizer), m13 (FineTunedClassifier), m14 (RAGEngine) import; M4 tool stub'lar.
- **Yadro (tinglovchi yozadi):**
  - Barcha tool sinflarini AgentExecutor ga ulash.
  - ReAct: Thought→Action→Observation siklini 3 qadamda kuzatish.
  - Murakkab vazifa: hujjat tahlili → sentiment → RAG javob → xulosa; demo tozalash.
- **corpus_subset:** uz_kb. **OFFLINE_FALLBACK:** `d16_checkpoints/` (yoki avvalgi mini korpuslar) — kichik original data.
- **gpu_required:** yo'q (rule-based router offline; LLM Kaggle).

**QULFLANGAN birinchi assert (L15 [I1] → P15).** Agent to'g'ri vositani tanlashini tekshirsin (toza, LLM SHART EMAS):
- query = `"Uzum Market ilovasi yaxshimi?"`; Tools = [sentiment_classify, retrieve_docs, summarize].
- **KUTILGAN: correct_tool = sentiment_classify; action_confidence > 0.7.**
- `r = agent.route("Uzum Market ilovasi yaxshimi?")`
  `assert r["tool"] == "sentiment_classify"  # Ma'ruza L15 [I1]-slayd bilan solishtiring`
  `assert r["confidence"] > 0.7`
- (Agent `route()` / `last_trace` ni ochib bersin — `run()` natija str, lekin tool tanlash kuzatiladigan bo'lsin.)

**m15 shartnomasi (capstone/contracts.py — QAT'IY, AYNAN MOS):**
```
class DocumentAssistantAgent:                  # kapstone yakuni; consumed_by: [] (defense demo)
    run(user_message: str) -> str              # to'liq ReAct agent sikli; javob o'zbekcha
    # Tools: sentiment_classify→m13, retrieve_docs→m14, summarize_text→m12,
    #        spell_correct→m04, extract_entities→m10
```
> ⚠️ DIQQAT: m15 — **YAKUNIY integratsiya** moduli. Shartnomada faqat `run()` (save/load YO'Q). consumed_by []
> (himoya/defense demo'da bevosita ishlatiladi). Assertlik uchun `route()`/`last_trace` qo'shimcha metod sifatida
> ochib berilishi mumkin (shartnoma "kamida run()" — qo'shimcha metod ruxsat). run() o'zbekcha javob qaytaradi.

## 5. ⚠️ langchain / LLM-IXTIYORIY DIZAYN (rule-based router fallback)

> ⚠️ MUHIM: **langchain/faiss/LLM API mahalliy YO'Q**. Shuning uchun **mahalliy tekshirish FALLBACK orqali**.
- **Kaggle yo'li (`USE_LANGCHAIN=True`, internet+LLM):** `create_react_agent` + `AgentExecutor` + LLM;
  m13/m14/m12/m04/m10 `Tool` obyektlariga o'raladi.
- **Offline yo'l (langchain'siz/LLM'siz):** **rule-based (keyword) router** — ReAct "Thought" ni o'zbekcha
  kalit so'zlar bo'yicha modellaydi: query dagi so'zlarga qarab to'g'ri toolni tanlaydi (sentiment/retrieve/
  summarize/spell/entities), so'ng o'sha toolni (haqiqiy offline modul yoki stub) chaqiradi va o'zbekcha javob yig'adi.
- **`USE_LANGCHAIN` / `USE_LLM` bayroqlari** yo'lni tanlaydi; builder mahalliy False ga majburlaydi (m12/m13/m14 naqshi).
- **route() — toza keyword mantiqi** (toolsiz ham ishlaydi): locked assert har doim path-independent.
- **Tool'lar:** notebook m13/m14/m12 (mahalliy quriladi: TF-IDF/LogReg, TF-IDF/numpy, ekstraktiv) + m04/m10
  uchun yengil stub/lambda (yoki haqiqiy modul) ulaydi. run() tanlangan toolni chaqirib, ekstraktiv javob beradi.
- Notebook GPU'siz/langchain'siz/LLM'siz uchdan-uchgacha ishlasin; haqiqiy LangChain kodi KO'RSATILADI,
  mahalliyda rule-based router BAJARILADI. Javob sifati demo-darajada, halol.

## 6. PRACTICE-NOTEBOOK TUZILISHI (skill, gold standard P14/P13 naqshi)

§1 Muhit (seeds, OFFLINE_FALLBACK, HAS_LANGCHAIN, USE_LLM) → §2 yaxlit natija (tayyor agent.run(murakkab so'rov)
demo — Thought→Action→Observation→javob) → §3 PRIMM periferiya (LangChain Tool/AgentExecutor/create_react_agent +
m12/m13/m14 import — to'liq beriladi, guard bilan; offline = rule-based router) → Checkpoint → §4 yadro:
**so'nuvchi tayanch** (Namuna: locked route → sentiment_classify, conf>0.7 → Birgalikda `# === SIZNING KODINGIZ ===`
tool'larni dict ga ulash + route mantiqi → Mustaqil: murakkab vazifada run() va ReAct trace kuzatish),
har blank → mos **assert** → §5 loyihaga ulash (m15 ni yozish, import test, run() test, git) → §6 tadqiqot + exit ticket.

## 7. SIFAT DARVOZALARI (MAHALLIY — kechiktirilmaydi)

- **JSON valid** (student + SOLUTIONS): nbformat 4.5; har katak `id`.
- **Uchdan-uchgacha bajariladi** (`OFFLINE_FALLBACK=True`, CPU): SOLUTIONS kataklari mahalliy ishga tushadi,
  **har assert o'tadi**, 0 istisno. Builder orqali exec. Mahalliy = rule-based router + offline tool'lar.
- **Student stub kataklar `compile()` toza.**
- **Birinchi (qulflangan) assert:** `route("Uzum Market ilovasi yaxshimi?")["tool"]=="sentiment_classify"` +
  `["confidence"]>0.7` — `# Ma'ruza L15 [I1]-slayd`.
- **Har blank region mos assert bilan;** m15 shartnoma mosligi (`run()` mavjud, str qaytaradi).
- **run() strukturaviy assert:** `run(...)` `str` qaytaradi (bo'sh emas); `last_trace` Thought/Action/Observation
  qadamlarini saqlaydi. (Aniq/sifatli javob EMAS — rule-based, demo, halol.)
- **ReAct trace:** murakkab vazifada kamida tool tanlash + observation kuzatiladi.
- **No GPU** mahalliy; seeds (random/np 42); checkpoint katak(lar)i.
- **Terminologiya grep toza;** yorliqlar `ijobiy`/`salbiy`; ASCII apostrof; U+2019 yo'q; **Kirill 0**
  (notebook'da ham skan); BOM yo'q.

## 8. SO'RALADIGAN PROMPT

**P15 (15-amaliyot: m15 DocumentAssistantAgent — LangChain ReAct agent, kapstone yakuni) ni ishlab chiqarish
uchun bosqichma-bosqich prompt** yozib bering —
- course_map Day 16 (practice 15) spetsifikatsiyasiga aniq mos: 4 subitem, periferiya (LangChain Tool/
  AgentExecutor/create_react_agent, m12/m13/m14 import), yadro (tool'larni ulash, ReAct 3-qadam trace,
  murakkab vazifa: hujjat→sentiment→RAG→xulosa); session_role wiring_and_polish;
- **qulflangan birinchi assert** = L15 [I1]: `route("Uzum Market ilovasi yaxshimi?")` → tool=sentiment_classify,
  confidence>0.7 (toza, `# Ma'ruza L15 [I1]-slayd`);
- **m15 contracts.py imzosiga AYNAN mos** (`run(user_message)->str`; kapstone yakuni, consumed_by []; save/load YO'Q;
  assertlik uchun `route()`/`last_trace` qo'shimcha metod ruxsat);
- **langchain/LLM-ixtiyoriy**: Kaggle create_react_agent + AgentExecutor + LLM; offline **rule-based keyword router**
  (o'zbekcha kalit so'zlar bo'yicha tool tanlash) + offline tool'lar (m13/m14/m12 + m04/m10 stub); `USE_LANGCHAIN`/
  `USE_LLM` bayroqlari, builder mahalliy False ga majburlaydi; GPU'siz/langchain'siz uchdan-uchgacha;
- 5 tool ulanishi: sentiment_classify→m13, retrieve_docs→m14, summarize_text→m12, spell_correct→m04, extract_entities→m10;
- run() strukturaviy assert (str, bo'sh emas; last_trace Thought/Action/Observation); javob rule-based/demo KUTILGAN — halol;
- practice-notebook tuzilishi (§1–§6, so'nuvchi tayanch, PRIMM);
- mahalliy darvozalar: JSON valid, SOLUTIONS CPU'da bajariladi (har assert o'tadi), terminologiya toza,
  ASCII, Kirill 0, seeds; run() test;
- 3 commit: `day16: practice — P15 …` / `day16: capstone — m15 …` / `day16: qa — P15 report`.

---

## Ilova — repozitoriy fayl holati

```
course/lectures/   : d01..d05, d06..d12, d13_transfer_learning, d14_rag, d15_agentlar  (tex+pdf; d02 tex)
course/practices/  : d02_p1 ... d15_p14_rag  (+ _SOLUTIONS, + d0N_checkpoints/)
capstone/modules/  : m01..m14 (+ m05b)   (m15 — P15 da quriladi)
course/milestones/ : w1_*, w2_*, w3_*
course/qa/         : d01, L1–L15, P1–P14, w1, w2, w3 (+ skriptlar)
```

So'nggi commitlar:
```
4f40b14 docs: HOLAT_HISOBOT.md — L15 ga yangilandi (AI agentlar ma'ruzasi maqsadi)
9645270 day15: lecture — L15 Sun'iy intellekt agentlarini yaratish
99054d4 docs: HOLAT_HISOBOT.md — P14 ga yangilandi (L14 yopildi, m14 keyingi)
a572aa4 day15: qa — P14 report (all gates PASS, 14/14 local asserts; TF-IDF char-ngram fallback)
e3a6075 day15: capstone — m14 RAGEngine (ST+FAISS+LLM + TF-IDF/numpy/ekstraktiv fallback)
```
```
origin/rtm = 4f40b14 (to'liq sinxron, 0 ortda)
```
