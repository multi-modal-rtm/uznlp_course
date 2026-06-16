# 3-hafta milestone: Neyron arxitekturalar integratsiyasi

**Sana:** 1-iyul 2026 (chorshanba) · **Topshirish muddati:** 6-iyul 2026 (dushanba, 09:30 gacha)
**Format:** asinxron, mustaqil · **Qamrov:** m01–m08 (klassik + neyron klassifikatorlar, embeddinglar)

Chorshanba — **integratsiya kuni**. Ikki hafta davomida siz klassik (m02) va neyron
(m07 RNN, m08 GRU/LSTM) klassifikatorlarni hamda embeddinglarni (m06) qurdingiz. Bugun
ularni **bir xil mezon**da yonma-yon qo'yib, qaysi arxitektura qachon ustun ekanini
o'lchaymiz.

Bu milestone bitta **to'liq integratsiya** — m01–m08 hammasi tayyor. Asosiy savol:

> Neyron tarmoqlar (RNN, LSTM) klassik modeldan (TF-IDF + LogReg) **har doim** yaxshiroqmi?

Javob ko'pincha "yo'q" — ma'lumot hajmiga bog'liq. Buni o'lchab ko'ramiz.

```
       bir xil train/test split (sentiment: ijobiy/salbiy)
                          │
        ┌─────────────────┼─────────────────┐
   m02 TF-IDF+LogReg   m07 RNN          m08 GRU/LSTM
        │                 │                  │
        └──────── F1 va inference vaqt taqqoslovi ────────┘
```

---

## O'z korpusingiz

Kurs davomida siz **o'z hujjat to'plamingiz** ustida ishlaysiz. Bu milestone uchun
o'zingizning sentiment-belgilangan matnlaringizdan (yoki tayyor
`course/practices/d03_checkpoints/uz_sentiment_mini.txt`) foydalaning.

---

## 1-topshiriq — klassifikator taqqoslovi (m02 vs m07 vs m08)

**Eng muhim qism.** Uchala klassifikatorni **bir xil** train/test split'da baholaymiz.

**Qadamlar:**

1. Sentiment ma'lumotini yuklang va **bir xil** train/test split qiling
   (`train_test_split`, `random_state=42`, `stratify`).
2. Uchala modelni **train** qismida o'qiting:
   - `m02 SentimentClassifier` (TF-IDF + LogReg/NB),
   - `m07 RNNClassifier` (`nn.RNN` yoki numpy fallback),
   - `m08 GRULSTMClassifier` (`arch='lstm'`).
3. **Test** qismida har biriga **F1** va **inference vaqt**ini hisoblang.

**Kutilgan natija:** kichik ma'lumotda klassik m02 ko'pincha neyron m07/m08 dan
**yuqori** F1 beradi (neyron modellar ko'p ma'lumot talab qiladi). Inference vaqti:
m08 (LSTM) odatda eng sekin. Yorliqlar faqat **`ijobiy`/`salbiy`**.

---

## 2-topshiriq — m06 embeddinglari: pretrained vs random (tadqiqot)

`m06 CustomWord2Vec` o'z korpusingizda o'rgatilgan so'z vektorlarini beradi. RNN/LSTM
ni **pretrained** (m06) embeddinglar bilan boshlash, tasodifiy (random) initdan
yaxshiroq bo'lishi mumkin.

**Qadamlar (tadqiqot/mulohaza):**

1. `m06` ni korpusda o'qiting; `embed("toshkent")` va `most_similar("toshkent")` ni ko'ring
   — vektorlar mazmunlimi (yaqin so'zlar yaqinmi)?
2. **Mulohaza qiling:** agar m07/m08 ning Embedding qatlamini m06 vektorlari bilan
   boshlasak (random o'rniga), kichik korpusda natija qanday o'zgarishi mumkin? Nega?

> Eslatma: m07/m08 ning joriy shartnomasi pretrained-embedding parametrini olmaydi
> (arxitektura o'zgartirilmaydi). Bu topshiriq — **tahliliy**: m06 vektorlari sifatini
> baholang va pretrained init ta'sirini muhokama qiling.

---

## O'z-o'zini tekshirish

Modullaringiz shartnomaga (`capstone/contracts.py`) mos ekanini va taqqoslov
ishlayotganini quyidagi skript bilan tekshiring:

```bash
python course/milestones/w3_check.py
```

Skript m01–m08 ni hamda klassifikator taqqoslovini (m02/m07/m08 bir xil split'da
F1 + inference) avtomatik sinaydi, har tekshiruv uchun **✓** chiqaradi va oxirida
"hammasi o'tdi" xabarini beradi. Biror tekshiruv `FAIL` bersa — xabardagi ko'rsatmaga
amal qilib, tegishli modulni tuzating.

---

## Yozma mulohaza (qisqa)

Topshiriq bilan birga **3–5 jumlalik** mulohaza yozing:

> Sizning ma'lumotingizda qaysi klassifikator eng yuqori F1 berdi — klassik (m02) yoki
> neyron (m07/m08)? Bu kutilganmi? Inference vaqti bo'yicha qaysi biri eng tez/sekin?
> m06 embeddinglari mazmunli yaqinliklarni topdimi? Kichik korpus uchun qaysi yondashuvni
> tavsiya qilasiz?

---

## Topshirish

- **Muddat:** 6-iyul 2026, dushanba, 09:30 (3-hafta yakuni).
- **Nima topshiriladi:** (1) klassifikator taqqoslov natijasi (F1 + inference jadvali),
  (2) m06 embedding tadqiqoti va mulohazasi, (3) `w3_check.py` muvaffaqiyatli o'tgani
  (ekran tasviri yoki natija), (4) yozma mulohaza.
- **Qanday:** o'z kapstone repozitoriyingizga (`nlp-course-capstone`) joylang va
  havolani ulashing.

> **Eslatma:** bu milestone neyron arxitekturalar (m01–m08) integratsiyasiga taalluqli.
> NER (m10) keyingi kun (P10) da quriladi; transformer/RAG/agent modullari (m11+)
> **4-hafta milestone (w4)** da birlashtiriladi. Generatsiya (m09) va POS (m05b) —
> pedagogik, bu milestone qamrovida emas.
