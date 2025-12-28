# 🇺🇿 Uzbek NLP Course (O'zbek tilidagi matnlarni tahlil qilish kursi)

Ushbu loyiha o'zbek tilidagi matnlarini qayta ishlash, tahlil qilish va mashinaviy o'qitish (Machine Learning) modellarini qo'llash uchun yaratilgan to'liq NLP (Natural Language Processing) (pipeline) jarayoni.

Loyiha bosqichma-bosqich rivojlanib boradi va har bir bosqich alohida mavzuni qamrab oladi.

## Loyiha Rejasi (Roadmap)

- [x] **1-Mavzu: Matnga Ishlov Berish (Preprocessing)**
    - HTML teglarni tozalash va matnni normallashtirish.
    - O'zbek tili uchun maxsus `Stemming` va `Lemmatization` (UzMorphAnalyser).
    - Matnni raqamli vektorlarga o'tkazish (TF-IDF, BoW).
- [ ] **2-Mavzu: Matnni Tasniflash (Text Classification)** *(Tez kunda)*
    - Yangiliklarni kategoriyalarga (Sport, Siyosat, Iqtisod) ajratish.
    - Logistic Regression va Naive Bayes modellarini o'qitish.
- [ ] **3-Mavzu: So'z Embedinglari (Word Embeddings)** *(Rejada)*
    - Word2Vec va FastText modellarini qo'llash.

---

## 1-Mavzu: Preprocessing va Vektorizatsiya

Birinchi bosqichda biz "xom" (raw) ma'lumotlarni tozalab, ularni mashina tushunadigan formatga keltirdik.

### Asosiy Xususiyatlari:
* **Custom Normalizer:** HTML kodlarni (`<div>`) va internetdagi noto'g'ri belgilarni (masalan, `5,9` yoki egri `‘` apostroflar) tozalaydi.
* **Lemmatization:** So'zlarning o'zagini aniqlash uchun `UzMorphAnalyser` kutubxonasidan foydalanildi (masalan: `maktablarimizda` -> `maktab`).
* **Vectorization:** Matnlar `scikit-learn` yordamida `Bag-of-Words` va `TF-IDF` matritsalariga aylantirildi va saqlandi.

### O'rnatish va Ishga Tushirish

Loyihani o'z kompyuteringizda ishga tushirish uchun quyidagi qadamlarni bajaring:

**1. Kerakli kutubxonalarni o'rnatish:**
```bash
pip install -r requirements.txt