# 🛡️ Bank-Level Financial Fraud Detection System (AI + Rule Based)


Bu proje, bankacılık işlemlerinde gerçekleşen dolandırıcılık (Fraud) girişimlerini tespit etmek amacıyla geliştirilmiş, **Yapay Zeka** ve **Banka İş Kurallarını (Business Rules)** birleştiren hibrit bir güvenlik sistemidir.

Proje, 1.2 Milyon satırlık gerçek işlem verisi üzerinde eğitilmiş ve **Algoritmik Önyargı (Bias)** analizleri yapılarak "Gerçek Hayat" senaryolarına göre optimize edilmiştir.

---

## 🚀 Projenin Farkı Ne? (Executive Summary)
Standart makine öğrenmesi projelerinden farklı olarak, bu projede **modelin hataları analiz edilmiş ve iş mantığı (Business Logic) ile kapatılmıştır.**

* **Dinamik Eşik (Dynamic Thresholding):** Standart 0.50 yerine, bankacılık standartlarına uygun **0.20 (%20 Risk)** eşiği kullanılmıştır.
* **Hibrit Karar Motoru:** Yapay zekanın "Güvenli" dediği ancak risk barındıran işlemler için **Sert Kurallar (Hard Rules)** devreye alınmıştır.
* **Etik AI Analizi:** Veri setindeki cinsiyet yanlılığı (Gender Bias) tespit edilmiş ve raporlanmıştır.

---

## 🧠 Teknik Mimari ve Model
* **Veri Seti:** Kaggle Fraud Detection Dataset (1.296.675 İşlem)
* **Algoritma:** Random Forest Classifier (n_estimators=100, class_weight='balanced')
* **Arayüz:** Streamlit (Python)
* **Başarı Oranı:** Test setinde %97 Precision, ancak gerçek hayat simülasyonunda "False Negative"leri engellemek önceliklendirilmiştir.

---

## 🔍 Vaka Analizi: "Erkek Kullanıcı Anomalisi" (Case Study)
Proje geliştirme sürecinde kritik bir **Algoritmik Yanlılık (Bias)** keşfedilmiştir.

### 1. Sorun Tespiti
Model test edilirken, **aynı harcama koşullarında (Gece 03:00, İnternet Alışverişi, Yüksek Tutar)**:
* **Kadın Kullanıcı:** %54 Risk (Şüpheli) 🚨
* **Erkek Kullanıcı:** %24 Risk (Güvenli) ✅
sonucu verdiği görülmüştür.

### 2. Neden? (Root Cause Analysis)
Kullanılan sentetik veri setinde, erkek kullanıcıların dolandırıcılık oranları istatistiksel olarak düşük kodlandığı için, model "Erkek" olmayı güçlü bir "Güvenli İşlem" sinyali olarak öğrenmiştir. Bu durum, gerçek hayatta erkek dolandırıcıların kaçmasına (False Negative) sebep olabilir.

### 3. Çözüm: Hibrit Güvenlik Katmanı 🛡️
Modeli yeniden eğitmek yerine, gerçek bir banka ortamını simüle eden **"Sert Kurallar" (Hard Rules)** sisteme entegre edilmiştir.

**Uygulanan Kurallar:**
1.  **Gece Yarısı Kuralı:** Saat 00:00 - 05:00 arasında yapılan 1000$ üzeri *tüm* internet harcamaları, AI skoru ne olursa olsun **BLOKE** edilir.
2.  **Maksimum Tutar Kuralı:** 10.000$ üzeri işlemler AI'dan bağımsız olarak **Manuel Onay**'a düşer.
3.  **Tolerans Eşiği:** Risk skoru 0.20 (%20) üzerindeki her işlem "Şüpheli" olarak işaretlenir ve kullanıcıya SMS onayı (Simülasyon) gönderilir.

---

## 📊 Kullanım Senaryosu (Demo)

Sistem `app.py` üzerinden çalıştırıldığında interaktif bir Dashboard sunar:

1.  **Senaryo 1 (Normal İşlem):** Gündüz 14:00, Market Alışverişi, 50$ -> **✅ GÜVENLİ**
2.  **Senaryo 2 (AI Tespiti):** Gece 03:00, Kart Sahibi Kadın, İnternet, 500$ -> **🚨 RİSKLİ (AI Yakaladı)**
3.  **Senaryo 3 (Kural Tespiti):** Gece 03:00, Kart Sahibi Erkek, İnternet, 1200$ -> **⛔ BLOKE (Kural Yakaladı)**
    *(Yapay zeka bunu güvenli saysa bile, yazdığımız kural motoru işlemi durdurur.)*

---

## 🛠️ Kurulum

```bash
# 1. Depoyu klonlayın
git clone [https://github.com/KULLANICI_ADINIZ/Financial-Fraud-Detection.git](https://github.com/KULLANICI_ADINIZ/Financial-Fraud-Detection.git)

# 2. Gereksinimleri yükleyin
pip install -r requirements.txt

# 3. Uygulamayı başlatın
streamlit run app.py
