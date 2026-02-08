# 🛡️ Financial Fraud Detection System (Bank-Level AI Analysis)

Bu proje, bankacılık işlemlerinde gerçekleşen dolandırıcılık (Fraud) girişimlerini tespit etmek amacıyla geliştirilmiş hibrit bir güvenlik sistemidir. **1.2 Milyon satırlık** gerçek işlem verisi üzerinde eğitilen Yapay Zeka (Random Forest) modeli ve Kural Tabanlı (Rule-Based) güvenlik politikaları birleştirilerek, gerçek dünya senaryolarına uygun bir dashboard tasarlanmıştır.

## 🚀 Proje Hakkında
Finansal güvenliği sağlamak adına sadece modelin doğruluğuna (Accuracy) değil, **İş Mantığına (Business Logic)** ve **Risk Yönetimine** odaklanılmıştır.

* **Veri Seti:** 1.296.675 İşlem Kaydı (Kaggle Fraud Detection Dataset)
* **Model:** Random Forest Classifier (Class Weight Balanced)
* **Arayüz:** Streamlit (Python)
* **Risk Yönetimi:** Dinamik Eşik (Dynamic Thresholding) + Sert Kurallar (Hard Rules)

## 📊 Veri Analizi ve Kritik Bulgular (Data Storytelling)
Proje geliştirme sürecinde veri seti üzerinde yapılan analizlerde şu kritik içgörüler elde edilmiştir:

### 1. Dengesiz Veri (Imbalanced Data)
Veri setindeki işlemlerin sadece **%0.6'sı** dolandırıcılık içermektedir.
* **Çözüm:** Model eğitilirken `class_weight='balanced'` parametresi kullanılarak, azınlık sınıfının (hırsızların) ağırlığı artırılmış ve modelin onları gözden kaçırması engellenmiştir.

### 2. Demografik Yanlılık (Algorithmic Bias) ⚠️
Veri setinin yapısı gereği, modelin **"Erkek"** kullanıcıları **"Kadın"** kullanıcılara göre istatistiksel olarak daha güvenli (Düşük Riskli) algıladığı tespit edilmiştir.
* **Gözlem:** Aynı şüpheli işlem (Gece 03:00, İnternet Alışverişi), kadın kullanıcıda **%54 Risk** verirken, erkek kullanıcıda **%24 Risk** vermektedir.
* **Alınan Önlem:** Bu yanlılığı (Bias) kırmak için sisteme **Yapay Zeka Skorundan bağımsız çalışan Sert Kurallar (Hard Rules)** eklenmiştir. Örneğin; "Gece yarısı yüksek tutarlı internet alışverişi yapan herkes, cinsiyet fark etmeksizin bloke edilir."

### 3. Hassasiyet Dengesi (Precision-Recall Tradeoff)
Standart AI modelleri %50 olasılık üzerini "Riskli" kabul eder. Ancak finans sektöründe %20 risk bile kabul edilemezdir.
* **Uygulama:** Projede karar eşiği (Threshold) **0.50'den 0.20'ye** çekilmiştir. Böylece "Şüpheli ama Temiz Görünen" işlemler de (Sarı Bölge) denetime takılarak güvenlik sıkılaştırılmıştır.

## 🛠️ Kurulum ve Kullanım

1.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install pandas numpy scikit-learn streamlit joblib
    ```
2.  Uygulamayı başlatın:
    ```bash
    streamlit run app.py
    ```

## 📈 Gelecek Geliştirmeler (Future Work)
Gerçek bir bankacılık entegrasyonunda şu adımlar atılmalıdır:
* **Fairness Constraints:** Cinsiyet gibi hassas veriler model eğitiminden çıkarılarak "Adil AI" prensipleri uygulanmalı.
* **SMOTE (Oversampling):** Dolandırıcılık verileri sentetik olarak çoğaltılarak model eğitimi dengelenmeli.
* **Real-time API:** Model bir REST API (FastAPI) olarak servise açılmalı.

---
**Geliştirici:** Göktuğ Demir
*Yönetim Bilişim Sistemleri (YBS) | Veri Analitiği & Siber Güvenlik*