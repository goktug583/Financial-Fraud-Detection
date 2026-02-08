import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. VERİYİ YÜKLEME (DOSYALARI OKUMA) ---
print("Dosyalar okunuyor... (Biraz sabır ☕)")
# Dosyaların bu Python dosyasıyla AYNI KLASÖRDE olduğundan emin ol.
try:
    train_df = pd.read_csv('fraudTrain.csv')
    # test_df'i şimdilik kullanmıyoruz, train_df'i bölüp test edeceğiz.
    print("Veri başarıyla yüklendi! İşleme başlıyoruz...")
except FileNotFoundError:
    print("HATA: 'fraud_train.csv' dosyası bulunamadı! Dosya yolunu kontrol et.")
    exit() # Dosya yoksa durdur.

# --- 2. ÖZELLİK MÜHENDİSLİĞİ (TARİH VE SAAT AYIKLAMA) ---
# Tarihleri makinenin anlayacağı formata çeviriyoruz
train_df['trans_date_trans_time'] = pd.to_datetime(train_df['trans_date_trans_time'])
train_df['dob'] = pd.to_datetime(train_df['dob'])

# Saat ve Gün bilgisini çekelim (Gece yapılan işlemler şüpheli olabilir)
train_df['hour'] = train_df['trans_date_trans_time'].dt.hour
train_df['day_of_week'] = train_df['trans_date_trans_time'].dt.dayofweek

# Yaş bilgisini hesaplayalım (Gençler ve yaşlılar farklı harcama yapar)
train_df['age'] = train_df['trans_date_trans_time'].dt.year - train_df['dob'].dt.year

# --- 3. KATEGORİK VERİYİ SAYISALLAŞTIRMA ---
# Makine "Erkek/Kadın" veya "Market/Benzin" yazısını anlamaz, sayıya çeviriyoruz.
encoder = LabelEncoder()
train_df['category'] = encoder.fit_transform(train_df['category'])
train_df['gender'] = encoder.fit_transform(train_df['gender'])

# --- 4. GEREKSİZ SÜTUNLARI ATMA ---
# İsim, adres, fiş numarası gibi şeyler modelin ezberlemesine yol açar, atıyoruz.
drop_cols = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 
             'first', 'last', 'street', 'city', 'state', 'zip', 'job', 'dob', 
             'trans_num', 'unix_time', 'merch_lat', 'merch_long', 'lat', 'long']

# Sadece işimize yarayacak temiz veriyi alıyoruz
X = train_df.drop(columns=drop_cols + ['is_fraud']) # Girdi Verileri (Sorular)
y = train_df['is_fraud'] # Hedef (Cevaplar: 0 mı 1 mi?)

print("Model eğitime hazırlanıyor... (Veri büyük olduğu için 1-2 dakika sürebilir)")

# --- 5. EĞİTİM VE TEST AYRIMI ---
# Verinin %80'i ile ders çalışacak, %20'si ile sınava girecek.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 6. MODEL EĞİTİMİ (RANDOM FOREST) ---
# class_weight='balanced' diyerek "Azınlıkta olan dolandırıcıları önemse" diyoruz.
# n_jobs=-1 diyerek bilgisayarın tüm çekirdeklerini kullanmasını söylüyoruz (Hız için).
model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
model.fit(X_train, y_train)

print("Eğitim tamamlandı! Şimdi test ediliyor...")

# --- 7. SONUÇLARI GÖRME ---
y_pred = model.predict(X_test)

print("\n--- DOLANDIRICILIK TESPİT RAPORU ---")
print(classification_report(y_test, y_pred))

print("\n--- CONFUSION MATRIX (KARMAŞIKLIK MATRİSİ) ---")
# [[Normali Normal Bildi, Normali Fraud Sandı]
#  [Fraud'u Normal Sandı, Fraud'u Fraud Bildi]]
print(confusion_matrix(y_test, y_pred))

# Hangi özellik (Feature) dolandırıcılığı yakalamada en etkiliymiş?
feature_importances = pd.DataFrame(model.feature_importances_,
                                   index = X.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
print("\n--- EN ÖNEMLİ KRİTERLER (Fraud Belirtileri) ---")
print(feature_importances.head(5))


import joblib

# 1. Modeli kaydet (Zeka)
joblib.dump(model, 'fraud_model.pkl')

# 2. Encoder'ı kaydet (Çevirmen - Market yazısını 1'e çeviren şey)
joblib.dump(encoder, 'category_encoder.pkl')

print("✅ Model ve Encoder başarıyla kaydedildi! Şimdi Dashboard yapabiliriz.")