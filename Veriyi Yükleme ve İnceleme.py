import pandas as pd

# 1. Dosyaları yükleyelim (Dosyaların kodla aynı klasörde olduğundan emin ol)
# Eğer hata alırsan dosya yolunu tam yaz (Örn: "C:/Users/Goktug/Downloads/fraud_train.csv")
print("Dosyalar okunuyor, biraz bekletebilir...")
train_df = pd.read_csv('fraudTrain.csv')
test_df = pd.read_csv('fraudTest.csv')

# 2. Verinin ilk 5 satırına bakalım (Ne varmış içinde?)
print("\n--- İLK 5 SATIR ---")
print(train_df.head())

# 3. Sütun isimleri ve veri tipleri (Sayısal mı, yazı mı?)
print("\n--- VERİ BİLGİSİ ---")
print(train_df.info())

# 4. En Kritik Kısım: Kaç tane dolandırıcılık (Fraud) var?
# Genelde hedef sütun 'is_fraud' veya 'Class' olur. 
# Kaggle setinde genelde 'is_fraud'dur.
if 'is_fraud' in train_df.columns:
    print("\n--- FRAUD DAĞILIMI ---")
    print(train_df['is_fraud'].value_counts())
    print(f"\nDolandırıcılık Oranı: %{train_df['is_fraud'].mean() * 100:.4f}")
else:
    print("\n'is_fraud' sütunu bulunamadı, sütun isimlerini kontrol et!")