import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. MODELİ VE ARAÇLARI YÜKLE ---
try:
    model = joblib.load('fraud_model.pkl')
    encoder = joblib.load('category_encoder.pkl')
except FileNotFoundError:
    st.error("HATA: Model dosyaları bulunamadı! Lütfen önce eğitimi çalıştırın.")
    st.stop()

# --- 2. SAYFA AYARLARI ---
st.set_page_config(page_title="Bank-Level Fraud Detection", layout="wide", page_icon="🛡️")

st.title("🛡️ Finansal Güvenlik ve Dolandırıcılık Tespit Sistemi")
st.markdown("""
**Sistem Statüsü:** 🟢 Aktif | **Mod:** Banka Prodüksiyon Ortamı (Yüksek Hassasiyet)
Bu panel, Random Forest algoritması ve Kural Tabanlı (Rule-Based) güvenlik politikaları ile çalışır.
""")
st.divider()

# --- 3. SOL MENÜ (INPUT) ---
st.sidebar.header("💳 İşlem Simülasyonu")

amt = st.sidebar.number_input("İşlem Tutarı ($)", min_value=0.0, max_value=20000.0, value=150.0, step=10.0)
hour = st.sidebar.slider("İşlem Saati", 0, 23, 14)
age = st.sidebar.slider("Müşteri Yaşı", 18, 90, 30)

category_translation = {
    "Market/Gıda": "grocery_pos",
    "Akaryakıt": "gas_transport",
    "İnternet Alışverişi": "shopping_net",
    "Online Hizmetler": "misc_net",
    "Seyahat": "travel",
    "Eğlence": "entertainment"
}
selected_category_tr = st.sidebar.selectbox("Harcama Yeri", list(category_translation.keys()))
selected_category_en = category_translation[selected_category_tr]

try:
    cat_val = encoder.transform([selected_category_en])[0]
except:
    cat_val = 0 

gender_tr = st.sidebar.radio("Kart Sahibi Cinsiyet", ['Erkek', 'Kadın'])
gender_val = 1 if gender_tr == 'Erkek' else 0

# --- 4. ANALİZ MOTORU ---
if st.sidebar.button("İŞLEMİ DENETLE 🚀", type="primary"):
    
    # Varsayılan değerlerle (Şehir Nüfusu: 10k, Gün: Çarşamba) feature seti
    features = [[cat_val, amt, gender_val, 10000, hour, 2, age]]
    
    # 1. YAPAY ZEKA TAHMİNİ (OLASILIK)
    ai_risk_score = model.predict_proba(features)[0][1]
    
    # 2. BANKA GÜVENLİK POLİTİKALARI (SERT KURALLAR)
    bank_policy_block = False
    policy_reason = ""

    # Kural 1: Gece yarısı (00-05) 1000$ üzeri internet harcaması KESİN BLOKE
    if (0 <= hour <= 5) and amt > 1000 and "shopping_net" in selected_category_en:
        bank_policy_block = True
        policy_reason = "Gece Yarısı Yüksek Tutar Limiti"

    # Kural 2: Tutar 10.000$ üzeriyse EK ONAY GEREKİR
    if amt > 10000:
        bank_policy_block = True
        policy_reason = "Maksimum İşlem Limiti Aşıldı"

    # --- SONUÇ EKRANI ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📡 Karar Merkezi")
        
        # BANKA EŞİK DEĞERİ: %20 (0.20)
        # Bankalar riski sevmez. %20 bile şüpheliyse durdurur.
        THRESHOLD = 0.20 
        
        if bank_policy_block:
            st.error("⛔ İŞLEM REDDEDİLDİ (KURAL İHLALİ)")
            st.write(f"**Sebep:** {policy_reason}")
            st.warning("Yapay Zeka skoruna bakılmaksızın Banka Politikası gereği işlem durduruldu.")
            
        elif ai_risk_score > THRESHOLD:
            st.error("🚨 ŞÜPHELİ İŞLEM TESPİT EDİLDİ")
            st.metric("Risk Skoru", f"%{ai_risk_score*100:.2f}", delta="-Riskli")
            st.write(f"Sistem Eşiği (%{THRESHOLD*100}) aşıldı. Müşteriye SMS onayı gönderiliyor...")
            
        else:
            st.success("✅ İŞLEM ONAYLANDI")
            st.metric("Güven Skoru", f"%{(1-ai_risk_score)*100:.2f}", delta="+Güvenli")
            
    with col2:
        st.subheader("📊 Detaylı Analiz")
        st.info("Neden Bu Karar Verildi?")
        
        if ai_risk_score > 0.5:
            st.write("🔴 **Yapay Zeka:** İşlem deseni geçmişteki dolandırıcılıklarla yüksek oranda eşleşiyor.")
        elif ai_risk_score > 0.2:
            st.write("🟠 **Yapay Zeka:** İşlemde bazı anormallikler var (Saat veya Tutar uyumsuzluğu).")
        else:
            st.write("🟢 **Yapay Zeka:** İşlem müşterinin rutin harcamalarına uygun.")
            
        st.write(f"- **İncelenen Tutar:** {amt}$")
        st.write(f"- **Risk Faktörü:** {selected_category_tr}")