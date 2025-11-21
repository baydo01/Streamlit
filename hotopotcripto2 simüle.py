import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Trader: Train/Test & Forecast", layout="wide")
st.title("🧠 AI Trader: Self-Optimizing Model & Forecasting")
st.markdown("""
Bu model:
1. **Train Data:** Geçmiş verilerle piyasa rejimlerini (HMM) öğrenir.
2. **Test Data:** Son 2 ayı (görmediği veriyi) simüle eder.
3. **Optimizasyon:** Test verisinde en yüksek kârı getiren **Ağırlık Kombinasyonunu** (AI vs Trend) kendi seçer.
4. **Forecasting:** Bir sonraki mum (saat/gün) için yön tahmini yapar.
""")

# --- AYARLAR ---
with st.sidebar:
    st.header("⚙️ Parametreler")
    tickers = st.multiselect("Coinler", ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "XRP-USD"], default=["BTC-USD", "ETH-USD"])
    interval = st.selectbox("Zaman Dilimi", ["1h", "1d"], index=1, help="1h seçerseniz son 730 gün verisi gelir.")
    test_window = st.number_input("Test Periyodu (Gün)", value=60, help="Son kaç gün Test verisi olsun?")
    capital = st.number_input("Başlangıç Sermayesi ($)", value=1000)
    
    st.markdown("---")
    st.info("Model, volatiliteyi HMM içine özellik olarak alır.")

# --- MATEMATİKSEL SKOR MOTORU (TREND AĞIRLIKLI) ---
def calculate_technical_score(df):
    """
    Geleneksel indikatörler. HMM (Rejim) ile birleştirilmek üzere skor üretir.
    Son dönem trendine daha duyarlıdır.
    """
    if len(df) < 50: return pd.Series(0, index=df.index)
    
    # 1. EMA Cross (Kısa Vadeli Trend)
    ema_short = df['close'].ewm(span=9, adjust=False).mean()
    ema_long = df['close'].ewm(span=21, adjust=False).mean()
    trend = np.where(ema_short > ema_long, 1, -1)
    
    # 2. Momentum (RSI Benzeri Hız)
    momentum = df['close'].pct_change(14).fillna(0) * 10 # Katsayı ile büyüt
    
    # 3. Volatilite Bazlı Ağırlık (Volatilite artarsa trende güven azalır)
    vol = df['close'].pct_change().rolling(20).std()
    # Volatilite düşükse trend sinyali güçlüdür, yüksekse zayıflat.
    vol_factor = 1 / (1 + (vol * 10))
    
    # Toplam Skor (-1 ile +1 arası normalize etmeye çalışıyoruz ama taşabilir)
    score = (trend * 0.6) + (momentum * 0.4)
    return pd.Series(score, index=df.index) * vol_factor

# --- CORE STRATEJİ FONKSİYONU ---
def run_advanced_simulation(ticker, interval, test_days, cap):
    # 1. VERİ ÇEKME
    # Yfinance limitleri: 1h verisi max 730 gün geriye gider.
    period = "2y" if interval == "1h" else "max"
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        # MultiIndex temizliği
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        # Eksik sütun tamamlama
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        df.dropna(inplace=True)
        
        if len(df) < 200: return None
    except: return None

    # 2. FEATURE ENGINEERING (Özellik Mühendisliği)
    # Log Return (Getiri)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    # Range (Volatilite temsili) -> HMM buna bayılır
    df['range'] = (df['high'] - df['low']) / df['close']
    # Teknik Skor
    df['tech_score'] = calculate_technical_score(df)
    
    df.dropna(inplace=True)

    # 3. TRAIN / TEST SPLIT
    split_date = df.index[-1] - timedelta(days=test_days)
    train_data = df[df.index <= split_date].copy()
    test_data = df[df.index > split_date].copy()
    
    if len(train_data) < 100 or len(test_data) < 10: return None

    # 4. HMM MODEL EĞİTİMİ (Sadece Train Data Üzerinde)
    # Özellikler: Getiri ve Volatilite (Range)
    X_train = train_data[['log_ret', 'range']].values
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    # Model: 3 Bileşenli (Ayı, Boğa, Yatay/Kararsız)
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
    try:
        model.fit(X_train_s)
    except: return None
    
    # Rejimlerin Anlamlandırılması (Hangi State Boğa?)
    means = model.means_[:, 0] # log_ret ortalamaları
    bull_state = np.argmax(means)
    bear_state = np.argmin(means)
    
    # 5. PREDICTION (Test Data Üzerinde)
    # Test verisini, Train'in scaler'ı ile dönüştür
    X_test = test_data[['log_ret', 'range']].values
    X_test_s = scaler.transform(X_test)
    hidden_states = model.predict(X_test_s)
    test_data['state'] = hidden_states
    
    # 6. OPTİMİZASYON DÖNGÜSÜ (Self-Betterment)
    # HMM (Yapay Zeka) ve Teknik Skor arasında en iyi ağırlığı bul.
    # weights = [0.0 (Sadece Teknik), 0.5 (Eşit), 1.0 (Sadece AI)]
    best_roi = -999
    best_w_hmm = 0.5
    best_equity = []
    
    possible_weights = np.arange(0.0, 1.1, 0.1) # 0.0, 0.1, ... 1.0
    
    # HODL Eğrisi (Kıyaslama için)
    hodl_return = (test_data['close'].iloc[-1] - test_data['close'].iloc[0]) / test_data['close'].iloc[0]
    
    for w in possible_weights:
        cash = cap
        coin = 0
        temp_equity = []
        
        for idx, row in test_data.iterrows():
            # HMM Sinyali (+1, -1, 0)
            hmm_sig = 1 if row['state'] == bull_state else (-1 if row['state'] == bear_state else 0)
            
            # Teknik Sinyal
            tech_sig = row['tech_score']
            
            # Ağırlıklı Karar
            decision = (w * hmm_sig) + ((1-w) * tech_sig)
            
            price = row['close']
            
            # İşlem (Basit threshold)
            if decision > 0.2 and cash > 0: # AL
                coin = cash / price
                cash = 0
            elif decision < -0.2 and coin > 0: # SAT
                cash = coin * price
                coin = 0
            
            val = cash + (coin * price)
            temp_equity.append(val)
        
        final_val = temp_equity[-1]
        roi = (final_val - cap) / cap
        
        if roi > best_roi:
            best_roi = roi
            best_w_hmm = w
            best_equity = temp_equity

    # 7. FORECASTING (GELECEK TAHMİNİ)
    # Son durum (state) nedir?
    last_state = hidden_states[-1]
    # Geçiş Matrisinden bir sonraki adım olasılıklarını al
    next_prob = model.transmat_[last_state]
    # En yüksek olasılıklı bir sonraki durum
    next_state = np.argmax(next_prob)
    
    # Yorumla
    forecast_text = "YATAY/BELİRSİZ"
    forecast_color = "gray"
    prob_val = next_prob[next_state]
    
    if next_state == bull_state:
        forecast_text = "YÜKSELİŞ (BULL)"
        forecast_color = "green"
    elif next_state == bear_state:
        forecast_text = "DÜŞÜŞ (BEAR)"
        forecast_color = "red"
        
    forecast_info = {
        "current_state": "Boğa" if last_state == bull_state else ("Ayı" if last_state == bear_state else "Yatay"),
        "next_prediction": forecast_text,
        "confidence": prob_val,
        "color": forecast_color
    }

    return {
        "ticker": ticker,
        "test_dates": test_data.index,
        "equity_curve": best_equity,
        "best_roi": best_roi,
        "hodl_roi": hodl_return,
        "best_weight": best_w_hmm,
        "forecast": forecast_info
    }

# --- ARAYÜZ ---
if st.button("🧪 Laboratuvarı Çalıştır (Train/Test + Forecast)"):
    if not tickers:
        st.error("Coin seçmelisin.")
    else:
        cols = st.columns(len(tickers))
        
        for i, ticker in enumerate(tickers):
            with cols[i]:
                st.markdown(f"### {ticker}")
                with st.spinner("Modelleniyor..."):
                    res = run_advanced_simulation(ticker, interval, test_window, capital)
                
                if res:
                    # METRİKLER
                    bot_kar = res['best_roi'] * 100
                    hodl_kar = res['hodl_roi'] * 100
                    
                    st.metric("Test ROI (Bot)", f"%{bot_kar:.2f}", delta=f"{bot_kar - hodl_kar:.2f}% vs HODL")
                    st.caption(f"Optimum Yapı: %{int(res['best_weight']*100)} Yapay Zeka + %{int((1-res['best_weight'])*100)} Teknik Analiz")
                    
                    # FORECAST KUTUSU
                    fc = res['forecast']
                    st.markdown(f"""
                    <div style="padding:10px; border-radius:5px; background-color:rgba(255,255,255,0.1); border:1px solid {fc['color']};">
                        <strong>🔮 Gelecek Tahmini ({interval}):</strong><br>
                        <span style="color:{fc['color']}; font-size:1.2em; font-weight:bold;">{fc['next_prediction']}</span><br>
                        <small>Güven: %{fc['confidence']*100:.1f} | Şu an: {fc['current_state']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # GRAFİK
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=res['test_dates'], y=res['equity_curve'], mode='lines', name='Bot', line=dict(color='#00ff00')))
                    # HODL çizgisini yaklaşık olarak çizelim (Sadece başlangıç ve bitiş noktası referansı)
                    start_p = res['equity_curve'][0]
                    end_p = start_p * (1 + res['hodl_roi'])
                    fig.add_trace(go.Scatter(x=[res['test_dates'][0], res['test_dates'][-1]], y=[start_p, end_p], name='HODL (Ref)', line=dict(dash='dot', color='white')))
                    
                    fig.update_layout(
                        title="Test Verisi Performansı",
                        margin=dict(l=0, r=0, t=30, b=0),
                        height=300,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("Yetersiz veri veya model hatası.")
