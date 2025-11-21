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

st.set_page_config(page_title="Hedge Fund: Ultimate AI", layout="wide")
st.title("🏆 Ultimate AI Trader: Turnuva + Doğrulama + Tahmin")
st.markdown("""
Bu sistem 3 katmanlı bir eleme yapar:
1. **Zaman Turnuvası:** Günlük (D), Haftalık (W), Aylık (M) verilerini yarıştırtır.
2. **Validasyon (Train/Test):** Geçmiş veriyi öğrenir (Train), son dönemi (Test) simüle eder.
3. **Optimizasyon:** AI ve Teknik Analiz arasındaki en iyi dengeyi bulur.
Sonuçta **en yüksek kârı getiren strateji** neyse onu raporlar.
""")

# --- AYARLAR ---
with st.sidebar:
    st.header("⚙️ Fon Ayarları")
    # Genişletilmiş Coin Listesi
    default_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "DOGE-USD", "TRX-USD", "LINK-USD"]
    selected_tickers = st.multiselect("Analiz Edilecek Coinler", default_tickers, default=["BTC-USD", "ETH-USD", "SOL-USD"])
    
    test_window_days = st.number_input("Test Periyodu (Gün)", value=90, help="Stratejinin son kaç gündeki performansına bakılsın?")
    capital = st.number_input("Coin Başı Sermaye ($)", value=1000)
    
    st.info("Not: Haftalık (W) veriler genellikle kriptoda daha temiz sinyal üretir.")

# --- YARDIMCI FONKSİYONLAR ---
def get_data(ticker):
    try:
        # En geniş veriyi alıp içeride resample yapacağız
        df = yf.download(ticker, period="max", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        return df
    except: return None

def calculate_features(df):
    """Teknik indikatörler ve HMM özellikleri"""
    df = df.copy()
    # HMM Özellikleri
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['range'] = (df['high'] - df['low']) / df['close']
    
    # Teknik Skor (Trend + Momentum)
    # EMA
    df['ema_s'] = df['close'].ewm(span=9).mean()
    df['ema_l'] = df['close'].ewm(span=21).mean()
    trend = np.where(df['ema_s'] > df['ema_l'], 1, -1)
    
    # RSI Benzeri Basit Momentum
    mom = df['close'].pct_change(14).fillna(0)
    
    # Volatilite (Ters orantı: Volatilite arttıkça teknik puana güven azalır)
    vol = df['close'].pct_change().rolling(10).std().fillna(0)
    vol_scaler = 1 / (1 + (vol*10))
    
    # Skor (-1 ile 1 arası kabaca)
    df['tech_score'] = ((trend * 0.6) + (np.sign(mom) * 0.4)) * vol_scaler
    
    df.dropna(inplace=True)
    return df

def fit_hmm(train_df):
    """Train datası üzerinde HMM eğitir"""
    X = train_df[['log_ret', 'range']].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    try:
        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
        model.fit(X_s)
        
        # Rejimleri belirle
        means = model.means_[:, 0]
        bull = np.argmax(means)
        bear = np.argmin(means)
        return model, scaler, bull, bear
    except:
        return None, None, None, None

def simulate_strategy(df, model, scaler, bull, bear, split_date, cap):
    """Test verisi üzerinde optimizasyon yapar"""
    test_data = df[df.index > split_date].copy()
    if len(test_data) < 5: return None
    
    # Tahminleri Yap
    X_test = test_data[['log_ret', 'range']].values
    X_test_s = scaler.transform(X_test)
    states = model.predict(X_test_s)
    test_data['state'] = states
    
    best_roi = -999
    best_equity = []
    best_w = 0.5
    
    # Ağırlık Optimizasyonu Döngüsü
    weights = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0] # 0: Full Teknik, 1: Full AI
    
    hodl_start = test_data['close'].iloc[0]
    hodl_end = test_data['close'].iloc[-1]
    hodl_roi = (hodl_end - hodl_start) / hodl_start
    
    for w in weights:
        cash = cap
        coin = 0
        equity = []
        
        for idx, row in test_data.iterrows():
            # AI Sinyali
            ai_sig = 1 if row['state'] == bull else (-1 if row['state'] == bear else 0)
            # Teknik Sinyal
            tech_sig = row['tech_score']
            
            decision = (w * ai_sig) + ((1-w) * tech_sig)
            
            price = row['close']
            if decision > 0.25 and cash > 0:
                coin = cash / price
                cash = 0
            elif decision < -0.25 and coin > 0:
                cash = coin * price
                coin = 0
            
            val = cash + (coin * price)
            equity.append(val)
            
        roi = (equity[-1] - cap) / cap
        if roi > best_roi:
            best_roi = roi
            best_equity = equity
            best_w = w
            
    # Forecasting (Gelecek Tahmini)
    last_state = states[-1]
    next_probs = model.transmat_[last_state]
    next_state_idx = np.argmax(next_probs)
    
    forecast_type = "YATAY"
    if next_state_idx == bull: forecast_type = "YÜKSELİŞ"
    elif next_state_idx == bear: forecast_type = "DÜŞÜŞ"
    
    prob = next_probs[next_state_idx]
    
    return {
        "roi": best_roi,
        "hodl_roi": hodl_roi,
        "equity": best_equity,
        "dates": test_data.index,
        "weight": best_w,
        "forecast": forecast_type,
        "prob": prob,
        "current_state": "Boğa" if last_state == bull else "Ayı" if last_state == bear else "Yatay"
    }

# --- ANA İŞLEYİCİ ---
def process_ticker(ticker, test_days, cap):
    raw_df = get_data(ticker)
    if raw_df is None or len(raw_df) < 365: return None
    
    # Zaman Dilimleri
    timeframes = {'Günlük (D)': 'D', 'Haftalık (W)': 'W-MON', 'Aylık (M)': 'ME'}
    
    champion_res = None
    champion_roi = -9999
    champion_tf = ""
    
    split_date = raw_df.index[-1] - timedelta(days=test_days)
    
    # TURNUVA DÖNGÜSÜ
    for tf_name, tf_code in timeframes.items():
        # Resample
        if tf_code == 'D':
            df_res = raw_df.copy()
        else:
            agg = {'close': 'last', 'high': 'max', 'low': 'min'}
            df_res = raw_df.resample(tf_code).agg(agg).dropna()
            
        if len(df_res) < 50: continue
        
        # Feature Engineering
        df_res = calculate_features(df_res)
        
        # Train Split
        train_df = df_res[df_res.index <= split_date]
        if len(train_df) < 30: continue
        
        # HMM Eğit
        model, scaler, bull, bear = fit_hmm(train_df)
        if model is None: continue
        
        # Simülasyon (Test)
        res = simulate_strategy(df_res, model, scaler, bull, bear, split_date, cap)
        if res is None: continue
        
        # Şampiyon Kontrolü
        if res['roi'] > champion_roi:
            champion_roi = res['roi']
            champion_res = res
            champion_tf = tf_name

    if champion_res:
        champion_res['tf_name'] = champion_tf
        champion_res['ticker'] = ticker
        return champion_res
    return None

# --- ARAYÜZ ---
if st.button("🚀 TAM TURNUVAYI BAŞLAT", type="primary"):
    if not selected_tickers:
        st.error("Lütfen coin seçin.")
    else:
        # Grid oluştur
        cols = st.columns(3)
        
        for i, ticker in enumerate(selected_tickers):
            col_idx = i % 3
            with cols[col_idx]:
                with st.spinner(f"{ticker} analiz ediliyor..."):
                    result = process_ticker(ticker, test_window_days, capital)
                
                if result:
                    # Kart Tasarımı
                    roi_pct = result['roi'] * 100
                    hodl_pct = result['hodl_roi'] * 100
                    alpha = roi_pct - hodl_pct
                    
                    # Renkler
                    card_color = "rgba(0, 255, 0, 0.1)" if roi_pct > 0 else "rgba(255, 0, 0, 0.1)"
                    border_color = "green" if roi_pct > 0 else "red"
                    fc_color = "green" if result['forecast'] == "YÜKSELİŞ" else ("red" if result['forecast'] == "DÜŞÜŞ" else "gray")
                    
                    st.markdown(f"""
                    <div style="border: 1px solid {border_color}; padding: 15px; border-radius: 10px; background-color: {card_color}; margin-bottom: 10px;">
                        <h3 style="margin:0;">{ticker}</h3>
                        <small>🏆 Şampiyon: <b>{result['tf_name']}</b> Grafiği</small>
                        <hr style="margin: 5px 0;">
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <div style="font-size:0.8em;">Bot Kârı</div>
                                <div style="font-size:1.5em; font-weight:bold; color:{'#0f0' if roi_pct>0 else '#f00'}">%{roi_pct:.1f}</div>
                            </div>
                            <div>
                                <div style="font-size:0.8em;">HODL Farkı</div>
                                <div style="font-size:1.2em; color:white;">{alpha:+.1f}%</div>
                            </div>
                        </div>
                        <div style="margin-top:10px; font-size:0.9em;">
                             ⚙️ Yapı: %{int(result['weight']*100)} AI + %{int((1-result['weight'])*100)} Teknik
                        </div>
                        <div style="margin-top:10px; background-color:rgba(0,0,0,0.3); padding:5px; border-radius:5px;">
                            🔮 <b>Tahmin ({result['tf_name']}):</b> <span style="color:{fc_color}; font-weight:bold;">{result['forecast']}</span><br>
                            <small>Güven: %{result['prob']*100:.1f} | Mevcut: {result['current_state']}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mini Grafik
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=result['dates'], y=result['equity'], mode='lines', line=dict(color='#00ff00', width=2)))
                    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=100, showlegend=False, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    fig.update_xaxes(visible=False)
                    fig.update_yaxes(visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"{ticker}: Yetersiz veri veya hata.")
