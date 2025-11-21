import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Kalman AI Trader", layout="wide")
st.title("🛡️ Kalman AI: Gürültüsüz Trend & Dinamik Zaman")
st.markdown("""
Bu model, bir önceki "Timeframe Master" yapısını **Kalman Filtresi** ile güçlendirir.
1.  **Kalman Filtresi:** Fiyattaki anlık sapmaları (gürültü) temizler, gerçek rotayı çizer.
2.  **Dinamik Pencere:** Aylık grafikte son 30 ayı, Günlük grafikte son 30 günü baz alır.
3.  **Turnuva:** Her coin için en temiz sinyali veren zaman dilimini otomatik seçer.
""")

# --- AYARLAR ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    default_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "AVAX-USD", "DOGE-USD"]
    selected_tickers = st.multiselect("Sepet", default_tickers, default=["BTC-USD", "ETH-USD", "SOL-USD"])
    
    capital = st.number_input("Coin Başı Başlangıç ($)", value=10.0)
    
    # "Dinamik Pencere" boyutu.
    # Aylık seçilirse son 30 ay, Günlük seçilirse son 30 gün eğitim verisi olur.
    window_size = st.slider("Öğrenme Penceresi (Bar Sayısı)", 20, 100, 30) 

# --- KALMAN FİLTRESİ (MATEMATİKSEL MOTOR) ---
def apply_kalman_filter(prices):
    """
    Basitleştirilmiş 1D Kalman Filtresi.
    Fiyat serisini pürüzsüzleştirir (Denoising).
    """
    # Başlangıç parametreleri
    n_iter = len(prices)
    sz = (n_iter,) # size
    
    # Q: Process variance (Sistemin hatası)
    # R: Measurement variance (Ölçüm hatası - Gürültü)
    Q = 1e-5 
    R = 0.01**2 

    # Başlangıç tahminleri
    xhat = np.zeros(sz)      # Posteriori estimate
    P = np.zeros(sz)         # Posteriori error estimate
    xhatminus = np.zeros(sz) # Priori estimate
    Pminus = np.zeros(sz)    # Priori error estimate
    K = np.zeros(sz)         # Kalman gain

    xhat[0] = prices.iloc[0]
    P[0] = 1.0

    for k in range(1, n_iter):
        # Time Update (Prediction)
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q

        # Measurement Update (Correction)
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (prices.iloc[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
        
    return pd.Series(xhat, index=prices.index)

# --- VERİ İŞLEME ---
def get_raw_data(ticker):
    try:
        # Max veri alıyoruz, içeride kırpacağız
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        return df
    except: return None

def process_data(df, timeframe):
    if df is None or len(df) < 100: return None
    
    # 1. RESAMPLING (Zaman Dilimi Dönüşümü)
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    
    if timeframe == 'W':
        df_res = df.resample('W').agg(agg_dict).dropna()
    elif timeframe == 'M':
        df_res = df.resample('ME').agg(agg_dict).dropna()
    else:
        df_res = df.copy()
    
    if len(df_res) < 50: return None

    # 2. KALMAN FİLTRESİ UYGULAMA (Gürültü Temizliği)
    # Ham kapanış fiyatı yerine Kalman'lanmış fiyatı kullanmak trendi netleştirir.
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'])
    
    # 3. FEATURE ENGINEERING (Kalman Fiyatı Üzerinden)
    # Log Return (Kalman'a göre)
    df_res['log_ret'] = np.log(df_res['kalman_close'] / df_res['kalman_close'].shift(1))
    
    # Volatilite (Gerçek fiyata göre, risk gerçektir)
    df_res['range'] = (df_res['high'] - df_res['low']) / df_res['close']
    
    # Trend Sinyali (Fiyat Kalman'ın üstünde mi?)
    df_res['trend_signal'] = np.where(df_res['close'] > df_res['kalman_close'], 1, -1)
    
    # Target (Gelecek hareket)
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_res.dropna(inplace=True)
    
    return df_res

# --- MODELLER (HMM + RF + KALMAN TREND) ---
def get_signals(df, current_idx, n_hmm, d_rf, learn_window):
    # DİNAMİK PENCERE (Senin istediğin özellik)
    # Eğer df Günlük ise, learn_window=30 -> 30 Gün alır.
    # Eğer df Aylık ise, learn_window=30 -> 30 Ay alır.
    start = max(0, current_idx - learn_window)
    
    train_data = df.iloc[start:current_idx]
    curr_row = df.iloc[current_idx]
    
    if len(train_data) < 10: return 0
    
    # 1. HMM (Rejim)
    hmm_sig = 0
    try:
        X = train_data[['log_ret', 'range']].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = GaussianHMM(n_components=n_hmm, covariance_type="diag", n_iter=20, random_state=42)
        model.fit(X_s)
        bull = np.argmax(model.means_[:, 0])
        bear = np.argmin(model.means_[:, 0])
        
        curr_feat = scaler.transform(curr_row[['log_ret', 'range']].values.reshape(1, -1))
        probs = model.predict_proba(curr_feat)[0]
        hmm_sig = probs[bull] - probs[bear]
    except: pass
    
    # 2. Random Forest (İndikatör)
    rf_sig = 0
    try:
        features = ['log_ret', 'range', 'trend_signal']
        clf = RandomForestClassifier(n_estimators=30, max_depth=d_rf, random_state=42)
        clf.fit(train_data[features], train_data['target'])
        
        curr_feat = pd.DataFrame([curr_row[features]])
        prob = clf.predict_proba(curr_feat)[0][1]
        rf_sig = (prob - 0.5) * 2
    except: pass
    
    # 3. Kalman Trend
    # Fiyat, Pürüzsüz Kalman çizgisinin üzerindeyse trend yukarıdır.
    k_trend = curr_row['trend_signal']
    
    # AĞIRLIKLANDIRMA
    # Kalman trendi çok güçlü bir sinyaldir, ona biraz daha ağırlık verebiliriz.
    return (hmm_sig * 0.3) + (rf_sig * 0.3) + (k_trend * 0.4)

# --- SİMÜLASYON ---
def run_strategy(ticker, start_cap, win_size):
    raw_df = get_raw_data(ticker)
    if raw_df is None: return None
    
    # Son 4 yılın verisini alalım ki Aylık analizde yeterli veri olsun
    raw_df = raw_df.iloc[-1460:] 
    
    best_roi = -9999
    best_res = None
    
    timeframes = {'Günlük': 'D', 'Haftalık': 'W', 'Aylık': 'M'}
    
    for tf_name, tf_code in timeframes.items():
        df = process_data(raw_df, tf_code)
        if df is None: continue
        
        cash = start_cap
        coin = 0
        equity = []
        dates = []
        
        # Simülasyon
        for i in range(len(df)):
            # Pencere boyutu (win_size) burada devreye giriyor
            sig = get_signals(df, i, 3, 5, win_size)
            price = df['close'].iloc[i]
            
            # Eşikler (Kalman olduğu için sinyaller daha temiz, eşiği 0.25 tutabiliriz)
            if sig > 0.25 and cash > 0:
                coin = cash / price
                cash = 0
            elif sig < -0.25 and coin > 0:
                cash = coin * price
                coin = 0
            
            equity.append(cash + (coin * price))
            dates.append(df.index[i])
            
        final = equity[-1]
        roi = (final - start_cap) / start_cap
        
        if roi > best_roi:
            best_roi = roi
            # HODL Hesabı (O periyot için)
            start_p = df['close'].iloc[0]
            end_p = df['close'].iloc[-1]
            hodl_val = (start_cap / start_p) * end_p
            
            best_res = {
                'ticker': ticker,
                'tf': tf_name,
                'final': final,
                'roi': roi,
                'hodl': hodl_val,
                'equity': equity,
                'dates': dates,
                'kalman_data': df['kalman_close'] # Grafik için
            }
            
    return best_res

# --- ARAYÜZ ---
if st.button("🛡️ KALMAN DESTEKLİ ANALİZİ BAŞLAT"):
    cols = st.columns(2)
    prog = st.progress(0)
    
    results = []
    
    for i, t in enumerate(selected_tickers):
        with cols[i % 2]:
            with st.spinner(f"{t} için en iyi zaman ve Kalman filtresi hesaplanıyor..."):
                res = run_strategy(t, capital, window_size)
            
            if res:
                results.append(res)
                is_profit = res['roi'] > 0
                color = "#00ff00" if is_profit else "#ff4444"
                alpha = res['final'] - res['hodl']
                
                # KART
                st.markdown(f"""
                <div style="border: 1px solid {color}; padding: 15px; border-radius: 10px; margin-bottom: 10px; background-color: rgba(255,255,255,0.05)">
                    <div style="display:flex; justify-content:space-between;">
                        <h3 style="margin:0">{t}</h3>
                        <span style="background-color:#333; padding:2px 8px; border-radius:5px; font-size:0.8em;">🕒 {res['tf']}</span>
                    </div>
                    <div style="font-size:0.8em; color:gray; margin-top:5px;">Kalman Filtresi Aktif ✅</div>
                    <hr style="margin:5px 0; border-color:#444;">
                    <div style="display:flex; justify-content:space-between; align-items:end;">
                        <div>
                            <div style="font-size:0.8em; color:gray">Bot Sonuç</div>
                            <div style="font-size:1.5em; font-weight:bold; color:{color}">${res['final']:.2f}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:0.8em; color:gray">HODL</div>
                            <div style="font-size:1.1em;">${res['hodl']:.2f}</div>
                            <div style="font-size:0.8em; color:{'#0f0' if alpha>0 else '#f44'}">Fark: {alpha:+.2f}$</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # GRAFİK
                fig = go.Figure()
                # Bot Eğrisi
                fig.add_trace(go.Scatter(x=res['dates'], y=res['equity'], name="Bot Bakiye", line=dict(color=color, width=2)))
                # HODL Referans
                fig.add_trace(go.Scatter(x=[res['dates'][0], res['dates'][-1]], y=[capital, res['hodl']], name="HODL", line=dict(color="gray", dash="dot")))
                
                fig.update_layout(height=150, margin=dict(t=0,b=0,l=0,r=0), showlegend=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                
        prog.progress((i+1)/len(selected_tickers))
    
    prog.empty()
    
    if results:
        total_final = sum([r['final'] for r in results])
        total_hodl = sum([r['hodl'] for r in results])
        total_start = capital * len(results)
        
        st.markdown("---")
        st.markdown("### 🏆 PORTFÖY ÖZETİ")
        c1, c2, c3 = st.columns(3)
        c1.metric("Toplam Başlangıç", f"${total_start:.0f}")
        c2.metric("Kalman Bot Bitiş", f"${total_final:.2f}", f"%{((total_final-total_start)/total_start)*100:.1f}")
        c3.metric("HODL Bitiş", f"${total_hodl:.2f}", delta=f"${total_final - total_hodl:.2f}")

        # Bilgi Notu
        st.info(f"""
        ℹ️ **Sistem Nasıl Çalıştı?**
        1. **Dinamik Pencere:** Seçilen "{window_size}" değeri; Günlük grafikte son {window_size} günü, Aylık grafikte son {window_size} ayı analiz etti.
        2. **Gürültü Filtresi:** Kalman Filtresi, ani iğne atışlarını (wick) yoksaydı ve ana trende odaklandı.
        """)
