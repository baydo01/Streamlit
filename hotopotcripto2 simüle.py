import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Timeframe Master AI", layout="wide")
st.title("⏳ Timeframe Master: Otomatik Zaman Seçimli AI")
st.markdown("""
Bu bot senin için şu soruyu çözer: **"Bu coine Günlük mü bakmalıyım, Haftalık mı?"**
1.  **Veri Çekme:** Ham veriyi alır.
2.  **Turnuva:** Günlük (D), Haftalık (W) ve Aylık (M) grafikleri oluşturur ve geçmişte test eder.
3.  **Seçim:** En yüksek kârı getiren zaman dilimini (Timeframe) ve stratejiyi seçer.
4.  **Uygulama:** Parayı o grafiğe göre yönetir.
""")

# --- AYARLAR ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    default_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "AVAX-USD", "DOGE-USD", "LINK-USD"]
    selected_tickers = st.multiselect("Sepet", default_tickers, default=["BTC-USD", "ETH-USD", "SOL-USD"])
    
    capital = st.number_input("Coin Başı Başlangıç ($)", value=10.0)
    
    # Test süresini uzun tutalım ki Haftalık/Aylık verilerde anlamlı olsun
    lookback_days = st.slider("Analiz Geçmişi (Gün)", 365, 1095, 730) 

# --- VERİ İŞLEME MOTORU (RESAMPLING DAHİL) ---
def get_raw_data(ticker):
    try:
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        return df
    except: return None

def process_data(df, timeframe):
    """
    Veriyi istenen zaman dilimine (D/W/M) çevirir ve indikatörleri ekler.
    """
    if df is None or len(df) < 50: return None
    
    # Resampling (Yeniden Örnekleme)
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    
    if timeframe == 'W':
        df_res = df.resample('W').agg(agg_dict).dropna()
    elif timeframe == 'M':
        df_res = df.resample('ME').agg(agg_dict).dropna()
    else: # Daily
        df_res = df.copy()
    
    if len(df_res) < 30: return None # Yetersiz veri
    
    # Feature Engineering (Zaman dilimine göre dinamik)
    # Log Return
    df_res['log_ret'] = np.log(df_res['close'] / df_res['close'].shift(1))
    
    # Range (Volatilite)
    df_res['range'] = (df_res['high'] - df_res['low']) / df_res['close']
    
    # Trend (SMA)
    df_res['ma_short'] = df_res['close'].rolling(10).mean()
    df_res['dist_ma'] = (df_res['close'] - df_res['ma_short']) / df_res['ma_short']
    
    # Momentum (RSI benzeri basit)
    df_res['mom'] = df_res['close'].pct_change(4) # 4 bar öncesine göre değişim
    
    # Target (Gelecek 1 bar artacak mı?)
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_res.dropna(inplace=True)
    
    return df_res

# --- MODELLER (HMM + RF + TREND) ---
def get_signals(df, current_idx, n_hmm, d_rf):
    # Rolling Window: Geçmiş 50 bar (Gün/Hafta/Ay neyse)
    start = max(0, current_idx - 50)
    train_data = df.iloc[start:current_idx]
    curr_row = df.iloc[current_idx]
    
    if len(train_data) < 20: return 0
    
    # 1. HMM Sinyali
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
    
    # 2. Random Forest Sinyali
    rf_sig = 0
    try:
        features = ['log_ret', 'range', 'dist_ma', 'mom']
        clf = RandomForestClassifier(n_estimators=30, max_depth=d_rf, random_state=42)
        clf.fit(train_data[features], train_data['target'])
        
        curr_feat = pd.DataFrame([curr_row[features]])
        prob = clf.predict_proba(curr_feat)[0][1]
        rf_sig = (prob - 0.5) * 2
    except: pass
    
    # 3. Basit Trend
    trend_sig = 1 if curr_row['close'] > curr_row['ma_short'] else -1
    
    # EŞİT AĞIRLIKLI ORTALAMA (Basit ve Etkili)
    return (hmm_sig + rf_sig + trend_sig) / 3.0

# --- TURNUVA VE SİMÜLASYON ---
def run_strategy(ticker, start_cap, history_days):
    raw_df = get_raw_data(ticker)
    if raw_df is None: return None
    
    # Son 'history_days' kadar veriyi al
    raw_df = raw_df.iloc[-history_days:]
    
    best_roi = -9999
    best_result = None
    best_config = ""
    
    # --- 1. TURNUVA: ZAMAN DİLİMLERİ ---
    timeframes = {'Günlük (D)': 'D', 'Haftalık (W)': 'W', 'Aylık (M)': 'M'}
    
    for tf_name, tf_code in timeframes.items():
        # Veriyi hazırla
        df = process_data(raw_df, tf_code)
        if df is None: continue
        
        # Validasyon (Test) Simülasyonu
        cash = start_cap
        coin = 0
        equity = []
        dates = []
        
        # Basit parametrelerle hızlı test (3 State HMM, 5 Depth RF)
        for i in range(len(df)):
            sig = get_signals(df, i, 3, 5)
            price = df['close'].iloc[i]
            
            # İşlem
            if sig > 0.2 and cash > 0:
                coin = cash / price
                cash = 0
            elif sig < -0.2 and coin > 0:
                cash = coin * price
                coin = 0
            
            equity.append(cash + (coin * price))
            dates.append(df.index[i])
            
        final_val = equity[-1]
        roi = (final_val - start_cap) / start_cap
        
        # Eğer bu zaman dilimi daha iyiyse, Şampiyon yap
        if roi > best_roi:
            best_roi = roi
            best_result = {
                'ticker': ticker,
                'final': final_val,
                'roi': roi,
                'equity': equity,
                'dates': dates,
                'tf_name': tf_name
            }
            
            # HODL Hesabı (O zaman diliminin başı ve sonuna göre)
            start_p = df['close'].iloc[0]
            end_p = df['close'].iloc[-1]
            hodl_val = (start_cap / start_p) * end_p
            best_result['hodl_val'] = hodl_val
            best_result['hodl_roi'] = (hodl_val - start_cap) / start_cap

    return best_result

# --- ARAYÜZ ---
if st.button("⏳ ZAMAN MAKİNESİNİ ÇALIŞTIR"):
    cols = st.columns(2)
    results = []
    
    progress = st.progress(0)
    
    for i, t in enumerate(selected_tickers):
        with cols[i % 2]:
            with st.spinner(f"{t} için en iyi zaman dilimi aranıyor..."):
                res = run_strategy(t, capital, lookback_days)
            
            if res:
                results.append(res)
                
                # Renkler
                is_profit = res['roi'] > 0
                color = "#00ff00" if is_profit else "#ff4444"
                alpha = res['final'] - res['hodl_val']
                
                # KART GÖRÜNÜMÜ
                st.markdown(f"""
                <div style="border: 1px solid {color}; padding: 15px; border-radius: 10px; margin-bottom: 10px; background-color: rgba(255,255,255,0.05)">
                    <div style="display:flex; justify-content:space-between;">
                        <h3 style="margin:0">{t}</h3>
                        <span style="background-color:#333; padding:2px 8px; border-radius:5px; font-size:0.8em;">🏆 {res['tf_name']}</span>
                    </div>
                    <hr style="margin:5px 0; border-color:#444;">
                    <div style="display:flex; justify-content:space-between; align-items:end;">
                        <div>
                            <div style="font-size:0.8em; color:gray">Bot Sonuç</div>
                            <div style="font-size:1.5em; font-weight:bold; color:{color}">${res['final']:.2f}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:0.8em; color:gray">HODL Sonuç</div>
                            <div style="font-size:1.1em;">${res['hodl_val']:.2f}</div>
                            <div style="font-size:0.8em; color:{'#0f0' if alpha>0 else '#f44'}">Alpha: {alpha:+.2f}$</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)# Grafik
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res['dates'], y=res['equity'], name="Bot", line=dict(color=color, width=2)))
                # HODL Referans (Başlangıç ve Bitiş Noktası)
                fig.add_trace(go.Scatter(x=[res['dates'][0], res['dates'][-1]], y=[capital, res['hodl_val']], 
                                         name="HODL", line=dict(color="gray", dash="dot")))
                
                fig.update_layout(height=150, margin=dict(t=0,b=0,l=0,r=0), showlegend=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
        
        progress.progress((i+1)/len(selected_tickers))
    
    progress.empty()
    
    if results:
        total_start = capital * len(results)
        total_end = sum([r['final'] for r in results])
        total_hodl = sum([r['hodl_val'] for r in results])
        
        st.markdown("---")
        st.markdown("### 🏆 PORTFÖY ÖZETİ")
        c1, c2, c3 = st.columns(3)
        c1.metric("Başlangıç", f"${total_start:.0f}")
        c2.metric("Bot Bitiş", f"${total_end:.2f}", f"%{((total_end-total_start)/total_start)*100:.1f}")
        c3.metric("HODL Bitiş", f"${total_hodl:.2f}", delta=f"${total_end - total_hodl:.2f}")
