import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Hedge Fund Lab: Tournament Simulator", layout="wide")

st.title("🏆 Turnuva Simülatörü: Botun Gözünden Geçmiş")
st.markdown("""
Bu modül, **Botun 'Turnuva Mantığını' çalıştırır.** Sizin yerinize (Günlük/Haftalık/Aylık) ve (Ağırlık Oranlarını) dener, **şampiyonu bulur** ve onun grafiğini çizer.
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    ticker = st.selectbox("Coin Seç", ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "AVAX-USD", "DOGE-USD", "ADA-USD"])
    start_date = st.date_input("Başlangıç Tarihi", value=pd.to_datetime("2020-01-01"))
    initial_capital = st.number_input("Başlangıç Kasası ($)", value=10000)
    commission = st.number_input("Komisyon Oranı", value=0.001, format="%.4f")
    st.info("Bot otomatik olarak en iyi zaman dilimini ve stratejiyi seçecektir.")

# --- 1. GELİŞMİŞ PUANLAMA (BOT İLE AYNI) ---
def calculate_custom_score(df):
    if len(df) < 366: return pd.Series(0, index=df.index)
    daily_steps = np.sign(df['close'].diff()).fillna(0)
    
    s1 = np.where(daily_steps.rolling(5).sum() > 0, 1, -1)
    s2 = np.where(daily_steps.rolling(35).sum() > 0, 1, -1)
    s3 = np.where(daily_steps.rolling(150).sum() < 0, 1, -1) # Tersine Mantık
    ma = df['close'].rolling(365).mean()
    s4 = np.where(ma > ma.shift(1), 1, -1)
    vol = df['close'].pct_change().rolling(10).std()
    s5 = np.where(vol < vol.shift(1), 1, -1)
    s6 = np.where(df['volume'] > df['volume'].rolling(20).mean(), 1, 0) if 'volume' in df.columns else 0
    s7 = np.where(df['close'] > df['open'], 1, -1) if 'open' in df.columns else 0
    
    return s1 + s2 + s3 + s4 + s5 + s6 + s7

# --- 2. TURNUVA MOTORU VE SİMÜLASYON ---
def run_tournament_simulation(ticker, start_date, initial_cap, comm):
    # 1. Veri Çek
    try:
        df_raw = yf.download(ticker, start=start_date, progress=False)
        if isinstance(df_raw.columns, pd.MultiIndex): df_raw.columns = df_raw.columns.get_level_values(0)
        df_raw.columns = [c.lower() for c in df_raw.columns]
        if 'close' not in df_raw.columns and 'adj close' in df_raw.columns: df_raw['close'] = df_raw['adj close']
        
        if len(df_raw) < 300: return None, "Yetersiz Veri"
    except Exception as e: return None, str(e)

    # Turnuva Ayarları
    timeframes = {'GÜNLÜK (D)': 'D', 'HAFTALIK (W)': 'W', 'AYLIK (M)': 'M'}
    weight_scenarios = [0.50, 0.70, 0.85, 0.90, 0.95]
    
    best_roi = -9999
    best_result = None # En iyi sonucu saklayacağız
    
    status_text = st.empty()
    
    # --- TURNUVA DÖNGÜSÜ ---
    for tf_name, tf_code in timeframes.items():
        status_text.text(f"Simüle ediliyor: {tf_name}...")
        
        # Resample
        if tf_code == 'D': df = df_raw.copy()
        else:
            agg = {'close': 'last', 'high': 'max', 'low': 'min'}
            if 'open' in df_raw.columns: agg['open'] = 'first'
            if 'volume' in df_raw.columns: agg['volume'] = 'sum'
            df = df_raw.resample(tf_code).agg(agg).dropna()
        
        if len(df) < 100: continue
        
        # İndikatörler
        df['log_ret'] = np.log(df['close']/df['close'].shift(1))
        df['range'] = (df['high'] - df['low'])/df['close']
        df['custom_score'] = calculate_custom_score(df)
        df.dropna(inplace=True)
        
        # HMM
        X = df[['log_ret', 'range']].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        try:
            model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
            model.fit(X_s)
            states = model.predict(X_s)
            df['state'] = states
        except: continue
        
        # Boğa/Ayı Tespiti
        state_stats = df.groupby('state')['log_ret'].mean()
        bull_state = state_stats.idxmax()
        bear_state = state_stats.idxmin()
        
        # Ağırlık Testleri
        for w_hmm in weight_scenarios:
            w_score = 1.0 - w_hmm
            cash = initial_cap
            coin = 0
            history = []
            buy_signals = []
            sell_signals = []
            
            for idx, row in df.iterrows():
                p = row['close']
                hm = 1 if row['state'] == bull_state else (-1 if row['state'] == bear_state else 0)
                sc = 1 if row['custom_score'] >= 3 else (-1 if row['custom_score'] <= -3 else 0)
                
                decision = (w_hmm * hm) + (w_score * sc)
                
                # İşlem (Bot Mantığı)
                if decision > 0.25 and cash > 0:
                    coin = (cash * (1 - comm)) / p
                    cash = 0
                    buy_signals.append((idx, p))
                elif decision < -0.25 and coin > 0:
                    cash = (coin * p) * (1 - comm)
                    coin = 0
                    sell_signals.append((idx, p))
                
                val = cash + (coin * p)
                history.append(val)
            
            final_val = history[-1]
            roi = (final_val - initial_cap) / initial_cap
            
            # ŞAMPİYON SEÇİMİ
            if roi > best_roi:
                best_roi = roi
                df['Strategy'] = history # Geçmiş bakiyeyi kaydet
                
                # HODL verisini hazırla
                first_p = df['close'].iloc[0]
                hodl_amt = initial_cap / first_p
                df['Hodl'] = hodl_amt * df['close']
                
                best_result = {
                    "df": df,
                    "tf_name": tf_name,
                    "w_hmm": w_hmm,
                    "buys": buy_signals,
                    "sells": sell_signals,
                    "final_val": final_val,
                    "roi": roi
                }
                
    status_text.empty()
    return best_result, None

# --- 3. ARAYÜZ VE ÇALIŞTIRMA ---
if st.button("🏆 TURNUVA SİMÜLASYONUNU BAŞLAT", type="primary"):
    with st.spinner("Yapay Zeka geçmişi tarıyor, en iyi stratejiyi arıyor..."):
        
        res, err = run_tournament_simulation(ticker, start_date, initial_capital, commission)
        
        if err:
            st.error(f"Hata: {err}")
        elif res is None:
            st.warning("Uygun strateji bulunamadı (Veri yetersiz olabilir).")
        else:
            # SONUÇLARI GÖSTER
            df = res['df']
            
            st.success(f"🎯 ŞAMPİYON BULUNDU: **{res['tf_name']}** Grafiği | Ağırlık: **%{int(res['w_hmm']*100)} Yapay Zeka**")
            
            # Metrikler
            c1, c2, c3 = st.columns(3)
            roi_pct = res['roi'] * 100
            hodl_final = df['Hodl'].iloc[-1]
            hodl_roi = ((hodl_final - initial_capital) / initial_capital) * 100
            
            c1.metric("Botun Kazancı", f"${res['final_val']:,.2f}", f"{roi_pct:.1f}%")
            c2.metric("HODL (Al-Unut)", f"${hodl_final:,.2f}", f"{hodl_roi:.1f}%")
            c3.metric("Bot vs HODL Farkı", f"${res['final_val'] - hodl_final:,.2f}", delta_color="normal")
            
            # Grafik Çizimi
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
            
            # Fiyat ve Sinyaller
            fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Fiyat', line=dict(color='gray', width=1)), row=1, col=1)
            
            # AL Sinyalleri (Yeşil Ok)
            if res['buys']:
                bd, bp = zip(*res['buys'])
                fig.add_trace(go.Scatter(x=bd, y=bp, mode='markers', name='AL', marker=dict(color='green', symbol='triangle-up', size=12)), row=1, col=1)
            
            # SAT Sinyalleri (Kırmızı Ok)
            if res['sells']:
                sd, sp = zip(*res['sells'])
                fig.add_trace(go.Scatter(x=sd, y=sp, mode='markers', name='SAT', marker=dict(color='red', symbol='triangle-down', size=12)), row=1, col=1)
            
            # Performans Karşılaştırma
            fig.add_trace(go.Scatter(x=df.index, y=df['Strategy'], name='Bot Bakiyesi', line=dict(color='purple', width=3)), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Hodl'], name='HODL Bakiyesi', line=dict(color='orange', dash='dot')), row=2, col=1)
            
            fig.update_layout(title=f"{ticker} Şampiyon Strateji Performansı", height=700)
            st.plotly_chart(fig, use_container_width=True)
            
            # İşlem Özeti
            st.write("### 📊 İşlem İstatistikleri")
            st.write(f"- Toplam İşlem Sayısı: **{len(res['buys']) + len(res['sells'])}**")
            if len(res['buys']) > 0:
                last_sig = "SAT" if len(res['sells']) >= len(res['buys']) else "AL (Hala Elinde)"
                st.write(f"- Son Durum: **{last_sig}**")

else:
    st.info("👈 Coin seç ve butona bas. Bot senin için binlerce kombinasyonu test edip en iyisini gösterecek.")
