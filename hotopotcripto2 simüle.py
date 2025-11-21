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
st.set_page_config(page_title="Hedge Fund Lab: Strategy Simulator", layout="wide")

st.title("🧪 Hedge Fund Lab: Strateji Simülatörü")
st.markdown("""
Bu modül, **HMM + Puan Bazlı Stratejiyi** geçmiş veriler üzerinde test eder. 
Gerçek işlem yapmaz, sadece **'Eğer çalıştırsaydık ne olurdu?'** sorusunu yanıtlar.
""")

# --- SIDEBAR AYARLARI ---
with st.sidebar:
    st.header("⚙️ Simülasyon Ayarları")
    ticker = st.selectbox("Coin Seç", ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "AVAX-USD", "DOGE-USD", "ADA-USD"])
    start_date = st.date_input("Başlangıç Tarihi", value=pd.to_datetime("2020-01-01"))
    initial_capital = st.number_input("Başlangıç Kasası ($)", value=10000)
    commission = st.number_input("Komisyon Oranı (Her işlem)", value=0.001, format="%.4f")
    
    st.markdown("---")
    st.header("🧠 Model Parametreleri")
    timeframe = st.selectbox("Zaman Dilimi", ["GÜNLÜK (1D)", "HAFTALIK (1W)", "AYLIK (1Mo)"])
    hmm_weight = st.slider("Yapay Zeka (HMM) Ağırlığı", 0.0, 1.0, 0.85, 0.05)
    score_weight = 1.0 - hmm_weight
    st.write(f"Teknik Puan Ağırlığı: **{score_weight:.2f}**")

# --- 1. GELİŞMİŞ PUANLAMA MOTORU (BOT İLE AYNI) ---
def calculate_custom_score(df):
    if len(df) < 366: return pd.Series(0, index=df.index)
    
    # Adım Sayma Mantığı (Bot ile birebir aynı)
    daily_steps = np.sign(df['close'].diff()).fillna(0)
    
    # 1. Kısa Vade (5 gün)
    s1 = np.where(daily_steps.rolling(5).sum() > 0, 1, -1)
    # 2. Orta Vade (35 gün)
    s2 = np.where(daily_steps.rolling(35).sum() > 0, 1, -1)
    # 3. Uzun Vade (Tersine / Mean Reversion 150 gün)
    s3 = np.where(daily_steps.rolling(150).sum() < 0, 1, -1)
    # 4. Makro Trend (Eğim)
    ma = df['close'].rolling(365).mean()
    s4 = np.where(ma > ma.shift(1), 1, -1)
    # 5. Volatilite
    vol = df['close'].pct_change().rolling(10).std()
    s5 = np.where(vol < vol.shift(1), 1, -1)
    # 6. Hacim
    s6 = np.where(df['volume'] > df['volume'].rolling(20).mean(), 1, 0) if 'volume' in df.columns else 0
    # 7. Mum
    s7 = np.where(df['close'] > df['open'], 1, -1) if 'open' in df.columns else 0
    
    return s1 + s2 + s3 + s4 + s5 + s6 + s7

# --- 2. VERİ HAZIRLIĞI ---
@st.cache_data(ttl=3600)
def get_simulation_data(ticker, start, tf_code):
    try:
        df = yf.download(ticker, start=start, progress=False)
        
        # MultiIndex Düzeltme
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        
        # Resample (Zaman Dilimi)
        mapping = {"GÜNLÜK (1D)": "D", "HAFTALIK (1W)": "W", "AYLIK (1Mo)": "M"}
        code = mapping[tf_code]
        
        if code != 'D':
            agg = {'close': 'last', 'high': 'max', 'low': 'min'}
            if 'open' in df.columns: agg['open'] = 'first'
            if 'volume' in df.columns: agg['volume'] = 'sum'
            df = df.resample(code).agg(agg).dropna()
            
        return df
    except Exception as e:
        st.error(f"Veri hatası: {e}")
        return pd.DataFrame()

# --- 3. SİMÜLASYON BUTONU ---
if st.button("🚀 SİMÜLASYONU BAŞLAT", type="primary"):
    with st.spinner("Veriler işleniyor ve zaman makinesi çalıştırılıyor..."):
        
        # 1. Veriyi Çek
        df = get_simulation_data(ticker, start_date, timeframe)
        
        if len(df) < 200:
            st.error("Yeterli veri yok. Tarihi geriye çekin veya başka coin seçin.")
            st.stop()
            
        # 2. İndikatörleri Hesapla
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['range'] = (df['high'] - df['low']) / df['close']
        df['custom_score'] = calculate_custom_score(df)
        df.dropna(inplace=True)
        
        # 3. HMM Modelini Eğit
        X = df[['log_ret', 'range']].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
        model.fit(X_s)
        states = model.predict(X_s)
        df['state'] = states
        
        # Durumları Etiketle
        state_means = df.groupby('state')['log_ret'].mean()
        bull_state = state_means.idxmax()
        bear_state = state_means.idxmin()
        
        # 4. Backtest Döngüsü
        cash = initial_capital
        coin = 0
        portfolio_history = []
        buy_signals = []
        sell_signals = []
        trade_log = []
        
        for idx, row in df.iterrows():
            price = row['close']
            
            # Sinyal Hesaplama
            hmm_sig = 1 if row['state'] == bull_state else (-1 if row['state'] == bear_state else 0)
            score_sig = 1 if row['custom_score'] >= 3 else (-1 if row['custom_score'] <= -3 else 0)
            
            decision = (hmm_weight * hmm_sig) + (score_weight * score_sig)
            
            action = "HOLD"
            
            # ALIM
            if decision > 0.25 and cash > 0:
                coin = (cash * (1 - commission)) / price
                cash = 0
                buy_signals.append((idx, price))
                action = "BUY"
                trade_log.append({"Tarih": idx, "İşlem": "AL", "Fiyat": price, "Bakiye": coin*price})
                
            # SATIM
            elif decision < -0.25 and coin > 0:
                cash = (coin * price) * (1 - commission)
                coin = 0
                sell_signals.append((idx, price))
                action = "SELL"
                trade_log.append({"Tarih": idx, "İşlem": "SAT", "Fiyat": price, "Bakiye": cash})
            
            # Toplam Değer
            current_val = cash + (coin * price)
            portfolio_history.append(current_val)
            
        df['Strategy'] = portfolio_history
        
        # 5. HODL (Al ve Tut) Karşılaştırması
        first_price = df['close'].iloc[0]
        coin_amt_hodl = initial_capital / first_price
        df['BuyHold'] = coin_amt_hodl * df['close']
        
        # --- SONUÇLARI GÖSTER ---
        
        # Metrikler
        final_val = df['Strategy'].iloc[-1]
        hodl_val = df['BuyHold'].iloc[-1]
        roi_strat = ((final_val - initial_capital) / initial_capital) * 100
        roi_hodl = ((hodl_val - initial_capital) / initial_capital) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Botun Sonuç Kasası", f"${final_val:,.2f}", f"{roi_strat:.2f}%")
        c2.metric("HODL (Al-Unut) Kasası", f"${hodl_val:,.2f}", f"{roi_hodl:.2f}%")
        c3.metric("Bot vs HODL Farkı", f"${final_val - hodl_val:,.2f}", delta_color="normal")
        
        # Grafik (Plotly)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
        
        # Fiyat ve Al/Sat Noktaları
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Fiyat', line=dict(color='gray', width=1)), row=1, col=1)
        
        # Al Sinyalleri
        if buy_signals:
            b_dates, b_prices = zip(*buy_signals)
            fig.add_trace(go.Scatter(x=b_dates, y=b_prices, mode='markers', name='AL', marker=dict(color='green', symbol='triangle-up', size=12)), row=1, col=1)
            
        # Sat Sinyalleri
        if sell_signals:
            s_dates, s_prices = zip(*sell_signals)
            fig.add_trace(go.Scatter(x=s_dates, y=s_prices, mode='markers', name='SAT', marker=dict(color='red', symbol='triangle-down', size=12)), row=1, col=1)
            
        # Strateji vs HODL Performansı
        fig.add_trace(go.Scatter(x=df.index, y=df['Strategy'], name='Bot Bakiyesi', line=dict(color='purple', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BuyHold'], name='HODL Bakiyesi', line=dict(color='orange', dash='dot')), row=2, col=1)
        
        fig.update_layout(title=f"{ticker} - Bot vs Piyasa Performansı", height=700)
        st.plotly_chart(fig, use_container_width=True)
        
        # İşlem Geçmişi
        with st.expander("📜 Detaylı İşlem Geçmişini Gör"):
            st.dataframe(pd.DataFrame(trade_log).style.format({"Fiyat": "${:.2f}", "Bakiye": "${:.2f}"}))

else:
    st.info("👈 Soldaki parametreleri ayarla ve 'Simülasyonu Başlat' butonuna bas.")