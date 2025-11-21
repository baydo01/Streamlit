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

st.set_page_config(page_title="Hedge Fund Lab: Portfolio Simulator", layout="wide")
st.title("🏆 Hedge Fund: Tam Portföy Simülasyonu")
st.markdown("Bu modül, tüm coinleri aynı anda simüle eder, her biri için en iyi stratejiyi bulur ve **Toplam Fon Performansını** gösterir.")

# --- AYARLAR ---
with st.sidebar:
    st.header("⚙️ Fon Ayarları")
    # Portföydeki tüm coinler
    default_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "AVAX-USD", "DOGE-USD", "ADA-USD"]
    selected_tickers = st.multiselect("Portföye Dahil Coinler", default_tickers, default=default_tickers)
    
    start_date = st.date_input("Başlangıç", value=pd.to_datetime("2020-01-01"))
    initial_capital_per_coin = st.number_input("Coin Başına Sermaye ($)", value=1000)
    commission = st.number_input("Komisyon Oranı", value=0.001, format="%.4f")
    
    st.markdown("---")
    st.info("ℹ️ **Yeni Matematik:** Puanlama artık +1/-1 değil, yüzdesel değişime göre yapılıyor (Örn: %10 artış = 0.10 puan).")

# --- 1. YENİ MATEMATİKSEL MOTOR (Yüzdesel Büyüklük) ---
def calculate_custom_score_magnitude(df):
    """
    YENİ MANTIK: Sadece yöne (+/-) değil, hareketin BÜYÜKLÜĞÜNE (%) bakar.
    """
    if len(df) < 366: return pd.Series(0, index=df.index)
    
    # 1. Kısa Vade (5 Günlük % Değişim)
    # Eğer %5 arttıysa skor +0.05 olur.
    s1 = df['close'].pct_change(5).fillna(0)
    
    # 2. Orta Vade (35 Günlük % Değişim)
    s2 = df['close'].pct_change(35).fillna(0)
    
    # 3. Uzun Vade (Tersine Mantık / Mean Reversion - 150 Gün)
    # Eğer %50 düştüyse (-0.50), biz bunu pozitife çevirip +0.50 (AL) puanı yazarız.
    # Eğer %200 arttıysa (2.0), biz bunu negatife çevirip -2.0 (SAT) puanı yazarız.
    s3 = df['close'].pct_change(150).fillna(0) * -1 
    
    # 4. Makro Trend (365 Günlük Eğim)
    # Hareketli ortalamanın günlük % değişimi
    ma = df['close'].rolling(365).mean()
    s4 = ma.pct_change().fillna(0) * 100 # Küçük rakam olduğu için katsayı ile büyütüyoruz
    
    # 5. Volatilite (Risk)
    # Oynaklık azalıyorsa (negatif değişim) -> İyi (+ Puan)
    vol = df['close'].pct_change().rolling(10).std()
    s5 = vol.pct_change().fillna(0) * -1 
    
    # 6. Hacim Gücü
    # Hacim ortalamadan % kaç fazla?
    if 'volume' in df.columns:
        vol_ma = df['volume'].rolling(20).mean()
        s6 = (df['volume'] - vol_ma) / vol_ma
        s6 = s6.fillna(0)
    else: s6 = 0
    
    # 7. Mum Gücü (Gövde büyüklüğü %)
    if 'open' in df.columns:
        s7 = (df['close'] - df['open']) / df['open']
    else: s7 = 0
    
    # SKALALAMA (Scaling)
    # Yüzdeler (0.05 gibi) HMM sinyaline (1.0) göre küçük kalabilir.
    # Bu yüzden teknik puanların etkisini hissettirmek için bir katsayı (Impact Factor) ile çarpıyoruz.
    IMPACT_FACTOR = 5.0 
    total_score = (s1 + s2 + s3 + s4 + s5 + s6 + s7) * IMPACT_FACTOR
    
    return total_score

# --- 2. TEK COIN İÇİN TURNUVA ---
def run_strategy_for_coin(ticker, start_date, cap, comm):
    try:
        df_raw = yf.download(ticker, start=start_date, progress=False)
        # MultiIndex Fix
        if isinstance(df_raw.columns, pd.MultiIndex): df_raw.columns = df_raw.columns.get_level_values(0)
        df_raw.columns = [c.lower() for c in df_raw.columns]
        if 'close' not in df_raw.columns and 'adj close' in df_raw.columns: df_raw['close'] = df_raw['adj close']
        
        if len(df_raw) < 200: return None
    except: return None

    timeframes = {'D': 'D', 'W': 'W', 'M': 'M'}
    weights = [0.50, 0.70, 0.85, 0.90]
    
    best_roi = -9999
    best_equity_curve = []
    best_info = ""
    
    # Veriyi hazırlayalım (HODL için)
    hodl_equity = (cap / df_raw['close'].iloc[0]) * df_raw['close']
    
    for tf_code in timeframes.values():
        # Resample
        if tf_code == 'D': df = df_raw.copy()
        else:
            agg = {'close': 'last', 'high': 'max', 'low': 'min'}
            if 'open' in df_raw.columns: agg['open'] = 'first'
            if 'volume' in df_raw.columns: agg['volume'] = 'sum'
            df = df_raw.resample(tf_code).agg(agg).dropna()
            
        if len(df) < 50: continue
        
        # İndikatörler (YENİ FONKSİYON)
        df['log_ret'] = np.log(df['close']/df['close'].shift(1))
        df['range'] = (df['high'] - df['low'])/df['close']
        df['custom_score'] = calculate_custom_score_magnitude(df)
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
        
        # Rejim Tespiti
        means = df.groupby('state')['log_ret'].mean()
        bull = means.idxmax()
        bear = means.idxmin()
        
        for w_hmm in weights:
            w_score = 1.0 - w_hmm
            cash = cap
            coin = 0
            equity = []
            dates = []
            
            for idx, row in df.iterrows():
                p = row['close']
                hm = 1 if row['state'] == bull else (-1 if row['state'] == bear else 0)
                
                # SKOR MANTIĞI: Artık skor -5.0 ile +5.0 arasında gelebilir (yüzdelerden dolayı)
                # Kararı normalize etmiyoruz, güçlü bir % artış (örneğin skor +2.0) kararı domine etsin istiyoruz.
                sc = row['custom_score'] 
                
                # Karar Formülü
                decision = (w_hmm * hm) + (w_score * sc)
                
                # İşlem (Eşik değerleri biraz genişlettik çünkü skorlar artık değişken)
                if decision > 0.30 and cash > 0: # AL
                    coin = (cash * (1 - comm)) / p
                    cash = 0
                elif decision < -0.30 and coin > 0: # SAT
                    cash = (coin * p) * (1 - comm)
                    coin = 0
                
                val = cash + (coin * p)
                equity.append(val)
                dates.append(idx)
            
            if not equity: continue
            final = equity[-1]
            roi = (final - cap) / cap
            
            if roi > best_roi:
                best_roi = roi
                # Equity eğrisini orijinal tarih aralığına (Günlük) genişletmemiz lazım (Grafik için)
                s_equity = pd.Series(equity, index=dates)
                # Reindex to daily to match HODL curve
                best_equity_curve = s_equity.reindex(df_raw.index, method='ffill').fillna(cap)
                best_info = f"{tf_code} | %{int(w_hmm*100)} AI"

    return best_equity_curve, hodl_equity, best_info, best_roi

# --- 3. ÇALIŞTIRMA BUTONU ---
if st.button("🚀 TÜM PORTFÖYÜ SİMÜLE ET", type="primary"):
    if not selected_tickers:
        st.error("Lütfen en az bir coin seçin.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Toplam Portföy Verileri
        total_bot_equity = None
        total_hodl_equity = None
        
        results_summary = []
        
        for i, t in enumerate(selected_tickers):
            status_text.text(f"Analiz ediliyor: {t}...")
            
            bot_series, hodl_series, info, roi = run_strategy_for_coin(t, start_date, initial_capital_per_coin, commission)
            
            if bot_series is not None:
                # Toplam Portföye Ekle
                if total_bot_equity is None:
                    total_bot_equity = bot_series
                    total_hodl_equity = hodl_series
                else:
                    total_bot_equity = total_bot_equity.add(bot_series, fill_value=0)
                    total_hodl_equity = total_hodl_equity.add(hodl_series, fill_value=0)
                
                results_summary.append({
                    "Coin": t,
                    "Şampiyon Strateji": info,
                    "Bot Kârı": f"%{roi*100:.1f}",
                    "Son Bakiye": f"${bot_series.iloc[-1]:,.2f}"
                })
            
            progress_bar.progress((i + 1) / len(selected_tickers))
            
        status_text.empty()
        
        if total_bot_equity is not None:
            # --- SONUÇLAR ---
            total_invested = initial_capital_per_coin * len(selected_tickers)
            final_bot = total_bot_equity.iloc[-1]
            final_hodl = total_hodl_equity.iloc[-1]
            
            bot_roi = ((final_bot - total_invested) / total_invested) * 100
            hodl_roi = ((final_hodl - total_invested) / total_invested) * 100
            
            st.success("✅ Simülasyon Tamamlandı!")
            
            # 1. ÖZET METRİKLER
            k1, k2, k3 = st.columns(3)
            k1.metric("Bot Toplam Kasa", f"${final_bot:,.0f}", f"%{bot_roi:.1f}")
            k2.metric("HODL Toplam Kasa", f"${final_hodl:,.0f}", f"%{hodl_roi:.1f}")
            k3.metric("Bot Farkı (Alpha)", f"${final_bot - final_hodl:,.0f}", delta_color="normal")
            
            # 2. GRAFİK
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=total_bot_equity.index, y=total_bot_equity, name='BOT Portföyü', line=dict(color='#00ff00', width=2)))
            fig.add_trace(go.Scatter(x=total_hodl_equity.index, y=total_hodl_equity, name='Sadece HODL', line=dict(color='gray', dash='dot')))
            fig.update_layout(title="Toplam Fon Performansı (Kümülatif)", template="plotly_dark", height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # 3. DETAY TABLOSU
            st.markdown("### 📊 Coin Bazlı Detaylar")
            st.dataframe(pd.DataFrame(results_summary))
            
        else:
            st.error("Veri alınamadı.")
else:
    st.info("Coinleri seçin ve 'Simüle Et' butonuna basın. Bot her coin için en iyi stratejiyi bulup toplam sonucu gösterecektir.")
