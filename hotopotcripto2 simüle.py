import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund: Validation + Test", layout="wide")
st.title("🔬 Auto-Tuner: Validation Destekli Otonom Fon")
st.markdown("""
Bu sistem **3 Aşamalı** bir süreç izler:
1.  **TRAIN:** Geçmiş veriyi öğrenir.
2.  **VALIDATION (Hazırlık):** Farklı `n_components` (HMM Durum Sayısı) değerlerini test eder ve **bu coin için en iyi ayarı** bulur.
3.  **TEST (Final):** Bulunan en iyi ayarla son dönemi simüle eder.
""")

# --- AYARLAR ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    tickers = st.multiselect("Coinler", ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "AVAX-USD", "PEPE-USD"], default=["BTC-USD", "ETH-USD"])
    capital = st.number_input("Coin Başı Sermaye ($)", value=1000)
    
    st.divider()
    test_days = st.number_input("Test Süresi (Gün)", value=60, help="Final sınavı (Dokunulmaz veri)")
    val_days = st.number_input("Validation Süresi (Gün)", value=30, help="Ayarların denendiği hazırlık dönemi")

# --- YARDIMCI FONKSİYONLAR ---
def get_data(ticker):
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
    
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['range'] = (df['high'] - df['low']) / df['close']
    df['target'] = np.sign(df['close'].shift(-1) - df['close'])
    df.dropna(inplace=True)
    return df

def train_hmm(train_data, n_states):
    """Belirli bir state sayısı ile model eğitir"""
    X = train_data[['log_ret', 'range']].values
    scaler = StandardScaler()
    try:
        X_s = scaler.fit_transform(X)
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=20, random_state=42)
        model.fit(X_s)
        means = model.means_[:, 0]
        bull = np.argmax(means)
        bear = np.argmin(means)
        return model, scaler, bull, bear
    except:
        return None, None, None, None

def get_signal(model, scaler, bull, bear, features):
    if model is None: return 0
    s_feat = scaler.transform(features.reshape(1, -1))
    probs = model.predict_proba(s_feat)[0]
    return probs[bull] - probs[bear]

# --- SİMÜLASYON ÇEKİRDEĞİ ---
def run_simulation(df, start_idx, end_idx, n_states):
    """Verilen tarih aralığında ve verilen state sayısıyla simülasyon yapar"""
    cash = 1000 # Sanal bakiye
    coin = 0
    equity = []
    
    # Basit bir backtest döngüsü (Hız için rolling window yapmıyoruz validationda, statik bakıyoruz)
    # Validation'da modelin genel başarısına bakacağız.
    
    # Train kısmı: Simülasyon başlangıcından önceki veri
    train_df = df.iloc[:start_idx]
    if len(train_df) < 50: return -9999
    
    model, scaler, bull, bear = train_hmm(train_df, n_states)
    if model is None: return -9999
    
    sim_df = df.iloc[start_idx:end_idx]
    
    for i in range(len(sim_df)):
        row = sim_df.iloc[i]
        feat = row[['log_ret', 'range']].values
        sig = get_signal(model, scaler, bull, bear, feat)
        
        price = row['close']
        if sig > 0.3 and cash > 0:
            coin = cash / price
            cash = 0
        elif sig < -0.3 and coin > 0:
            cash = coin * price
            coin = 0
        
        equity.append(cash + (coin * price))
        
    if not equity: return -9999
    return (equity[-1] - 1000) / 1000 # ROI

def run_full_process(ticker, t_days, v_days, cap):
    df = get_data(ticker)
    if len(df) < (t_days + v_days + 100): return None
    
    # Zaman Çizelgesi: [ ... TRAIN ... ] [ VALIDATION ] [ TEST ]
    test_start_idx = len(df) - t_days
    val_start_idx = test_start_idx - v_days
    
    # --- AŞAMA 1: VALIDATION (EN İYİ AYARI BUL) ---
    best_n = 3
    best_val_roi = -9999
    
    # Denenecek Ayarlar: State Sayısı (Karmaşıklık)
    # 2: Boğa/Ayı
    # 3: Boğa/Ayı/Yatay
    # 4: Boğa/Ayı/Yatay/Panik
    options = [2, 3, 4] 
    
    tuning_logs = []
    
    for n in options:
        # Validation aralığında test et
        roi = run_simulation(df, val_start_idx, test_start_idx, n)
        tuning_logs.append(f"• Ayar {n} State -> ROI: %{roi*100:.1f}")
        if roi > best_val_roi:
            best_val_roi = roi
            best_n = n
            
    # --- AŞAMA 2: TEST (FİNAL KOŞU) ---
    # Artık 'best_n' değerini biliyoruz. Test verisinde bunu kullanacağız.
    # Test aşamasında Rolling Window (Meta-Learning) kullanalım ki en güçlü hali olsun.
    
    start_idx = test_start_idx
    cash = cap
    coin = 0
    equity = []
    dates = []
    
    # Hakem listeleri
    macro_correct = [0]*5
    micro_correct = [0]*5
    
    # Test Loop
    for i in range(start_idx, len(df)-1):
        # Rolling Veri
        df_macro = df.iloc[:i]
        df_micro = df.iloc[i-60:i] # Son 60 gün hafızası
        
        curr = df.iloc[i]
        curr_feat = curr[['log_ret', 'range']].values
        
        # Seçilen EN İYİ STATE SAYISI (best_n) ile modelleri eğit
        # Macro
        macro_m, macro_s, macro_bull, macro_bear = train_hmm(df_macro, best_n)
        macro_sig = get_signal(macro_m, macro_s, macro_bull, macro_bear, curr_feat)
        
        # Micro
        micro_m, micro_s, micro_bull, micro_bear = train_hmm(df_micro, best_n)
        micro_sig = get_signal(micro_m, micro_s, micro_bull, micro_bear, curr_feat)
        
        # Meta-Learning Ağırlık
        m_score = sum(macro_correct)
        mi_score = sum(micro_correct)
        total = m_score + mi_score
        w_macro = m_score / total if total > 0 else 0.5
        w_micro = 1.0 - w_macro
        
        final_sig = (macro_sig * w_macro) + (micro_sig * w_micro)
        
        # İşlem
        p = curr['close']
        if final_sig > 0.3 and cash > 0:
            coin = cash / p
            cash = 0
        elif final_sig < -0.3 and coin > 0:
            cash = coin * p
            coin = 0
            
        equity.append(cash + (coin * p))
        dates.append(curr.name)
        
        # Skorlama
        act = curr['target']
        macro_correct.pop(0); macro_correct.append(1 if np.sign(macro_sig)==act else 0)
        micro_correct.pop(0); micro_correct.append(1 if np.sign(micro_sig)==act else 0)
        
    final_roi = (equity[-1] - cap) / cap
    hodl_roi = (df.iloc[-1]['close'] - df.iloc[start_idx]['close']) / df.iloc[start_idx]['close']
    
    return {
        "ticker": ticker,
        "best_n": best_n,
        "tuning_logs": tuning_logs,
        "roi": final_roi,
        "hodl": hodl_roi,
        "equity": equity,
        "dates": dates,
        "final_bal": equity[-1]
    }

# --- ÇALIŞTIR ---
if st.button("🚀 Auto-Tuner Botlarını Başlat"):
    results = []
    
    # 2'li kolon düzeni
    cols = st.columns(2)
    
    for i, t in enumerate(tickers):
        col = cols[i % 2]
        with col:
            st.write(f"⏳ **{t}** Analiz ediliyor (Validasyon + Test)...")
            res = run_full_process(t, test_days, val_days, capital)
            
            if res:
                # KART GÖRÜNÜMÜ
                bot_roi_pct = res['roi'] * 100
                hodl_roi_pct = res['hodl'] * 100
                alpha = bot_roi_pct - hodl_roi_pct
                
                border_color = "#00ff00" if alpha > 0 else "#ff0000"
                
                st.markdown(f"""
                <div style="border: 1px solid {border_color}; padding: 15px; border-radius: 10px; background-color: rgba(255,255,255,0.05);">
                    <h3>{t}</h3>
                    <small>🎯 Seçilen Ayar: <b>{res['best_n']} States (Durum)</b></small>
                    <div style="font-size:0.8em; color:gray;">{' | '.join(res['tuning_logs'])}</div>
                    <hr>
                    <div style="display:flex; justify-content:space-between;">
                        <div>Bot: <b style="color:{'#0f0' if bot_roi_pct>0 else '#f00'}">%{bot_roi_pct:.1f}</b></div>
                        <div>HODL: <b>%{hodl_roi_pct:.1f}</b></div>
                        <div>Alpha: <b style="color:white">%{alpha:.1f}</b></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Grafik
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res['dates'], y=res['equity'], line=dict(color='#00ff00', width=2), name="Bot"))
                st.plotly_chart(fig, use_container_width=True)
                
                results.append(res)
    
    if results:
        total_bal = sum([r['final_bal'] for r in results])
        total_inv = capital * len(results)
        total_roi = (total_bal - total_inv) / total_inv
        st.success(f"🏆 TOPLAM PORTFÖY SONUCU: ${total_bal:,.0f} ( ROI: %{total_roi*100:.1f} )")
        
