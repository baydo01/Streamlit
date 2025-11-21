import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Ultimate AI: Val + Ensemble", layout="wide")
st.title("💎 Ultimate AI: Validation + Ensemble + Otonom Ağırlık")
st.markdown("""
Bu sistem **Tam Profesyonel** bir süreç izler:
1. **Validation (Hazırlık):** Geçmiş veride HMM (State Sayısı) ve RF (Derinlik) için en iyi ayarları bulur.
2. **Ensemble (Meclis):** En iyi ayarlarla HMM, Trend (Linear) ve RF modellerini çalıştırır.
3. **Dinamik Yönetim:** Hangi model o an başarılıysa, yetkiyi ona verir.
""")

# --- AYARLAR ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    tickers = st.multiselect("Coinler", ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "XRP-USD"], default=["BTC-USD"])
    capital = st.number_input("Sermaye ($)", value=1000)
    test_days = st.number_input("Test Süresi (Gün)", value=90)
    val_days = st.number_input("Validation Süresi (Gün)", value=45, help="Ayarların denendiği hazırlık süresi")

def get_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        
        # Feature Engineering
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['range'] = (df['high'] - df['low']) / df['close']
        df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).apply(lambda x: x[x>0].mean()/abs(x[x<0].mean()) if len(x[x<0])>0 else 0)))
        df['ma_50'] = df['close'].rolling(50).mean()
        df['dist_ma'] = (df['close'] - df['ma_50']) / df['ma_50']
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int) # 1=Artış
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df
    except: return pd.DataFrame()

# --- MODELLER ---

def get_hmm_signal(train_df, current_feat, n_states):
    try:
        X = train_df[['log_ret', 'range']].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=50, random_state=42)
        model.fit(X_s)
        means = model.means_[:, 0]
        bull = np.argmax(means)
        bear = np.argmin(means)
        
        curr_s = scaler.transform(current_feat.reshape(1, -1))
        probs = model.predict_proba(curr_s)[0]
        return probs[bull] - probs[bear]
    except: return 0

def get_linear_trend_signal(history_prices):
    """ARIMA yerine Hızlı Lineer Regresyon"""
    try:
        lookback = 30
        if len(history_prices) < lookback: return 0
        prices = history_prices.iloc[-lookback:].values.reshape(-1, 1)
        X = np.arange(len(prices)).reshape(-1, 1)
        reg = LinearRegression().fit(X, prices)
        
        pred = reg.predict(np.array([[lookback]]))[0][0]
        curr = prices[-1][0]
        
        if pred > curr * 1.002: return 1
        elif pred < curr * 0.998: return -1
        else: return 0
    except: return 0

def get_rf_signal(train_df, current_feat_row, max_depth):
    try:
        features = ['log_ret', 'range', 'rsi', 'dist_ma']
        X = train_df[features]
        y = train_df['target']
        
        clf = RandomForestClassifier(n_estimators=50, max_depth=max_depth, random_state=42)
        clf.fit(X, y)
        
        curr_x = pd.DataFrame([current_feat_row], columns=features)
        prob = clf.predict_proba(curr_x)[0][1]
        return (prob - 0.5) * 2
    except: return 0

# --- VALIDATION LOOP (AYAR BULUCU) ---
def tune_parameters(df, val_start, val_end):
    """HMM ve RF için en iyi parametreleri bulur"""
    val_data = df.iloc[val_start:val_end]
    train_ref = df.iloc[:val_start] # Validasyon öncesi veri
    
    if len(train_ref) < 50: return 3, 5 # Yetersiz veride varsayılan
    
    # 1. HMM Tuning
    best_hmm_n = 3
    best_roi = -999
    for n in [2, 3]:
        # Hızlıca simüle et
        roi = 0
        # Basitlik için sadece son güne bakmıyoruz, küçük bir döngü kuruyoruz validation içinde
        # Ama hız için sadece son 20 günü simüle edelim
        sub_val = val_data.iloc[-20:] 
        cash=1000; coin=0
        for i in range(len(sub_val)):
            row = sub_val.iloc[i]
            # Modeli eğit (Validasyon train datasında)
            sig = get_hmm_signal(train_ref, row[['log_ret', 'range']].values, n)
            p = row['close']
            if sig > 0.1 and cash>0: coin=cash/p; cash=0
            elif sig < -0.1 and coin>0: cash=coin*p; coin=0
        
        final = cash + (coin * sub_val.iloc[-1]['close'])
        if final > best_roi:
            best_roi = final
            best_hmm_n = n
            
    # 2. Random Forest Tuning
    best_rf_depth = 5
    best_acc = 0
    for d in [3, 7]: # Sığ ağaç vs Derin ağaç
        # RF Validasyonu (Doğruluk oranı üzerinden)
        # Train
        features = ['log_ret', 'range', 'rsi', 'dist_ma']
        clf = RandomForestClassifier(n_estimators=30, max_depth=d, random_state=42)
        clf.fit(train_ref[features], train_ref['target'])
        
        # Test (Validasyon setinde)
        preds = clf.predict(val_data[features])
        acc = np.mean(preds == val_data['target'])
        
        if acc > best_acc:
            best_acc = acc
            best_rf_depth = d
            
    return best_hmm_n, best_rf_depth

# --- SİMÜLASYON MAIN ---
def run_simulation(ticker, t_days, v_days, cap):
    df = get_data(ticker)
    if len(df) < (t_days + v_days + 60): return None
    
    test_start = len(df) - t_days
    val_start = test_start - v_days
    
    # 1. AŞAMA: VALIDATION (EN İYİ AYARLARI SEÇ)
    best_n, best_depth = tune_parameters(df, val_start, test_start)
    
    # 2. AŞAMA: TEST (ENSEMBLE RUN)
    cash = cap
    coin = 0
    equity = []
    dates = []
    
    # Hata Skorları (Başlangıçta eşit)
    errors = {'HMM': 1.0, 'TREND': 1.0, 'RF': 1.0}
    weights_log = {'HMM':[], 'TREND':[], 'RF':[]}
    
    # Test Loop
    for i in range(test_start, len(df)-1):
        # Rolling Window (Son 60 gün hafızası)
        train_window = df.iloc[i-60:i]
        curr = df.iloc[i]
        
        # A. Sinyalleri Al (Seçilen en iyi ayarlarla)
        hmm_sig = get_hmm_signal(train_window, curr[['log_ret', 'range']].values, best_n)
        trend_sig = get_linear_trend_signal(train_window['close'])
        rf_sig = get_rf_signal(train_window, curr[['log_ret', 'range', 'rsi', 'dist_ma']], best_depth)
        
        # B. Dinamik Ağırlık Hesapla (Minimum Hata Prensibi)
        inv_hmm = 1 / max(errors['HMM'], 0.001)
        inv_trend = 1 / max(errors['TREND'], 0.001)
        inv_rf = 1 / max(errors['RF'], 0.001)
        
        total_inv = inv_hmm + inv_trend + inv_rf
        w_hmm = inv_hmm / total_inv
        w_trend = inv_trend / total_inv
        w_rf = inv_rf / total_inv
        
        weights_log['HMM'].append(w_hmm)
        weights_log['TREND'].append(w_trend)
        weights_log['RF'].append(w_rf)
        
        # C. Birleşik Karar
        final_sig = (hmm_sig * w_hmm) + (trend_sig * w_trend) + (rf_sig * w_rf)
        
        # D. İşlem
        p = curr['close']
        # Eşik (Agresiflik): 0.2
        if final_sig > 0.2 and cash > 0:
            coin = cash / p
            cash = 0
        elif final_sig < -0.2 and coin > 0:
            cash = coin * p
            coin = 0
            
        equity.append(cash + (coin * p))
        dates.append(curr.name)
        
        # E. Hata Güncelleme (Learning)
        # Yarın ne oldu?
        actual_move = np.sign(df['close'].iloc[i+1] - p)
        
        # Decay (Eski hataları unut)
        decay = 0.90 
        errors['HMM'] = (errors['HMM']*decay) + (abs(np.sign(hmm_sig)-actual_move)*(1-decay))
        errors['TREND'] = (errors['TREND']*decay) + (abs(np.sign(trend_sig)-actual_move)*(1-decay))
        errors['RF'] = (errors['RF']*decay) + (abs(np.sign(rf_sig)-actual_move)*(1-decay))

    final_roi = (equity[-1] - cap) / cap
    hodl_roi = (df.iloc[-1]['close'] - df.iloc[test_start]['close']) / df.iloc[test_start]['close']
    
    return {
        "ticker": ticker,
        "best_n": best_n,
        "best_depth": best_depth,
        "roi": final_roi,
        "hodl": hodl_roi,
        "equity": equity,
        "dates": dates,
        "weights": weights_log,
        "final_bal": equity[-1]
    }

# --- ARAYÜZ ---
if st.button("💎 SİSTEMİ ÇALIŞTIR"):
    if not tickers: st.error("Coin seçin.")
    else:
        results = []
        cols = st.columns(2)
        
        for i, t in enumerate(tickers):
            with cols[i%2]:
                with st.spinner(f"{t} Validasyon ve Test yapılıyor..."):
                    res = run_simulation(t, test_days, val_days, capital)
                
                if res:
                    bot_p = res['roi']*100
                    hodl_p = res['hodl']*100
                    alpha = bot_p - hodl_p
                    color = "#00ff00" if alpha > 0 else "#ff4444"
                    
                    st.markdown(f"""
                    <div style="border: 1px solid {color}; padding: 10px; border-radius: 10px; margin-bottom:10px;">
                        <h3>{t}</h3>
                        <small>⚙️ Ayarlar: <b>{res['best_n']}</b> State HMM | <b>{res['best_depth']}</b> Derinlik RF</small>
                        <div style="display:flex; justify-content:space-between; margin-top:5px;">
                            <span>Bot: <b style="color:{color}">%{bot_p:.1f}</b></span>
                            <span>HODL: <b>%{hodl_p:.1f}</b></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Grafik (Equity)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=res['dates'], y=res['equity'], name="Bot", line=dict(color=color)))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Grafik (Weights)
                    fig_w = go.Figure()
                    fig_w.add_trace(go.Scatter(x=res['dates'], y=res['weights']['HMM'], name="HMM", stackgroup='one'))
                    fig_w.add_trace(go.Scatter(x=res['dates'], y=res['weights']['TREND'], name="Trend", stackgroup='one'))
                    fig_w.add_trace(go.Scatter(x=res['dates'], y=res['weights']['RF'], name="RF", stackgroup='one'))
                    fig_w.update_layout(height=200, margin=dict(t=0,b=0,l=0,r=0), title="Yapay Zeka Karar Dağılımı")
                    st.plotly_chart(fig_w, use_container_width=True)
                    
                    results.append(res)
        
        if results:
            total = sum([r['final_bal'] for r in results])
            roi = (total - (capital*len(results))) / (capital*len(results))
            st.success(f"🏆 PORTFÖY SONUCU: ${total:,.0f} ( ROI: %{roi*100:.1f} )")
