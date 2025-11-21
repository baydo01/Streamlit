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

st.set_page_config(page_title="Ultimate AI: Full Portfolio", layout="wide")
st.title("💎 Ultimate AI: Tam Portföy Yönetimi")
st.markdown("""
Bu sistem, seçilen **TÜM COINLER** için ayrı ayrı:
1.  **Validasyon:** En iyi State ve Derinlik ayarını bulur.
2.  **Ensemble:** HMM + Trend + RF modellerini çarpıştırır.
3.  **Portföy Analizi:** Tüm sonuçları toplar ve Genel Fon Performansını çıkarır.
""")

# --- AYARLAR ---
with st.sidebar:
    st.header("⚙️ Fon Ayarları")
    # GENİŞLETİLMİŞ LİSTE (Major Coinler)
    default_tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD", "TRX-USD", "LINK-USD"]
    selected_tickers = st.multiselect("Portföydeki Coinler", default_tickers, default=default_tickers)
    
    capital = st.number_input("Coin Başı Sermaye ($)", value=1000)
    test_days = st.number_input("Test Süresi (Gün)", value=90)
    val_days = st.number_input("Validation Süresi (Gün)", value=45)

# --- CORE FONKSİYONLAR ---
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
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int) 
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df
    except: return pd.DataFrame()

# --- SİNYAL MOTORLARI ---
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

# --- VALIDATION ---
def tune_parameters(df, val_start, val_end):
    val_data = df.iloc[val_start:val_end]
    train_ref = df.iloc[:val_start]
    if len(train_ref) < 50: return 3, 5
    
    # HMM Tune
    best_hmm_n = 3; best_roi = -999
    for n in [2, 3]:
        sub_val = val_data.iloc[-25:] 
        cash=1000; coin=0
        for i in range(len(sub_val)):
            row = sub_val.iloc[i]
            sig = get_hmm_signal(train_ref, row[['log_ret', 'range']].values, n)
            p = row['close']
            if sig > 0.1 and cash>0: coin=cash/p; cash=0
            elif sig < -0.1 and coin>0: cash=coin*p; coin=0
        final = cash + (coin * sub_val.iloc[-1]['close'])
        if final > best_roi: best_roi = final; best_hmm_n = n
            
    # RF Tune
    best_rf_depth = 5; best_acc = 0
    features = ['log_ret', 'range', 'rsi', 'dist_ma']
    for d in [3, 7]:
        clf = RandomForestClassifier(n_estimators=30, max_depth=d, random_state=42)
        clf.fit(train_ref[features], train_ref['target'])
        preds = clf.predict(val_data[features])
        acc = np.mean(preds == val_data['target'])
        if acc > best_acc: best_acc = acc; best_rf_depth = d
            
    return best_hmm_n, best_rf_depth

# --- SİMÜLASYON MAIN ---
def run_simulation(ticker, t_days, v_days, cap):
    df = get_data(ticker)
    if len(df) < (t_days + v_days + 60): return None
    
    test_start = len(df) - t_days
    val_start = test_start - v_days
    
    # 1. Validation
    best_n, best_depth = tune_parameters(df, val_start, test_start)
    
    # 2. Test
    cash = cap
    coin = 0
    equity = []
    dates = []
    
    errors = {'HMM': 1.0, 'TREND': 1.0, 'RF': 1.0}
    weights_log = {'HMM':[], 'TREND':[], 'RF':[]}
    
    for i in range(test_start, len(df)-1):
        train_window = df.iloc[i-60:i]
        curr = df.iloc[i]
        
        # Sinyaller
        hmm_sig = get_hmm_signal(train_window, curr[['log_ret', 'range']].values, best_n)
        trend_sig = get_linear_trend_signal(train_window['close'])
        rf_sig = get_rf_signal(train_window, curr[['log_ret', 'range', 'rsi', 'dist_ma']], best_depth)
        
        # Ağırlıklar
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
        
        final_sig = (hmm_sig * w_hmm) + (trend_sig * w_trend) + (rf_sig * w_rf)
        
        # İşlem
        p = curr['close']
        if final_sig > 0.2 and cash > 0:
            coin = cash / p
            cash = 0
        elif final_sig < -0.2 and coin > 0:
            cash = coin * p
            coin = 0
            
        equity.append(cash + (coin * p))
        dates.append(curr.name)
        
        # Hata Güncelleme
        actual_move = np.sign(df['close'].iloc[i+1] - p)
        decay = 0.90 
        errors['HMM'] = (errors['HMM']*decay) + (abs(np.sign(hmm_sig)-actual_move)*(1-decay))
        errors['TREND'] = (errors['TREND']*decay) + (abs(np.sign(trend_sig)-actual_move)*(1-decay))
        errors['RF'] = (errors['RF']*decay) + (abs(np.sign(rf_sig)-actual_move)*(1-decay))

    final_val = equity[-1]
    roi = (final_val - cap) / cap
    hodl_roi = (df.iloc[-1]['close'] - df.iloc[test_start]['close']) / df.iloc[test_start]['close']
    
    return {
        "ticker": ticker,
        "best_n": best_n,
        "best_depth": best_depth,
        "roi": roi,
        "hodl": hodl_roi,
        "equity": equity,
        "dates": dates,
        "final_bal": final_val
    }

# --- ARAYÜZ ---
if st.button("🚀 TÜM PORTFÖYÜ BAŞLAT"):
    results = []
    
    # İlerleme Çubuğu
    prog_bar = st.progress(0)
    status_txt = st.empty()
    
    # Grid Layout
    cols = st.columns(2)
    
    for i, t in enumerate(selected_tickers):
        status_txt.text(f"Analiz ediliyor: {t}...")
        prog_bar.progress((i) / len(selected_tickers))
        
        col = cols[i % 2]
        with col:
            res = run_simulation(t, test_days, val_days, capital)
            
            if res:
                results.append(res)
                
                # Kart Tasarımı
                bot_pct = res['roi'] * 100
                hodl_pct = res['hodl'] * 100
                alpha = bot_pct - hodl_pct
                color = "#00ff00" if alpha > 0 else "#ff4444"
                bg_color = "rgba(0, 255, 0, 0.05)" if alpha > 0 else "rgba(255, 0, 0, 0.05)"
                
                st.markdown(f"""
                <div style="border-left: 5px solid {color}; background-color: {bg_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <h4 style="margin:0;">{t}</h4>
                    <small>🎯 {res['best_n']} State | {res['best_depth']} Depth</small>
                    <div style="display:flex; justify-content:space-between; margin-top:5px;">
                        <div>Bot: <b style="color:{color}">%{bot_pct:.1f}</b></div>
                        <div>HODL: <b>%{hodl_pct:.1f}</b></div>
                    </div>
                    <div>Alpha: <b style="color:white">%{alpha:.1f}</b></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Mini Grafik
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res['dates'], y=res['equity'], line=dict(color=color, width=2)))
                fig.update_layout(height=100, margin=dict(t=0,b=0,l=0,r=0), showlegend=False)
                fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
    prog_bar.empty()
    status_txt.empty()

    if results:
        # --- PORTFÖY ÖZETİ ---
        total_invested = capital * len(results)
        total_balance = sum([r['final_bal'] for r in results])
        portfolio_roi = (total_balance - total_invested) / total_invested
        
        # HODL Toplamı
        total_hodl_balance = 0
        for r in results:
             # Eğer HODL yapsaydık ne olurdu?
             # Başlangıç parası * (1 + hodl_roi)
             total_hodl_balance += capital * (1 + r['hodl'])
        
        portfolio_hodl_roi = (total_hodl_balance - total_invested) / total_invested
        portfolio_alpha = portfolio_roi - portfolio_hodl_roi
        
        st.markdown("---")
        st.markdown("### 🏆 GENEL PORTFÖY SONUCU")
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Toplam Kasa", f"${total_balance:,.0f}", f"%{portfolio_roi*100:.1f}")
        k2.metric("Piyasa (HODL) Değeri", f"${total_hodl_balance:,.0f}", f"%{portfolio_hodl_roi*100:.1f}")
        k3.metric("FON ALPHA", f"%{portfolio_alpha*100:.1f}", delta_color="normal")
        
        # Son Yorum
        if portfolio_alpha > 0:
            st.success("✅ Tebrikler! Yapay Zeka, piyasa düşüşüne rağmen (veya yükselişten daha fazla) değer üreterek 'Alpha' yarattı.")
        else:
            st.warning("⚠️ Bot piyasanın gerisinde kaldı. Validasyon süresini artırmayı veya daha agresif modeller eklemeyi düşünebilirsiniz.")

