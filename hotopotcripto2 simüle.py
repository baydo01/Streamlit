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

st.set_page_config(page_title="Smart Flow AI v2", layout="wide")
st.title("🌊 Smart Flow AI: Hatasız & Optimize")
st.markdown("""
Bu versiyonda:
1.  **NameError Hatası:** Giderildi, grafikler çalışıyor.
2.  **Daha Seçici:** Bot artık %25 (0.25) üzerinde güçlü sinyal görmedikçe alım yapmaz, **USD'de bekler.** Bu, düşüş piyasasında sermayeyi korur.
""")

# --- AYARLAR ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    default_tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "AVAX-USD", "DOGE-USD", "LINK-USD"]
    selected_tickers = st.multiselect("Havuzdaki Coinler", default_tickers, default=["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "BNB-USD"])
    
    total_capital = st.number_input("Toplam Fon Sermayesi ($)", value=10000)
    test_days = st.number_input("Test Süresi (Gün)", value=90)
    val_days = st.number_input("Validation Süresi (Gün)", value=45)

# --- CORE MOTORLAR (HMM, TREND, RF) ---
def get_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        
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

def tune_parameters(df, val_start, val_end):
    val_data = df.iloc[val_start:val_end]
    train_ref = df.iloc[:val_start]
    if len(train_ref) < 50: return 3, 5
    
    best_hmm_n = 3; best_roi = -999
    for n in [2, 3]:
        sub_val = val_data.iloc[-20:] 
        cash=1000; coin=0
        for i in range(len(sub_val)):
            row = sub_val.iloc[i]
            sig = get_hmm_signal(train_ref, row[['log_ret', 'range']].values, n)
            p = row['close']
            if sig > 0.1 and cash>0: coin=cash/p; cash=0
            elif sig < -0.1 and coin>0: cash=coin*p; coin=0
        final = cash + (coin * sub_val.iloc[-1]['close'])
        if final > best_roi: best_roi = final; best_hmm_n = n
            
    best_rf_depth = 5; best_acc = 0
    features = ['log_ret', 'range', 'rsi', 'dist_ma']
    for d in [3, 7]:
        clf = RandomForestClassifier(n_estimators=30, max_depth=d, random_state=42)
        clf.fit(train_ref[features], train_ref['target'])
        preds = clf.predict(val_data[features])
        acc = np.mean(preds == val_data['target'])
        if acc > best_acc: best_acc = acc; best_rf_depth = d
    return best_hmm_n, best_rf_depth

# --- SİNYAL ÜRETİMİ (PRE-CALCULATION) ---
def generate_signals(tickers, t_days, v_days):
    signal_matrix = {}
    tuning_results = {}
    common_dates = None
    progress_bar = st.progress(0)
    status = st.empty()
    coin_data = {} 
    
    for i, t in enumerate(tickers):
        status.text(f"Veri İndiriliyor: {t}...")
        df = get_data(t)
        if len(df) < (t_days + v_days + 60): continue
        coin_data[t] = df
        if common_dates is None: common_dates = df.index[-t_days:]
        else: common_dates = common_dates.intersection(df.index[-t_days:])
        progress_bar.progress((i+1) / (len(tickers)*2))

    if common_dates is None or len(common_dates) == 0: return None, None, None

    final_signals = pd.DataFrame(index=common_dates, columns=tickers)
    price_matrix = pd.DataFrame(index=common_dates, columns=tickers)
    
    idx_counter = 0
    for t in tickers:
        if t not in coin_data: continue
        status.text(f"Yapay Zeka Modelleri Eğitiliyor: {t}...")
        df = coin_data[t]
        
        test_start_idx = df.index.get_loc(common_dates[0])
        val_start_idx = test_start_idx - v_days
        best_n, best_depth = tune_parameters(df, val_start_idx, test_start_idx)
        tuning_results[t] = f"{best_n} State | {best_depth} Depth"
        
        errors = {'HMM': 1.0, 'TREND': 1.0, 'RF': 1.0}
        signals = []
        for date in common_dates:
            curr_idx = df.index.get_loc(date)
            train_window = df.iloc[curr_idx-60:curr_idx]
            curr = df.iloc[curr_idx]
            
            hmm_sig = get_hmm_signal(train_window, curr[['log_ret', 'range']].values, best_n)
            trend_sig = get_linear_trend_signal(train_window['close'])
            rf_sig = get_rf_signal(train_window, curr[['log_ret', 'range', 'rsi', 'dist_ma']], best_depth)
            
            inv_total = (1/errors['HMM']) + (1/errors['TREND']) + (1/errors['RF'])
            w_hmm = (1/errors['HMM']) / inv_total
            w_trend = (1/errors['TREND']) / inv_total
            w_rf = (1/errors['RF']) / inv_total
            
            final_sig = (hmm_sig * w_hmm) + (trend_sig * w_trend) + (rf_sig * w_rf)
            signals.append(final_sig)
            
            if curr_idx + 1 < len(df):
                actual_move = np.sign(df.iloc[curr_idx+1]['close'] - curr['close'])
                decay = 0.90
                errors['HMM'] = (errors['HMM']*decay) + (abs(np.sign(hmm_sig)-actual_move)*(1-decay))
                errors['TREND'] = (errors['TREND']*decay) + (abs(np.sign(trend_sig)-actual_move)*(1-decay))
                errors['RF'] = (errors['RF']*decay) + (abs(np.sign(rf_sig)-actual_move)*(1-decay))
                for k in errors: errors[k] = max(errors[k], 0.01)

        final_signals[t] = signals
        price_matrix[t] = df.loc[common_dates]['close']
        idx_counter += 1
        progress_bar.progress(0.5 + (idx_counter/len(tickers)*0.5))

    status.empty()
    progress_bar.empty()
    return final_signals, price_matrix, tuning_results

# --- PORTFÖY SİMÜLASYONU (SMART FLOW) ---
def run_smart_portfolio(signals, prices, initial_capital):
    cash = initial_capital
    holdings = {t: 0 for t in signals.columns}
    equity_curve = []
    dates = signals.index
    allocation_history = []
    
    # !!! KRİTİK AYAR: Eşik Değeri Yükseltildi !!!
    # Eski: 0.2 -> Çok sık alıyordu.
    # Yeni: 0.25 -> Daha emin olunca alıyor.
    BUY_THRESH = 0.25 
    
    for date in dates:
        current_equity = cash
        for t, qty in holdings.items():
            current_equity += qty * prices.loc[date, t]
            
        daily_signals = signals.loc[date]
        buy_candidates = daily_signals[daily_signals > BUY_THRESH].index.tolist()
        
        if len(buy_candidates) > 0:
            target_per_coin = current_equity / len(buy_candidates)
            
            # Satışlar
            for t in holdings:
                if t not in buy_candidates and holdings[t] > 0:
                    revenue = holdings[t] * prices.loc[date, t]
                    cash += revenue * 0.999 
                    holdings[t] = 0
            
            # Alımlar
            for t in buy_candidates:
                current_pos_val = holdings[t] * prices.loc[date, t]
                if current_pos_val < target_per_coin:
                    needed = target_per_coin - current_pos_val
                    if cash >= needed:
                        qty_to_buy = (needed * 0.999) / prices.loc[date, t]
                        holdings[t] += qty_to_buy
                        cash -= needed
                    else:
                        if cash > 0:
                            qty_to_buy = (cash * 0.999) / prices.loc[date, t]
                            holdings[t] += qty_to_buy
                            cash = 0
                elif current_pos_val > target_per_coin * 1.05:
                    excess = current_pos_val - target_per_coin
                    qty_to_sell = excess / prices.loc[date, t]
                    holdings[t] -= qty_to_sell
                    cash += excess * 0.999
        else:
            # HEPSİNİ SAT -> USD (CASH)
            for t in holdings:
                if holdings[t] > 0:
                    cash += holdings[t] * prices.loc[date, t] * 0.999
                    holdings[t] = 0
        
        final_equity = cash
        for t, qty in holdings.items():
            final_equity += qty * prices.loc[date, t]
        
        equity_curve.append(final_equity)
        
        # Alokasyon Kaydı
        if final_equity > 0:
            alloc = {t: (holdings[t]*prices.loc[date, t])/final_equity for t in holdings}
            alloc['CASH'] = cash / final_equity
        else:
            alloc = {t: 0 for t in holdings}
            alloc['CASH'] = 1.0
        allocation_history.append(alloc)
        
    return equity_curve, allocation_history

# --- ARAYÜZ ---
if st.button("🌊 SİSTEMİ BAŞLAT"):
    if not selected_tickers: st.error("Coin seçin.")
    else:
        with st.spinner("Modeller çalışıyor..."):
            sig_df, price_df, tunings = generate_signals(selected_tickers, test_days, val_days)
        
        if sig_df is not None:
            equity, alloc_hist = run_smart_portfolio(sig_df, price_df, total_capital)
            
            final_bal = equity[-1]
            roi = (final_bal - total_capital) / total_capital
            
            # Benchmark (HODL)
            bench_final = 0
            per_coin_inv = total_capital / len(selected_tickers)
            for t in selected_tickers:
                start_p = price_df[t].iloc[0]
                end_p = price_df[t].iloc[-1]
                bench_final += (per_coin_inv / start_p) * end_p
            
            bench_roi = (bench_final - total_capital) / total_capital
            alpha = roi - bench_roi
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Smart Flow Kasa", f"${final_bal:,.0f}", f"%{roi*100:.1f}")
            c2.metric("Statik Sepet (HODL)", f"${bench_final:,.0f}", f"%{bench_roi*100:.1f}")
            c3.metric("ALPHA", f"%{alpha*100:.1f}", delta_color="normal")
            
            # Grafik 1: Performans
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sig_df.index, y=equity, name="Smart Flow", line=dict(color="#00ff00", width=3)))
            
            bench_curve = []
            for date in sig_df.index:
                val = 0
                for t in selected_tickers:
                    start_p = price_df[t].iloc[0]
                    curr_p = price_df.loc[date, t]
                    val += (per_coin_inv / start_p) * curr_p
                bench_curve.append(val)
            fig.add_trace(go.Scatter(x=sig_df.index, y=bench_curve, name="HODL Sepeti", line=dict(color="gray", dash="dot")))
            fig.update_layout(title="Portföy Performansı", template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Grafik 2: Para Akışı (HATA BURADAYDI)
            st.markdown("### 📊 Para Akışı: Sermaye Nereye Gidiyor?")
            st.caption("Gri alan USD (Güvenli Liman), Renkli alanlar Coin yatırımlarıdır.")
            
            # --- DÜZELTME: allocation_history değişkenini doğru aldık ---
            alloc_df = pd.DataFrame(alloc_hist, index=sig_df.index)
            
            fig2 = go.Figure()
            # CASH
            fig2.add_trace(go.Scatter(
                x=alloc_df.index, y=alloc_df['CASH'],
                mode='lines', stackgroup='one', name='NAKİT (USD)',
                line=dict(width=0.5, color='gray')
            ))
            # Coinler
            for t in selected_tickers:
                fig2.add_trace(go.Scatter(
                    x=alloc_df.index, y=alloc_df[t],
                    mode='lines', stackgroup='one', name=t,
                    line=dict(width=0.5)
                ))
            
            fig2.update_layout(
                yaxis=dict(title="Portföy Oranı (0-1)", range=[0, 1]),
                template="plotly_dark", height=450
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            st.write("Modellerin Kullandığı Ayarlar:", tunings)
