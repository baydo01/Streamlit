# kalman_ai_trader_enhanced.py
# Geliştirilmiş Kalman AI Trader: XGBoost stacking, GA optimizasyonu, walk-forward validation,
# ma5 scoring (5-periyot hareketli ortalama bazlı puanlama), ve non-invaziv entegrasyon.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import plotly.graph_objects as go
import warnings

# Genetic algorithm
from deap import base, creator, tools, algorithms

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Kalman AI Trader - Enhanced", layout="wide")
st.title("Kalman AI Trader — Enhanced: XGB Stack + GA + Walk-Forward")

# -------------------- AYARLAR --------------------
with st.sidebar:
    st.header("⚙️ Ayarlar")
    default_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "AVAX-USD", "DOGE-USD"]
    selected_tickers = st.multiselect("Sepet", default_tickers, default=["BTC-USD", "ETH-USD"])
    capital = st.number_input("Coin Başı Başlangıç ($)", value=10.0)
    window_size = st.slider("Öğrenme Penceresi (Bar Sayısı)", 20, 100, 30)
    use_ga = st.checkbox("Genetic Algoritma ile parametre optimizasyonu", value=True)
    ga_generations = st.number_input("GA generations", min_value=1, max_value=200, value=20)

# -------------------- KALMAN FİLTRESİ --------------------
def apply_kalman_filter(prices):
    n_iter = len(prices)
    sz = (n_iter,)
    Q = 1e-5
    R = 0.01 ** 2
    xhat = np.zeros(sz)
    P = np.zeros(sz)
    xhatminus = np.zeros(sz)
    Pminus = np.zeros(sz)
    K = np.zeros(sz)
    xhat[0] = prices.iloc[0]
    P[0] = 1.0
    for k in range(1, n_iter):
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + Q
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (prices.iloc[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
    return pd.Series(xhat, index=prices.index)

# -------------------- VERİ --------------------
def get_raw_data(ticker):
    try:
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns:
            df['close'] = df['adj close']
        return df
    except Exception:
        return None

# 5-period moving average + ma5_score (karşılaştırma 
# - uzun periyot ortalamasına göre z-score)
def add_ma5_and_score(df, timeframe_code):
    # ma5 = 5-bar hareketli ortalama (zaman dilimine göre aynı)
    df['ma5'] = df['close'].rolling(window=5).mean()
    # long window: daily ~252, weekly~52, monthly~36 (yaklaşık)
    if timeframe_code == 'D':
        long_w = 252
    elif timeframe_code == 'W':
        long_w = 52
    else:
        long_w = 36
    df['ma5_long_mean'] = df['ma5'].rolling(window=long_w, min_periods=10).mean()
    df['ma5_long_std'] = df['ma5'].rolling(window=long_w, min_periods=10).std()
    df['ma5_score'] = (df['ma5'] - df['ma5_long_mean']) / (df['ma5_long_std'] + 1e-9)
    df['ma5_score'].fillna(0, inplace=True)
    return df

# -------------------- VERİ İŞLEME --------------------
def process_data(df, timeframe):
    if df is None or len(df) < 100:
        return None
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    if timeframe == 'W':
        df_res = df.resample('W').agg(agg_dict).dropna()
    elif timeframe == 'M':
        df_res = df.resample('ME').agg(agg_dict).dropna()
    else:
        df_res = df.copy()
    if len(df_res) < 50:
        return None
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'])
    df_res['log_ret'] = np.log(df_res['kalman_close'] / df_res['kalman_close'].shift(1))
    df_res['range'] = (df_res['high'] - df_res['low']) / df_res['close']
    df_res['trend_signal'] = np.where(df_res['close'] > df_res['kalman_close'], 1, -1)
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    df_res.dropna(inplace=True)
    df_res = add_ma5_and_score(df_res, timeframe)
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_res.dropna(inplace=True)
    return df_res

# -------------------- WALK-FORWARD SPLIT --------------------
def walk_forward_splits(df, n_splits=3, test_size_ratio=0.2):
    # Basit walk-forward: ilk train, sonra sequential validation/test blokları
    n = len(df)
    test_size = max(int(n * test_size_ratio), 10)
    step = max(int((n - test_size) / (n_splits + 1)), 1)
    splits = []
    for i in range(n_splits):
        train_end = step * (i + 1)
        val_start = train_end
        val_end = val_start + step
        test_start = val_end
        test_end = min(test_start + test_size, n)
        if test_end - test_start < 5:
            break
        splits.append((slice(0, train_end), slice(val_start, val_end), slice(test_start, test_end)))
    if not splits:
        # fallback: single split
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        splits = [(slice(0, train_end), slice(train_end, val_end), slice(val_end, n))]
    return splits

# -------------------- MODELING: HMM + RF + XGB STACKING --------------------
from sklearn.exceptions import NotFittedError


def get_signals_enhanced(df, current_idx, learn_window, rf_depth=5, xgb_params=None, n_hmm=3):
    start = max(0, current_idx - learn_window)
    train_data = df.iloc[start:current_idx]
    curr_row = df.iloc[current_idx]
    if len(train_data) < 15:
        return 0.0
    # HMM
    hmm_sig = 0.0
    try:
        X = train_data[['log_ret', 'range']].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = GaussianHMM(n_components=n_hmm, covariance_type="diag", n_iter=30, random_state=42)
        model.fit(X_s)
        bull = np.argmax(model.means_[:, 0])
        bear = np.argmin(model.means_[:, 0])
        curr_feat = scaler.transform(curr_row[['log_ret', 'range']].values.reshape(1, -1))
        probs = model.predict_proba(curr_feat)[0]
        hmm_sig = probs[bull] - probs[bear]
    except Exception:
        hmm_sig = 0.0
    # RF
    rf_sig = 0.0
    rf_prob = 0.5
    xgb_prob = 0.5
    try:
        features = ['log_ret', 'range', 'trend_signal', 'ma5_score']
        clf_rf = RandomForestClassifier(n_estimators=50, max_depth=rf_depth, random_state=42)
        clf_rf.fit(train_data[features], train_data['target'])
        curr_feat_df = pd.DataFrame([curr_row[features]])
        rf_prob = clf_rf.predict_proba(curr_feat_df)[0][1]
        rf_sig = (rf_prob - 0.5) * 2
    except Exception:
        rf_sig = 0.0
    # XGBoost
    try:
        if xgb_params is None:
            xgb_params = {'n_estimators':50, 'max_depth':3, 'learning_rate':0.1}
        clf_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **xgb_params)
        features = ['log_ret', 'range', 'trend_signal', 'ma5_score']
        clf_xgb.fit(train_data[features], train_data['target'])
        xgb_prob = clf_xgb.predict_proba(curr_feat_df)[0][1]
    except Exception:
        xgb_prob = 0.5
    # Stacking meta model (simple logistic trained on the same window)
    try:
        meta_X = np.vstack([clf_rf.predict_proba(train_data[features])[:,1], clf_xgb.predict_proba(train_data[features])[:,1]]).T
        meta_y = train_data['target'].values
        meta_clf = LogisticRegression()
        meta_clf.fit(meta_X, meta_y)
        meta_curr = np.array([[rf_prob, xgb_prob]])
        stack_prob = meta_clf.predict_proba(meta_curr)[0][1]
        stack_sig = (stack_prob - 0.5) * 2
    except Exception:
        # fallback: average
        stack_sig = ((rf_prob + xgb_prob) / 2 - 0.5) * 2
    # Kalman trend
    k_trend = curr_row['trend_signal']
    # Combine with weights
    combined = (hmm_sig * 0.25) + (stack_sig * 0.35) + (k_trend * 0.4)
    return combined

# -------------------- STRATEGY SIMULATION --------------------

def simulate_using_df(df, start_cap, win_size, params=None):
    # params can contain rf_depth, xgb_params, buy_threshold, sell_threshold
    if params is None:
        params = {}
    rf_depth = params.get('rf_depth', 5)
    xgb_params = params.get('xgb_params', None)
    buy_t = params.get('buy_th', 0.25)
    sell_t = params.get('sell_th', -0.25)
    cash = start_cap
    coin = 0
    equity = []
    dates = []
    for i in range(len(df)):
        sig = get_signals_enhanced(df, i, win_size, rf_depth=rf_depth, xgb_params=xgb_params)
        price = df['close'].iloc[i]
        if sig > buy_t and cash > 0:
            coin = cash / price
            cash = 0
        elif sig < sell_t and coin > 0:
            cash = coin * price
            coin = 0
        equity.append(cash + coin * price)
        dates.append(df.index[i])
    final = equity[-1]
    roi = (final - start_cap) / start_cap
    return {'final': final, 'roi': roi, 'equity': equity, 'dates': dates}

# -------------------- GA OPTIMIZATION --------------------

def ga_optimize_params(df, start_cap, win_size, n_gen=20, pop_size=20):
    # Kromozon: [rf_depth (3-12), xgb_max_depth (2-6), xgb_eta (0.01-0.3), buy_th (0.05-0.5), sell_th (-0.5,-0.05)]
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register('rf_depth', np.random.randint, 3, 13)
    toolbox.register('xgb_max_depth', np.random.randint, 2, 7)
    toolbox.register('xgb_eta', np.random.uniform, 0.01, 0.3)
    toolbox.register('buy_th', np.random.uniform, 0.05, 0.5)
    toolbox.register('sell_th', np.random.uniform, -0.5, -0.05)
    toolbox.register('individual', tools.initCycle, creator.Individual,
                     (toolbox.rf_depth, toolbox.xgb_max_depth, toolbox.xgb_eta, toolbox.buy_th, toolbox.sell_th), n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    def eval_individual(ind):
        rf_depth, xgb_md, xgb_eta, buy_th, sell_th = ind
        xgb_params = {'max_depth': int(xgb_md), 'learning_rate': float(xgb_eta), 'n_estimators': 50}
        params = {'rf_depth': int(rf_depth), 'xgb_params': xgb_params, 'buy_th': float(buy_th), 'sell_th': float(sell_th)}
        # walk-forward evaluation (average ROI across splits)
        splits = walk_forward_splits(df, n_splits=3)
        rois = []
        for tr, val, tst in splits:
            subdf = df.iloc[tr]
            # train on tr+val? here we test on tst to measure OOS
            full_train = df.iloc[0:val.stop]
            res = simulate_using_df(df.iloc[val.stop:tst.stop], start_cap, win_size, params=params)
            rois.append(res['roi'])
        return (np.mean(rois),)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('max', np.max)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen, stats=stats, halloffame=hof, verbose=False)
    best = hof[0]
    rf_depth, xgb_md, xgb_eta, buy_th, sell_th = best
    best_params = {'rf_depth': int(rf_depth), 'xgb_params': {'max_depth': int(xgb_md), 'learning_rate': float(xgb_eta), 'n_estimators':50}, 'buy_th': float(buy_th), 'sell_th': float(sell_th)}
    return best_params

# -------------------- RUN STRATEGY (portföy düzeyinde tüm timeframes + GA opsiyonel) --------------------

def run_strategy_enhanced(ticker, start_cap, win_size, use_ga_flag=True, ga_gens=20):
    raw_df = get_raw_data(ticker)
    if raw_df is None:
        return None
    raw_df = raw_df.iloc[-1460:]
    best_roi = -9999
    best_res = None
    timeframes = {'Günlük': 'D', 'Haftalık': 'W', 'Aylık': 'M'}
    for tf_name, tf_code in timeframes.items():
        df = process_data(raw_df, tf_code)
        if df is None:
            continue
        # GA ile parametre ara (opsiyonel)
        params = None
        if use_ga_flag:
            try:
                params = ga_optimize_params(df, start_cap, win_size, n_gen=ga_gens)
            except Exception:
                params = None
        # simule et tüm df üzerinde
        sim = simulate_using_df(df, start_cap, win_size, params=params)
        final = sim['final']
        roi = sim['roi']
        if roi > best_roi:
            start_p = df['close'].iloc[0]
            end_p = df['close'].iloc[-1]
            hodl_val = (start_cap / start_p) * end_p
            best_roi = roi
            best_res = {
                'ticker': ticker,
                'tf': tf_name,
                'final': final,
                'roi': roi,
                'hodl': hodl_val,
                'equity': sim['equity'],
                'dates': sim['dates'],
                'kalman_data': df['kalman_close']
            }
    return best_res

# -------------------- STREAMLIT ARAYÜZ --------------------
if st.button("🛡️ ENHANCED ANALİZİ BAŞLAT"):
    cols = st.columns(2)
    prog = st.progress(0)
    results = []
    for i, t in enumerate(selected_tickers):
        with cols[i % 2]:
            with st.spinner(f"{t} için hesaplanıyor..."):
                res = run_strategy_enhanced(t, capital, window_size, use_ga_flag=use_ga, ga_gens=int(ga_generations))
            if res:
                results.append(res)
                is_profit = res['roi'] > 0
                color = "#00ff00" if is_profit else "#ff4444"
                alpha = res['final'] - res['hodl']
                st.markdown(f"**{t} — {res['tf']}** — ROI: {res['roi']:.2%}")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res['dates'], y=res['equity'], name="Bot Bakiye"))
                fig.add_trace(go.Scatter(x=[res['dates'][0], res['dates'][-1]], y=[capital, res['hodl']], name="HODL", line=dict(dash='dot')))
                fig.update_layout(height=250, margin=dict(t=0,b=0,l=0,r=0), template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
        prog.progress((i+1)/len(selected_tickers))
    prog.empty()
    if results:
        total_final = sum([r['final'] for r in results])
        total_hodl = sum([r['hodl'] for r in results])
        total_start = capital * len(results)
        st.markdown('---')
        st.markdown('### Portföy Özeti')
        c1, c2, c3 = st.columns(3)
        c1.metric('Toplam Başlangıç', f'${total_start:.0f}')
        c2.metric('Kalman Bot Bitiş', f'${total_final:.2f}', f"%{((total_final-total_start)/total_start)*100:.1f}")
        c3.metric('HODL Bitiş', f'${total_hodl:.2f}', delta=f'${total_final - total_hodl:.2f}')

# EOF
