import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Grand Council AI", layout="wide")
st.title("🏛️ Grand Council: HMM + ARIMA + Random Forest")
st.markdown("""
Bu sistem 3 farklı yapay zeka modelini çalıştırır ve **Dinamik Ağırlıklandırma** ile en başarılı olanın sözünü dinler.
1. **HMM:** Piyasa rejimini (Risk) koklar.
2. **ARIMA:** Matematiksel trendi (Yönü) hesaplar.
3. **Random Forest:** Teknik indikatörler arasındaki karmaşık ilişkileri çözer.
""")

# --- AYARLAR ---
with st.sidebar:
    st.header("⚙️ Konsey Ayarları")
    ticker = st.selectbox("Coin", ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "XRP-USD"])
    capital = st.number_input("Sermaye ($)", value=1000)
    history_days = st.slider("Geriye Dönük Hafıza (Gün)", 60, 365, 180)

# --- VERİ HAZIRLIĞI VE FEATURE ENGINEERING ---
def get_data(ticker):
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
    
    # Feature Engineering (Random Forest için)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['range'] = (df['high'] - df['low']) / df['close']
    df['rsi'] = 100 - (100 / (1 + df['close'].pct_change().rolling(14).apply(lambda x: x[x>0].mean()/abs(x[x<0].mean()) if len(x[x<0])>0 else 0)))
    df['ma_50'] = df['close'].rolling(50).mean()
    df['dist_ma'] = (df['close'] - df['ma_50']) / df['ma_50']
    
    # Target (Yarın artacak mı? 1=Evet, 0=Hayır)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)
    return df

# --- MODEL 1: HMM (Rejim Uzmanı) ---
def get_hmm_signal(train_data, current_feat):
    """Piyasa Boğa ise +1, Ayı ise -1"""
    try:
        X = train_data[['log_ret', 'range']].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=50, random_state=42)
        model.fit(X_s)
        
        means = model.means_[:, 0]
        bull = np.argmax(means)
        bear = np.argmin(means)
        
        curr_s = scaler.transform(current_feat.reshape(1, -1))
        probs = model.predict_proba(curr_s)[0]
        
        # Olasılık farkı sinyali
        return probs[bull] - probs[bear]
    except: return 0

# --- MODEL 2: ARIMA (Trend Uzmanı) ---
def get_arima_signal(history_prices):
    """Gelecek fiyat tahmini > Şu anki fiyat ise +1"""
    try:
        # Hız için basit bir (5,1,0) modeli kullanıyoruz
        # Not: Loop içinde Auto-ARIMA çok yavaş olur, sabit order kullandık.
        model = ARIMA(history_prices, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        pred_price = forecast.iloc[0] if isinstance(forecast, pd.Series) else forecast[0]
        
        current_price = history_prices.iloc[-1]
        
        if pred_price > current_price * 1.001: return 1 # %0.1 artış bekliyorsa AL
        elif pred_price < current_price * 0.999: return -1 # SAT
        else: return 0
    except: return 0

# --- MODEL 3: RANDOM FOREST (Teknik İndikatör Uzmanı) ---
def get_rf_signal(train_df, current_feat_row):
    """Teknik verilere bakıp Yön Tahmini (Classification)"""
    try:
        features = ['log_ret', 'range', 'rsi', 'dist_ma']
        X = train_df[features]
        y = train_df['target']
        
        clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        clf.fit(X, y)
        
        # Tahmin (0 veya 1) -> Sinyale çevir (-1 veya 1)
        curr_x = pd.DataFrame([current_feat_row], columns=features)
        prediction = clf.predict(curr_x)[0] # 1 (Artış) veya 0 (Düşüş)
        
        prob = clf.predict_proba(curr_x)[0][1] # Artış olasılığı
        
        # Olasılık üzerinden güç belirle
        return (prob - 0.5) * 2 # 0.8 olasılık -> 0.6 sinyal gücü
    except: return 0

# --- SİMÜLASYON ---
if st.button("🏛️ Meclisi Topla ve Simüle Et"):
    df = get_data(ticker)
    
    if len(df) < history_days + 50:
        st.error("Veri yetersiz.")
    else:
        start_idx = len(df) - history_days
        
        cash = capital
        coin = 0
        equity = []
        dates = []
        
        # Modellerin geçmiş performans skorları (Loss based weights)
        # Başlangıçta eşit güveniyoruz (Hata skorları eşit ve düşük)
        errors = {'HMM': 1.0, 'ARIMA': 1.0, 'RF': 1.0} 
        
        weights_history = {'HMM': [], 'ARIMA': [], 'RF': []}
        
        progress = st.progress(0)
        
        # --- ROLLING WINDOW LOOP ---
        for i in range(start_idx, len(df)-1):
            prog = (i - start_idx) / history_days
            progress.progress(min(prog, 1.0))
            
            # Veri Pencereleri
            # Son 60 gün eğitim için (Modeller hafızalarını taze tutsun)
            train_window = df.iloc[i-60:i]
            current_row = df.iloc[i]
            
            # --- 1. MODELLERİ DİNLE (Sinyal Al) ---
            
            # HMM
            hmm_sig = get_hmm_signal(train_window, current_row[['log_ret', 'range']].values)
            
            # ARIMA (Sadece kapanış fiyat serisini alır)
            arima_sig = get_arima_signal(train_window['close'])
            
            # Random Forest
            rf_sig = get_rf_signal(train_window, current_row[['log_ret', 'range', 'rsi', 'dist_ma']].iloc[0] if isinstance(current_row, pd.DataFrame) else current_row[['log_ret', 'range', 'rsi', 'dist_ma']])
            
            # --- 2. DİNAMİK AĞIRLIKLANDIRMA (MINIMUM LOSS) ---
            # Hata ne kadar küçükse, ağırlık o kadar büyük olur (Inverse Weighting)
            # Ağırlık = 1 / Hata_Skoru
            inv_err_hmm = 1 / errors['HMM']
            inv_err_arima = 1 / errors['ARIMA']
            inv_err_rf = 1 / errors['RF']
            
            total_inv_err = inv_err_hmm + inv_err_arima + inv_err_rf
            
            w_hmm = inv_err_hmm / total_inv_err
            w_arima = inv_err_arima / total_inv_err
            w_rf = inv_err_rf / total_inv_err
            
            # Kayıt (Grafik için)
            weights_history['HMM'].append(w_hmm)
            weights_history['ARIMA'].append(w_arima)
            weights_history['RF'].append(w_rf)
            
            # --- 3. KARAR VE İŞLEM ---
            # Konsensüs Sinyali
            ensemble_signal = (hmm_sig * w_hmm) + (arima_sig * w_arima) + (rf_sig * w_rf)
            
            price = current_row['close']
            if ensemble_signal > 0.2 and cash > 0:
                coin = cash / price
                cash = 0
            elif ensemble_signal < -0.2 and coin > 0:
                cash = coin * price
                coin = 0
                
            equity.append(cash + (coin * price))
            dates.append(df.index[i])
            
            # --- 4. PERFORMANS ÖLÇÜMÜ (LOSS UPDATE) ---
            # Yarın ne oldu?
            actual_move = np.sign(df['close'].iloc[i+1] - price) # +1 veya -1
            
            # Her modelin hatasını hesapla (Decay Factor ile)
            # Decay 0.95: Eski hataları yavaş yavaş unut, yeni hatalara odaklan.
            decay = 0.95
            
            # Hata = |Tahmin - Gerçek| 
            # Tahmin doğruysa (işaretler aynıysa) hata azdır.
            err_h = abs(np.sign(hmm_sig) - actual_move) 
            err_a = abs(np.sign(arima_sig) - actual_move)
            err_r = abs(np.sign(rf_sig) - actual_move)
            
            # Hata skorunu güncelle (Exponential Moving Average of Errors)
            errors['HMM'] = (errors['HMM'] * decay) + (err_h * (1-decay))
            errors['ARIMA'] = (errors['ARIMA'] * decay) + (err_a * (1-decay))
            errors['RF'] = (errors['RF'] * decay) + (err_r * (1-decay))
            
            # Sıfıra bölünme hatasını engellemek için taban koy
            for k in errors: errors[k] = max(errors[k], 0.01)

        progress.empty()
        
        # --- SONUÇLAR ---
        final_roi = (equity[-1] - capital) / capital
        hodl_roi = (df['close'].iloc[-1] - df['close'].iloc[start_idx]) / df['close'].iloc[start_idx]
        
        # Metrikler
        c1, c2, c3 = st.columns(3)
        c1.metric("Ensemble (Meclis) Kârı", f"%{final_roi*100:.1f}", f"${equity[-1]:.0f}")
        c2.metric("HODL", f"%{hodl_roi*100:.1f}")
        c3.metric("Alpha", f"%{(final_roi - hodl_roi)*100:.1f}")
        
        # GRAFİKLER
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.1, row_heights=[0.6, 0.4],
                            subplot_titles=("Portföy Performansı", "Model Otorite Dağılımı (Kimin Sözü Geçiyor?)"))
        
        # 1. Equity Curve
        fig.add_trace(go.Scatter(x=dates, y=equity, name="Ensemble Bot", line=dict(color="#00ff00")), row=1, col=1)
        
        # 2. Ağırlıklar (Stacked Area)
        fig.add_trace(go.Scatter(x=dates, y=weights_history['HMM'], name="HMM (Rejim)", stackgroup='one', line=dict(width=0)), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=weights_history['ARIMA'], name="ARIMA (Trend)", stackgroup='one', line=dict(width=0)), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=weights_history['RF'], name="Random Forest (Teknik)", stackgroup='one', line=dict(width=0)), row=2, col=1)
        
        fig.update_layout(height=700, template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        ℹ️ **Grafik Analizi:**
        Alttaki renkli grafik, "Meclis" içindeki güç dağılımını gösterir.
        * Bir dönem **ARIMA** (Trend) alanı genişlediyse, o dönem trendler çok netti ve ARIMA haklı çıktı demektir.
        * Piyasa karışınca **Random Forest** veya **HMM** alanı genişler.
        * Bot, **"Dün kim haklı çıktıysa bugün parayı ona emanet et"** mantığıyla (Minimum Loss) çalışır.
        """)
