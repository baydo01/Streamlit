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

st.set_page_config(page_title="Auto-Adaptive AI", layout="wide")
st.title("🧬 Tam Otonom Adaptif Model (Meta-Learning)")
st.markdown("""
Bu sistemde **"Geçmişin Önemi"** ayarı yoktur. Sistem buna kendi karar verir:
1.  Her gün **Macro** (Tarihsel) ve **Micro** (Güncel) modeller ayrı ayrı tahmin yapar.
2.  **Hakem (Meta-Learner):** Son 10 gündeki başarılarına bakar. Kim daha iyi bildiyse, bugünkü işlem yetkisini ona verir.
3.  Sonuç: Piyasa değiştiğinde modelin "Kime güveneceğini" değiştirdiği dinamik bir yapı.
""")

# --- AYARLAR ---
with st.sidebar:
    ticker = st.selectbox("Coin Seç", ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "XRP-USD"])
    capital = st.number_input("Başlangıç ($)", value=1000)
    test_days = st.number_input("Simülasyon Süresi (Gün)", value=180, help="Geriye dönük son kaç gün simüle edilsin?")
    st.info("⚠️ Dikkat: Bu model her gün için HMM'i yeniden eğittiği için işlem biraz zaman alabilir.")

def get_data(ticker):
    df = yf.download(ticker, period="2y", interval="1d", progress=False) # 2 Yıl veri yeterli
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
    
    # Feature Engineering
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['range'] = (df['high'] - df['low']) / df['close']
    df['volatility'] = df['log_ret'].rolling(20).std()
    
    # Hedef (Yarın artacak mı?) - Başarı ölçümü için
    df['target'] = np.sign(df['close'].shift(-1) - df['close']) # 1 (Artış) veya -1 (Düşüş)
    
    df.dropna(inplace=True)
    return df

def train_predict_hmm(train_data, current_features):
    """Verilen veriyle eğitir, o an için tahmin üretir"""
    if len(train_data) < 30: return 0
    
    X = train_data[['log_ret', 'range']].values
    scaler = StandardScaler()
    try:
        X_s = scaler.fit_transform(X)
        # Hız için iterasyon düşük, covariance diag
        model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=20, random_state=42)
        model.fit(X_s)
        
        means = model.means_[:, 0]
        bull = np.argmax(means)
        bear = np.argmin(means)
        
        # Tahmin
        curr_s = scaler.transform(current_features.reshape(1, -1))
        probs = model.predict_proba(curr_s)[0]
        
        signal = probs[bull] - probs[bear] # -1 ile 1 arası
        return signal
    except:
        return 0

# --- SİMÜLASYON ---
if st.button("🧬 Otonom Evrimi Başlat"):
    df = get_data(ticker)
    
    if len(df) < test_days + 100:
        st.error("Yetersiz veri.")
    else:
        # Simülasyon Başlangıcı
        start_idx = len(df) - test_days
        
        cash = capital
        coin = 0
        equity = []
        dates = []
        
        # Analiz Kayıtları
        history_weights = [] # Modelin geçmişe verdiği ağırlık
        macro_scores = []
        micro_scores = []
        
        # Son 10 günün tahmin başarısını tutan listeler
        macro_correctness = [0] * 10
        micro_correctness = [0] * 10
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        # --- GÜN GÜN İLERLEME (ROLLING LOOP) ---
        for i in range(start_idx, len(df)-1): # Son günün target'ı yoktur
            prog = (i - start_idx) / test_days
            progress_bar.progress(min(prog, 1.0))
            
            # 1. VERİ SETLERİNİ HAZIRLA
            # Macro: Başlangıçtan bugüne kadar olan her şey (Geniş hafıza)
            df_macro = df.iloc[:i] 
            # Micro: Sadece son 60 gün (Kısa hafıza)
            df_micro = df.iloc[i-60:i]
            
            current_row = df.iloc[i]
            curr_feat = current_row[['log_ret', 'range']].values
            
            # 2. MODELLERİ ÇALIŞTIR (Tahmin Al)
            macro_sig = train_predict_hmm(df_macro, curr_feat)
            micro_sig = train_predict_hmm(df_micro, curr_feat)
            
            # 3. DİNAMİK AĞIRLIK HESAPLA (HAKEM)
            # Son 10 günde kim daha başarılıydı?
            macro_perf = sum(macro_correctness)
            micro_perf = sum(micro_correctness)
            total_perf = macro_perf + micro_perf
            
            if total_perf == 0:
                weight_macro = 0.5 # Bilgi yoksa eşit
            else:
                weight_macro = macro_perf / total_perf
            
            weight_micro = 1.0 - weight_macro
            
            # Kayıt (Grafik için)
            history_weights.append(weight_macro)
            
            # 4. HİBRİT KARAR
            final_signal = (macro_sig * weight_macro) + (micro_sig * weight_micro)
            
            # 5. İŞLEM YAP
            price = current_row['close']
            if final_signal > 0.3 and cash > 0:
                coin = cash / price
                cash = 0
            elif final_signal < -0.3 and coin > 0:
                cash = coin * price
                coin = 0
            
            val = cash + (coin * price)
            equity.append(val)
            dates.append(df.index[i])
            
            # 6. ÖĞRENME (GERÇEKLEŞME KONTROLÜ)
            # Yarın fiyat ne oldu? (Target sütunundan bak)
            actual_move = current_row['target'] # +1 veya -1
            
            # Modellerin tahmini doğru muydu?
            # Eğer Macro sinyali pozitifti ve fiyat arttıysa -> Başarılı (1 puan)
            macro_success = 1 if np.sign(macro_sig) == actual_move else 0
            micro_success = 1 if np.sign(micro_sig) == actual_move else 0
            
            # Listeyi kaydır (En eskiyi sil, yeniyi ekle)
            macro_correctness.pop(0)
            macro_correctness.append(macro_success)
            micro_correctness.pop(0)
            micro_correctness.append(micro_success)
            
        progress_bar.empty()
        status.empty()
        
        # --- RAPORLAMA ---
        final_roi = (equity[-1] - capital) / capital
        hodl_roi = (df['close'].iloc[-1] - df['close'].iloc[start_idx]) / df['close'].iloc[start_idx]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Otonom Bot Kârı", f"%{final_roi*100:.1f}", f"${equity[-1]:.0f}")
        c2.metric("HODL", f"%{hodl_roi*100:.1f}")
        c3.metric("Alpha", f"%{(final_roi - hodl_roi)*100:.1f}")
        
        # GRAFİK: PnL ve Ağırlık Değişimi
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.1, row_heights=[0.7, 0.3],
                            subplot_titles=("Portföy Performansı", "Yapay Zeka Karar Mekanizması (Macro vs Micro Güveni)"))
        
        # Üst Panel: Equity
        fig.add_trace(go.Scatter(x=dates, y=equity, name="Bot Bakiye", line=dict(color="#00ff00")), row=1, col=1)
        
        # Alt Panel: Ağırlıklar (Area Chart)
        fig.add_trace(go.Scatter(
            x=dates, y=history_weights, name="Geçmişe Güven (Macro)",
            stackgroup='one', line=dict(width=0, color='blue'), opacity=0.5
        ), row=2, col=1)
        
        # Micro güveni (1 - Macro) olarak dolaylı görünür ama görsel için ekleyelim
        micro_w_list = [1-w for w in history_weights]
        fig.add_trace(go.Scatter(
            x=dates, y=micro_w_list, name="Trende Güven (Micro)",
            stackgroup='one', line=dict(width=0, color='orange'), opacity=0.5
        ), row=2, col=1)
        
        fig.update_layout(height=600, template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("""
        **Grafiği Nasıl Okumalısın?**
        * **Alt Grafik (Mavi/Turuncu):** Botun beyninin içi.
        * **Mavi Alan Genişlerse:** Bot diyor ki *"Piyasa çok karışık, son günlere güvenilmez, ben eski tecrübelerime (Macro) sığınıyorum."*
        * **Turuncu Alan Genişlerse:** Bot diyor ki *"Şu an yeni bir trend var, eski veriler geçersiz, son 2 aya (Micro) odaklanıyorum."*
        """)
