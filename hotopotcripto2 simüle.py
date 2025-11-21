import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund: Meta-Learning Portfolio", layout="wide")
st.title("🏦 Otonom Hedge Fonu: Çoklu Coin Simülasyonu")
st.markdown("""
Bu modül, geliştirdiğimiz **Meta-Learning (Hakemli)** yapıyı tüm portföye uygular.
Her coin için ayrı bir "Bot" atanır. Her bot kendi coininin karakterine göre **Macro (Geçmiş)** veya **Micro (Trend)** veriye güveneceğine kendisi karar verir.
""")

# --- AYARLAR ---
with st.sidebar:
    st.header("⚙️ Fon Ayarları")
    # Geniş Portföy
    default_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "DOGE-USD"]
    selected_tickers = st.multiselect("Portföydeki Coinler", default_tickers, default=["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"])
    
    capital_per_coin = st.number_input("Coin Başına Sermaye ($)", value=1000)
    test_days = st.number_input("Test Süresi (Gün)", value=90)
    
    st.divider()
    st.info("Botlar şu an hesaplama yapıyor... Bu işlem biraz zaman alabilir.")

# --- CORE MOTOR (TEK COIN İÇİN) ---
def train_predict_hmm(train_data, current_features):
    """HMM Modeli eğit ve o anlık sinyal üret"""
    if len(train_data) < 30: return 0
    
    X = train_data[['log_ret', 'range']].values
    scaler = StandardScaler()
    try:
        X_s = scaler.fit_transform(X)
        # Hız optimizasyonu: iterasyon 15, diag kovaryans
        model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=15, random_state=42)
        model.fit(X_s)
        
        means = model.means_[:, 0]
        bull = np.argmax(means)
        bear = np.argmin(means)
        
        curr_s = scaler.transform(current_features.reshape(1, -1))
        probs = model.predict_proba(curr_s)[0]
        
        return probs[bull] - probs[bear]
    except:
        return 0

def run_meta_simulation(ticker, days, cap):
    # Veri İndir
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
    
    # Feature Engineering
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['range'] = (df['high'] - df['low']) / df['close']
    df['target'] = np.sign(df['close'].shift(-1) - df['close']) # Başarı ölçümü için
    df.dropna(inplace=True)
    
    if len(df) < days + 60: return None
    
    start_idx = len(df) - days
    cash = cap
    coin = 0
    equity = []
    
    # Hakem Hafızası
    macro_correctness = [0] * 10
    micro_correctness = [0] * 10
    
    # Simülasyon Loop
    for i in range(start_idx, len(df)-1):
        # Veri Dilimleri
        df_macro = df.iloc[:i]      # Tüm geçmiş
        df_micro = df.iloc[i-60:i]  # Son 2 ay
        
        curr_feat = df.iloc[i][['log_ret', 'range']].values
        
        # Tahminler
        macro_sig = train_predict_hmm(df_macro, curr_feat)
        micro_sig = train_predict_hmm(df_micro, curr_feat)
        
        # Hakem Kararı (Ağırlık)
        m_perf = sum(macro_correctness)
        mi_perf = sum(micro_correctness)
        total = m_perf + mi_perf
        w_macro = m_perf / total if total > 0 else 0.5
        w_micro = 1.0 - w_macro
        
        # Sinyal Birleştirme
        final_signal = (macro_sig * w_macro) + (micro_sig * w_micro)
        
        # İşlem
        price = df.iloc[i]['close']
        if final_signal > 0.25 and cash > 0:
            coin = cash / price
            cash = 0
        elif final_signal < -0.25 and coin > 0:
            cash = coin * price
            coin = 0
            
        equity.append(cash + (coin * price))
        
        # Öğrenme (Puanlama)
        actual = df.iloc[i]['target']
        macro_correctness.pop(0)
        macro_correctness.append(1 if np.sign(macro_sig) == actual else 0)
        micro_correctness.pop(0)
        micro_correctness.append(1 if np.sign(micro_sig) == actual else 0)

    # Sonuç Hesaplama
    final_val = equity[-1]
    roi = (final_val - cap) / cap
    hodl_roi = (df.iloc[-1]['close'] - df.iloc[start_idx]['close']) / df.iloc[start_idx]['close']
    
    return {
        "ticker": ticker,
        "final_balance": final_val,
        "roi": roi,
        "hodl_roi": hodl_roi,
        "equity_curve": equity
    }

# --- ÇALIŞTIRMA BUTONU ---
if st.button("🚀 FONU BAŞLAT (Tüm Botları Çalıştır)", type="primary"):
    if not selected_tickers:
        st.error("Lütfen coin seçin.")
    else:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Portföy Döngüsü
        for idx, t in enumerate(selected_tickers):
            status_text.text(f"Bot çalışıyor: {t} (%{int((idx/len(selected_tickers))*100)})")
            res = run_meta_simulation(t, test_days, capital_per_coin)
            if res:
                results.append(res)
            progress_bar.progress((idx + 1) / len(selected_tickers))
            
        status_text.empty()
        progress_bar.empty()
        
        if results:
            # --- TOPLAM PORTFÖY ANALİZİ ---
            total_invested = capital_per_coin * len(results)
            total_final = sum([r['final_balance'] for r in results])
            total_roi = (total_final - total_invested) / total_invested
            
            # HODL Toplamı
            # Her coinin hodl getirisini sermaye ile çarpıp toplayalım
            total_hodl_balance = sum([capital_per_coin * (1 + r['hodl_roi']) for r in results])
            total_hodl_roi = (total_hodl_balance - total_invested) / total_invested
            
            # ALPHA
            alpha = total_roi - total_hodl_roi
            
            st.success("✅ Simülasyon Tamamlandı!")
            
            # 1. BÜYÜK ÖZET (SCOREBOARD)
            k1, k2, k3 = st.columns(3)
            k1.metric("FON TOPLAM DEĞERİ", f"${total_final:,.0f}", f"%{total_roi*100:.1f} (Net Kâr)")
            k2.metric("Piyasa (HODL) Değeri", f"${total_hodl_balance:,.0f}", f"%{total_hodl_roi*100:.1f}")
            k3.metric("FON ALPHA (FARK)", f"${total_final - total_hodl_balance:,.0f}", f"%{alpha*100:.1f}", delta_color="normal")
            
            st.markdown("---")
            
            # 2. DETAYLI TABLO
            summary_data = []
            for r in results:
                summary_data.append({
                    "Coin": r['ticker'],
                    "Bot Kârı (%)": f"%{r['roi']*100:.1f}",
                    "HODL (%)": f"%{r['hodl_roi']*100:.1f}",
                    "Alpha (Fark)": f"%{(r['roi'] - r['hodl_roi'])*100:.1f}",
                    "Son Bakiye": f"${r['final_balance']:.0f}"
                })
            st.dataframe(pd.DataFrame(summary_data))
            
            # 3. KÜMÜLATİF GRAFİK
            # Tüm coinlerin equity eğrilerini toplayarak Fon Grafiği çiz
            # (Basitlik için sadece toplamı değil, her coini ayrı çizelim karışmasın)
            
            fig = go.Figure()
            
            # Ana Fon Eğrisi (Yaklaşık birleştirme - uzunluklar eşitse)
            min_len = min([len(r['equity_curve']) for r in results])
            total_equity_curve = np.zeros(min_len)
            
            for r in results:
                # Son 'min_len' kadarını al
                curve = np.array(r['equity_curve'][-min_len:])
                total_equity_curve += curve
                
                # Bireysel Çizgiler (Opak)
                fig.add_trace(go.Scatter(y=curve, name=r['ticker'], opacity=0.3, line=dict(width=1)))

            # Toplam Fon Çizgisi (Kalın)
            fig.add_trace(go.Scatter(y=total_equity_curve, name="TOPLAM FON", line=dict(color="#00ff00", width=4)))
            
            fig.update_layout(title="Fon Büyümesi vs Bireysel Coinler", template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("Hiçbir coin için sonuç üretilemedi.")
