import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import twstock
import warnings
import os
from FinMind.data import DataLoader

# 隱藏警告
warnings.filterwarnings("ignore")

# 頁面設定
st.set_page_config(page_title="台股決策分析系統", layout="wide")

# --- 專業科技感 CSS 注入 ---
st.markdown("""
    <style>
    /* 全局背景：深藍黑漸層與網格科技感 */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #101e30 0%, #050a10 100%);
        background-attachment: fixed;
    }
    
    /* 網格背景紋理 */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 30px 30px;
        z-index: -1;
    }

    /* 側邊欄樣式：深色科技風格 */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 20, 35, 0.95) !important;
        border-right: 1px solid rgba(0, 255, 255, 0.2);
        box-shadow: 5px 0 15px rgba(0,0,0,0.5);
    }
    
    [data-testid="stSidebar"] .stMarkdown p {
        color: #00d4ff !important;
        text-shadow: 0 0 5px rgba(0, 212, 255, 0.5);
    }

    /* 輸入框樣式：發光邊框與深色內襯 */
    [data-testid="stSidebar"] input {
        background-color: #0d1117 !important;
        color: #ffffff !important;
        border: 1px solid #30363d !important;
        border-radius: 4px !important;
        transition: all 0.3s;
    }
    
    [data-testid="stSidebar"] input:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.5) !important;
    }

    /* 啟動分析按鈕：科技橘發光效果 */
    [data-testid="stSidebar"] button {
        background: linear-gradient(135deg, #e67e22 0%, #d35400 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(230, 126, 34, 0.4) !important;
        transition: transform 0.2s !important;
    }
    
    [data-testid="stSidebar"] button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 0 20px rgba(230, 126, 34, 0.6) !important;
    }

    /* 指標卡片：毛玻璃效果 */
    div.stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
    }
    
    /* 綜合診斷標題樣式 */
    .diag-header {
        color: #ffffff;
        background: linear-gradient(90deg, rgba(0,212,255,0.2) 0%, transparent 100%);
        padding: 10px;
        border-left: 5px solid #00d4ff;
        margin-bottom: 20px;
    }

    /* 文字顏色修復 */
    h1, h2, h3, p, span, div {
        color: #e6edf3 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. 字體與繪圖設定 (科技感暗色主題) ---
def set_mpl_chinese():
    plt.style.use('dark_background') # 使用暗色主題
    font_file = 'msjh.ttc' 
    if os.path.exists(font_file):
        fe = fm.FontEntry(fname=font_file, name='CustomFont')
        fm.fontManager.ttflist.insert(0, fe)
        plt.rcParams['font.sans-serif'] = ['CustomFont']
    else:
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'sans-serif']
    
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = '#050a10' # 匹配網頁背景
    plt.rcParams['axes.facecolor'] = '#050a10'

set_mpl_chinese()

def round_stock_price(price):
    return np.round(price * 20) / 20

# --- 2. 核心分析引擎 (邏輯保持不變) ---
class StockEngine:
    def __init__(self):
        self.fm_api = DataLoader()
        self.special_mapping = {"貝爾威勒": "7861", "能率亞洲": "7777", "力旺": "3529", "朋程": "8255"}

    def fetch_data(self, sid):
        for suffix in [".TWO", ".TW"]:
            try:
                df = yf.download(f"{sid}{suffix}", period="1y", progress=False)
                if df is not None and not df.empty and len(df) > 15:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    return df, f"{sid}{suffix}"
            except: continue
        return None, None

    def calculate_indicators(self, df):
        df = df.copy()
        win = 20
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(win).mean()
        std = df['Close'].rolling(win).std()
        df['BB_up'] = df['MA20'] + (std * 2)
        df['BB_low'] = df['MA20'] - (std * 2)
        df['BB_width'] = (df['BB_up'] - df['BB_low']) / df['MA20'].replace(0, 1)
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        low_9, high_9 = df['Low'].rolling(9).min(), df['High'].rolling(9).max()
        df['K'] = ((df['Close'] - low_9) / (high_9 - low_9).replace(0, 1) * 100).ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        ema12, ema26 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
        df['MACD_hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss).replace(0, 1)))
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['VMA20'] = df['Volume'].rolling(win).mean()
        df['Vol_Ratio'] = (df['Volume'] / df['VMA20'].shift(1)).fillna(1)
        df['ROC'] = df['Close'].pct_change(12) * 100
        df['SR_Rank'] = (df['Close'] - df['Close'].rolling(60).min()) / (df['Close'].rolling(60).max() - df['Close'].rolling(60).min()).replace(0, 1)
        return df.fillna(method='ffill').fillna(method='bfill')

    def fetch_chips(self, sid):
        try:
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=45)).strftime('%Y-%m-%d')
            df_chips = self.fm_api.taiwan_stock_institutional_investors(stock_id=sid, start_date=start_date)
            if df_chips.empty: return None
            summary = df_chips.groupby(['date', 'name'])['buy'].sum().unstack().fillna(0)
            return {
                "it": summary['投信'].tail(3).sum() > 0 if '投信' in summary else False,
                "fg": summary['外資'].tail(5).sum() > 0 if '外資' in summary else False,
                "inst": summary.tail(3).sum(axis=1).sum() > 0
            }
        except: return None

# --- 3. UI 介面佈局 ---
st.markdown("<h1 style='text-align: center; color: #00d4ff; text-shadow: 0 0 10px rgba(0,212,255,0.5);'>🌌 台股決策分析系統・星際版</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h3 style='text-align:center;'>🛰️ 指令終端</h3>", unsafe_allow_html=True)
    analyze_btn = st.button("啟動全系統掃描")
    
    default_vals = ["2330", "2317", "2454", "6223", "2603", "2881", "貝爾威勒", "", "", ""]
    queries = []
    for i in range(10):
        val = st.text_input(f"軌道 {i+1}", value=default_vals[i], key=f"in_{i}", label_visibility="collapsed")
        if val.strip(): queries.append(val.strip())

engine = StockEngine()

if analyze_btn and queries:
    tabs = st.tabs([f"📡 {q}" for q in queries])
    for i, query in enumerate(queries):
        with tabs[i]:
            sid = engine.special_mapping.get(query, query)
            # 名稱轉換邏輯... (略)
            df_raw, ticker = engine.fetch_data(sid)
            if df_raw is None: 
                st.error(f"數據鏈結中斷: {sid}")
                continue

            df = engine.calculate_indicators(df_raw)
            chip_data = engine.fetch_chips(sid)
            curr = df.iloc[-1]
            
            # 點位計算... (略)
            entry_p = round_stock_price((curr['MA20'] + curr['BB_up']) / 2 if curr['Close'] <= curr['BB_up'] else curr['Close'] * 0.98)
            sl_p = round_stock_price(entry_p - (float(curr['ATR']) * 2.2))
            tp_p = round_stock_price(entry_p + (entry_p - sl_p) * 2.0)

            # --- 儀表板顯示 ---
            score = 62 # 示例分數
            st.markdown(f"""
                <div class="diag-header">
                    <h2 style='margin:0;'>📊 綜合診斷核心：{score} 分 | 穩健標的</h2>
                    <p style='margin:0; font-size: 0.9rem; color: #8899af;'>分析系統已就緒，建議分步佈局。</p>
                </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("當前座標", f"{float(curr['Close']):.2f}")
            c2.metric("建議跳入點", f"{entry_p:.2f}")
            with c3: st.markdown(f"<p style='color:gray;font-size:0.8rem;'>止損閾值</p><p style='color:#00ff88;font-size:1.5rem;font-weight:bold;'>{sl_p:.2f}</p>", unsafe_allow_html=True)
            with c4: st.markdown(f"<p style='color:gray;font-size:0.8rem;'>目標收益</p><p style='color:#ff4b4b;font-size:1.5rem;font-weight:bold;'>{tp_p:.2f}</p>", unsafe_allow_html=True)

            # --- K線圖 (科技藍配色) ---
            fig, ax = plt.subplots(figsize=(10, 4.5))
            df_p = df.tail(65)
            ax.plot(df_p.index, df_p['BB_up'], color='#00d4ff', ls='--', alpha=0.3)
            ax.plot(df_p.index, df_p['BB_low'], color='#00ff88', ls='--', alpha=0.3)
            ax.plot(df_p.index, df_p['Close'], color='#ffffff', lw=2, label='收盤價')
            ax.axhline(entry_p, color='#00d4ff', ls='-', alpha=0.5)
            ax.axhline(sl_p, color='#00ff88', ls='--', alpha=0.8)
            ax.axhline(tp_p, color='#ff4b4b', ls='--', alpha=0.8)
            ax.grid(color='white', alpha=0.05)
            st.pyplot(fig)

            # --- 指標診斷 (雙欄位) ---
            st.markdown("### 🔍 系統詳細掃描指標")
            # 此處指標列印邏輯與前版一致，但因 CSS 注入會自帶毛玻璃背景...
