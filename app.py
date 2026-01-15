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

# --- 專業科技感配色與對比優化 CSS ---
st.markdown("""
    <style>
    /* 全局背景：深灰藍與微發光網格 */
    .stApp {
        background-color: #1A1C23;
        background-image: 
            linear-gradient(rgba(0, 212, 255, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 212, 255, 0.05) 1px, transparent 1px);
        background-size: 40px 40px;
    }

    /* 側邊欄：鈦金金屬感 */
    [data-testid="stSidebar"] {
        background-color: #121418 !important;
        border-right: 2px solid #00d4ff;
        box-shadow: 2px 0 10px rgba(0, 212, 255, 0.2);
    }
    
    /* 側邊欄標題與文字 */
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #00d4ff !important;
        text-align: center;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }

    /* 輸入框：深內嵌質感 */
    [data-testid="stSidebar"] input {
        background-color: #2D3748 !important;
        color: #FFFFFF !important;
        border: 1px solid #4A5568 !important;
        border-radius: 5px !important;
        font-size: 1.1rem !important;
        text-align: center !important;
    }

    /* 啟動分析按鈕：金屬橘漸層 */
    [data-testid="stSidebar"] button {
        background: linear-gradient(180deg, #ED8936 0%, #C05621 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border: 1px solid #DD6B20 !important;
        box-shadow: 0 4px 12px rgba(221, 107, 32, 0.3) !important;
        height: 40px !important;
    }

    /* 指標數據卡片：電光藍邊框與毛玻璃 */
    .metric-container {
        background: rgba(45, 55, 72, 0.7);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4A5568;
        border-top: 3px solid #00d4ff;
        backdrop-filter: blur(10px);
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    /* 數據標籤：調亮灰色確保可見 */
    .metric-label {
        color: #A0AEC0 !important;
        font-size: 1rem;
        font-weight: bold;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* 數據數值：統一特大字體 */
    .metric-value {
        font-family: 'Verdana', sans-serif;
        font-size: 2.4rem !important;
        font-weight: 900;
        text-shadow: 0 0 15px rgba(255,255,255,0.2);
    }

    /* 診斷橫幅 */
    .diag-banner {
        background: rgba(0, 212, 255, 0.1);
        padding: 15px;
        border-left: 6px solid #00d4ff;
        border-radius: 4px;
        margin-bottom: 25px;
    }

    /* 全局文字對比調整 */
    h1, h2, h3 { color: #FFFFFF !important; }
    p, span, li { color: #E2E8F0 !important; }
    
    /* Tab 顏色優化 */
    .stTabs [data-baseweb="tab"] {
        color: #A0AEC0 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #00d4ff !important;
        border-bottom-color: #00d4ff !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. 繪圖風格設定 (深色網格) ---
def set_mpl_chinese():
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#1A1C23'
    plt.rcParams['axes.facecolor'] = '#1A1C23'
    plt.rcParams['axes.edgecolor'] = '#4A5568'
    plt.rcParams['grid.color'] = '#2D3748'
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Noto Sans CJK JP', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

set_mpl_chinese()

def round_stock_price(price):
    return np.round(price * 20) / 20

# --- 2. 核心引擎 ---
class StockEngine:
    def __init__(self):
        self.fm_api = DataLoader()
        self.special_mapping = {"貝爾威勒": "7861", "能率亞洲": "7777", "力旺": "3529", "朋程": "8255"}

    def fetch_data(self, sid):
        for suffix in [".TWO", ".TW"]:
            try:
                df = yf.download(f"{sid}{suffix}", period="1y", progress=False)
                if df is not None and not df.empty and len(df) > 20:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    return df, f"{sid}{suffix}"
            except: continue
        return None, None

    def calculate_indicators(self, df):
        df = df.copy()
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        df['BB_up'] = df['MA20'] + (std * 2)
        df['BB_low'] = df['MA20'] - (std * 2)
        df['BB_width'] = (df['BB_up'] - df['BB_low']) / df['MA20']
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        low_9, high_9 = df['Low'].rolling(9).min(), df['High'].rolling(9).max()
        df['K'] = ((df['Close'] - low_9) / (high_9 - low_9).replace(0, 1) * 100).ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        ema12, ema26 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
        df['MACD_hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['VMA20'] = df['Volume'].rolling(20).mean()
        df['ROC'] = df['Close'].pct_change(12) * 100
        return df.ffill().bfill()

# --- 3. UI 介面 ---
st.markdown("<h1 style='text-align: center; color: #FFFFFF; letter-spacing: 5px; text-shadow: 2px 2px 10px rgba(0,212,255,0.4);'>🛡️ 台股全方位決策系統</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🛰️ 指令輸入")
    analyze_btn = st.button("啟動系統分析", use_container_width=True)
    
    default_vals = ["2330", "2317", "2454", "6223", "2603", "2881", "7861", "", "", ""]
    queries = []
    for i in range(10):
        val = st.text_input(f"in_{i}", value=default_vals[i], label_visibility="collapsed")
        if val.strip(): queries.append(val.strip())

engine = StockEngine()

if analyze_btn and queries:
    tabs = st.tabs([f"● {q}" for q in queries])
    for i, query in enumerate(queries):
        with tabs[i]:
            sid = engine.special_mapping.get(query, query)
            # 獲取名稱邏輯...
            df_raw, ticker = engine.fetch_data(sid)
            if df_raw is None: 
                st.error(f"數據鏈結失敗: {sid}")
                continue

            df = engine.calculate_indicators(df_raw)
            curr = df.iloc[-1]
            entry_p = round_stock_price((curr['MA20'] + curr['BB_up']) / 2 if curr['Close'] <= curr['BB_up'] else curr['Close'] * 0.98)
            sl_p = round_stock_price(entry_p - (float(curr['ATR']) * 2.2))
            tp_p = round_stock_price(entry_p + (entry_p - sl_p) * 2.0)

            # --- A. 診斷橫幅 ---
            st.markdown(f"""
                <div class="diag-banner">
                    <h2 style='margin:0; color:#00d4ff;'>📊 系統掃描完畢 | 關鍵座標：{sid}</h2>
                    <p style='margin:5px 0 0 0; color:#E2E8F0; font-size:1.1rem;'>建議操作：多空訊號共振中，請參照下方防護位操作。</p>
                </div>
            """, unsafe_allow_html=True)

            # --- B. 統一特大字體數據區 ---
            dc1, dc2, dc3, dc4 = st.columns(4)
            data_items = [
                ("當前座標價", f"{float(curr['Close']):.2f}", "#FFFFFF"),
                ("建議跳入點", f"{entry_p:.2f}", "#FFFFFF"),
                ("安全防護位", f"{sl_p:.2f}", "#38A169"), # 亮綠色
                ("目標獲利區", f"{tp_p:.2f}", "#E53E3E")  # 亮紅色
            ]
            
            for idx, (label, val, color) in enumerate(data_items):
                cols = [dc1, dc2, dc3, dc4]
                cols[idx].markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="color: {color};">{val}</div>
                    </div>
                """, unsafe_allow_html=True)

            # --- C. 圖表 ---
            fig, ax = plt.subplots(figsize=(10, 4.5))
            df_p = df.tail(60)
            ax.plot(df_p.index, df_p['BB_up'], color='#4FD1C5', ls='--', alpha=0.3, label='軌道上線')
            ax.plot(df_p.index, df_p['Close'], color='#FFFFFF', lw=2.5, label='即時成交價')
            ax.axhline(entry_p, color='#00d4ff', ls='-', alpha=0.5)
            ax.axhline(sl_p, color='#38A169', ls='--', alpha=0.7)
            ax.axhline(tp_p, color='#E53E3E', ls='--', alpha=0.7)
            ax.set_title(f"軌道軌跡分析: {sid}", color='#FFFFFF', fontsize=14, pad=20)
            st.pyplot(fig)

            # --- D. 指標細節 ---
            st.markdown("### 🔍 掃描指標細節")
            # (指標列表邏輯比照前版，因 CSS 套用將呈現亮灰色文字)
