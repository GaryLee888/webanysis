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

# --- CSS 修飾：移除編號、縮小輸入框寬度、標題置中 ---
st.markdown("""
    <style>
    /* 側邊欄背景與文字顏色 */
    [data-testid="stSidebar"] {
        background-color: #1c2833;
        color: #fcf3cf;
    }
    
    /* 隱藏標籤 */
    [data-testid="stSidebar"] .stTextInput label {
        display: none;
    }
    
    /* 讓輸入框容器置中並縮小寬度 (50%) */
    [data-testid="stSidebar"] .stTextInput {
        width: 50% !important;
        margin: 0 auto !important;
        margin-bottom: -15px !important;
    }

    /* 調整輸入框內文字與高度 */
    [data-testid="stSidebar"] input {
        height: 30px !important;
        font-size: 0.9rem !important;
        text-align: center !important; /* 輸入內容也置中 */
        border-radius: 2px !important;
    }

    /* 啟動分析按鈕樣式 */
    [data-testid="stSidebar"] button {
        background-color: #e67e22 !important;
        color: white !important;
        margin-top: 25px !important;
        width: 80% !important;
        margin-left: auto !important;
        margin-right: auto !important;
        display: block !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. 字體設定 (解決圖表方塊字) ---
def set_mpl_chinese():
    font_file = 'msjh.ttc' 
    if os.path.exists(font_file):
        fe = fm.FontEntry(fname=font_file, name='CustomFont')
        fm.fontManager.ttflist.insert(0, fe)
        plt.rcParams['font.sans-serif'] = ['CustomFont']
    else:
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 

set_mpl_chinese()

# --- 2. 輔助工具：價格對齊 0.05 ---
def round_stock_price(price):
    return np.round(price * 20) / 20

# --- 3. 核心引擎 ---
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
        df['MFI'] = 50 + (df['Close'].diff().rolling(14).mean() * 10)
        df['VMA20'] = df['Volume'].rolling(win).mean()
        df['BIAS5'] = (df['Close'] - df['MA5']) / df['MA5'] * 100
        df['BIAS20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
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

# --- 4. UI 介面 ---
st.title("🚀 台股決策分析系統")

with st.sidebar:
    # 標題置中
    st.markdown("<h3 style='color:#fcf3cf; text-align:center;'>代碼/名稱</h3>", unsafe_allow_html=True)
    
    default_vals = ["2330", "2317", "2454", "6223", "2603", "2881", "貝爾威勒", "", "", ""]
    queries = []
    
    # 移除編號，純輸入框顯示
    for i in range(10):
        val = st.text_input("", value=default_vals[i], key=f"in_{i}")
        if val.strip():
            queries.append(val.strip())
            
    analyze_btn = st.button("啟動分析")

engine = StockEngine()

if analyze_btn and queries:
    tabs = st.tabs([f" {q} " for q in queries])
    for i, query in enumerate(queries):
        with tabs[i]:
            sid = engine.special_mapping.get(query, query)
            stock_name = query
            if not sid.isdigit():
                for code, info in twstock.codes.items():
                    if query in info.name: sid = code; stock_name = info.name; break
            elif sid in twstock.codes:
                stock_name = twstock.codes[sid].name

            df_raw, ticker = engine.fetch_data(sid)
            if df_raw is None: 
                st.error(f"查無數據: {sid}")
                continue

            df = engine.calculate_indicators(df_raw)
            chip_data = engine.fetch_chips(sid)
            curr, prev = df.iloc[-1], df.iloc[-2]
            
            entry_p = round_stock_price((curr['MA20'] + curr['BB_up']) / 2 if curr['Close'] <= curr['BB_up'] else curr['Close'] * 0.98)
            sl_p = round_stock_price(entry_p - (float(curr['ATR']) * 2.2))
            tp_p = round_stock_price(entry_p + (entry_p - sl_p) * 2.0)

            st.markdown(f"### 📊 綜合診斷")
            # 後續邏輯保持不變...
            st.info(f"當前標的: {stock_name} ({sid})")
            
            # (下略詳細診斷與圖表程式碼，與前版一致)
