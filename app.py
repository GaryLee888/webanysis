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

# --- 專業科技感 CSS ---
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at 50% 50%, #101e30 0%, #050a10 100%);
        background-attachment: fixed;
    }
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
    [data-testid="stSidebar"] {
        background-color: rgba(16, 26, 40, 0.95) !important;
        border-right: 1px solid rgba(0, 212, 255, 0.2);
    }
    [data-testid="stSidebar"] .stTextInput, [data-testid="stSidebar"] .stButton {
        width: 150px !important;
        margin-left: 45px !important;
        padding: 0 !important;
    }
    [data-testid="stSidebar"] input {
        height: 35px !important;
        font-size: 1.1rem !important;
        text-align: center !important;
        margin-bottom: 4px !important;
    }
    [data-testid="stSidebar"] button {
        background-color: #e67e22 !important;
        color: white !important;
        height: 35px !important;
        margin-bottom: 8px !important;
    }
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        backdrop-filter: blur(10px);
        text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: bold; }
    h1, h2, h3, p, span { color: #e6edf3 !important; }
    </style>
    """, unsafe_allow_html=True)

def set_mpl_chinese():
    plt.style.use('dark_background')
    font_file = 'msjh.ttc' 
    if os.path.exists(font_file):
        fe = fm.FontEntry(fname=font_file, name='CustomFont')
        fm.fontManager.ttflist.insert(0, fe)
        plt.rcParams['font.sans-serif'] = ['CustomFont']
    else:
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = '#050a10'
    plt.rcParams['axes.facecolor'] = '#050a10'

set_mpl_chinese()

def round_stock_price(price):
    return np.round(price * 20) / 20

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
        # 確保數據為 Float
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 1. 均線類
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        
        # 2. 布林通道與乖離
        std = df['Close'].rolling(20).std()
        df['BB_up'] = df['MA20'] + (std * 2)
        df['BB_low'] = df['MA20'] - (std * 2)
        df['BB_width'] = (df['BB_up'] - df['BB_low']) / df['MA20']
        df['BIAS5'] = (df['Close'] - df['MA5']) / df['MA5'] * 100
        df['BIAS20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
        
        # 3. 波動與動能
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
        
        # 4. 成交量類
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['MFI'] = 50 + (df['Close'].diff().rolling(14).mean() * 10)
        df['VMA20'] = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = (df['Volume'] / df['VMA20'].shift(1)).fillna(1)
        df['ROC'] = df['Close'].pct_change(12) * 100
        df['SR_Rank'] = (df['Close'] - df['Close'].rolling(60).min()) / (df['Close'].rolling(60).max() - df['Close'].rolling(60).min()).replace(0, 1)
        
        return df.ffill().bfill()

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

# --- UI 介面 ---
st.markdown("<h1 style='text-align: center; color: #00d4ff;'>🌌 台股全方位決策系統</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h3 style='text-align:center; color:#fcf3cf;'>🛰️ 終端指令區</h3>", unsafe_allow_html=True)
    analyze_btn = st.button("啟動分析")
    default_vals = ["2330", "2317", "2454", "6223", "2603", "2881", "貝爾威勒", "", "", ""]
    queries = []
    for i in range(10):
        val = st.text_input("", value=default_vals[i], key=f"in_{i}", label_visibility="collapsed")
        if val.strip(): queries.append(val.strip())

engine = StockEngine()

if analyze_btn and queries:
    tabs = st.tabs([f"📡 {q}" for q in queries])
    for i, query in enumerate(queries):
        with tabs[i]:
            sid = engine.special_mapping.get(query, query)
            stock_name = query
            if not sid.isdigit():
                for code, info in twstock.codes.items():
                    if query in info.name: sid = code; stock_name = info.name; break
            elif sid in twstock.codes: stock_name = twstock.codes[sid].name

            df_raw, ticker = engine.fetch_data(sid)
            if df_raw is None: 
                st.error(f"數據鏈結失敗: {sid}")
                continue

            df = engine.calculate_indicators(df_raw)
            chip_data = engine.fetch_chips(sid)
            curr = df.iloc[-1]
            
            entry_p = round_stock_price((curr['MA20'] + curr['BB_up']) / 2 if curr['Close'] <= curr['BB_up'] else curr['Close'] * 0.98)
            sl_p = round_stock_price(entry_p - (float(curr['ATR']) * 2.2))
            tp_p = round_stock_price(entry_p + (entry_p - sl_p) * 2.0)

            # --- 指標診斷清單 (確保 KeyError 不再發生) ---
            indicator_list = [
                ("均線趨勢", (1.0 if curr['Close'] > curr['MA20'] else 0.0), "多頭", "空頭"),
                ("軌道位階", (1.0 if curr['Close'] > curr['BB_up'] else 0.5 if curr['Close'] > curr['MA20'] else 0.0), "上位", "中位", "下位"),
                ("KD動能", (1.0 if curr['K'] > curr['D'] else 0.0), "向上", "向下"),
                ("MACD趨勢", (1.0 if curr['MACD_hist'] > 0 else 0.0), "紅柱", "綠柱"),
                ("RSI強弱", (1.0 if curr['RSI'] > 50 else 0.0), "強勢", "弱勢"),
                ("均線排列", (1.0 if curr['MA5'] > curr['MA10'] else 0.0), "多頭", "糾結"),
                ("威廉指標", (1.0 if curr['K'] > 50 else 0.0), "看多", "看空"),
                ("乖離率", (1.0 if abs(curr['BIAS20']) < 10 else 0.0), "安全", "過熱"),
                ("波幅擠壓", (1.0 if curr['BB_width'] < 0.1 else 0.0), "蓄勢", "發散"),
                ("量價配合", (1.0 if curr['Close'] >= df.iloc[-2]['Close'] else 0.0), "穩健", "背離"),
                ("能量潮", (1.0 if curr['OBV'] > df['OBV'].mean() else 0.0), "集中", "渙散"),
                ("資金流向", (1.0 if curr['MFI'] > 50 else 0.0), "流入", "流出"),
                ("成交均量", (1.0 if curr['Volume'] > curr['VMA20'] else 0.0), "量增", "量縮"),
                ("價格變動", (1.0 if curr['ROC'] > 0 else 0.0), "正向", "負向"),
                ("[籌] 投信連買", (1.0 if chip_data and chip_data['it'] else 0.0), "佈局中", "無動作"),
                ("[籌] 外資波段", (1.0 if chip_data and chip_data['fg'] else 0.0), "加碼中", "調節中"),
                ("[籌] 法人集結", (1.0 if chip_data and chip_data['inst'] else 0.0), "共識買", "分散")
            ]
            score = int((sum([it[1] for it in indicator_list]) / len(indicator_list)) * 100)

            # --- 數據呈現 ---
            st.markdown(f"### 📊 綜合診斷：{score} 分")
            dc1, dc2, dc3, dc4 = st.columns(4)
            vals = [("現價", f"{curr['Close']:.2f}", "#fff"), ("建議買點", f"{entry_p:.2f}", "#fff"), 
                    ("防護閾值", f"{sl_p:.2f}", "#00ff88"), ("目標星雲", f"{tp_p:.2f}", "#ff4b4b")]
            for idx, (l, v, c) in enumerate(vals):
                with [dc1, dc2, dc3, dc4][idx]:
                    st.markdown(f'<div class="metric-container"><div style="color:#889af;">{l}</div><div class="metric-value" style="color:{c};">{v}</div></div>', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(10, 4.5))
            df_p = df.tail(65)
            ax.plot(df_p.index, df_p['BB_up'], color='#00d4ff', ls='--', alpha=0.3)
            ax.plot(df_p.index, df_p['BB_low'], color='#00ff88', ls='--', alpha=0.3)
            ax.plot(df_p.index, df_p['Close'], color='#fff', lw=2)
            ax.axhline(entry_p, color='#00d4ff', alpha=0.4); ax.axhline(sl_p, color='#00ff88', ls='--'); ax.axhline(tp_p, color='#ff4b4b', ls='--')
            st.pyplot(fig)

            st.markdown("### 🔍 深度掃描報告")
            ic1, ic2 = st.columns(2)
            for idx, it in enumerate(indicator_list):
                col = ic1 if idx < len(indicator_list)//2 + 1 else ic2
                icon = "🔴" if it[1] == 1.0 else "🟠" if it[1] == 0.5 else "🟢"
                color = "#ff4b4b" if it[1] == 1.0 else "orange" if it[1] == 0.5 else "#00ff88"
                col.markdown(f'<div>{icon} {it[0]}: <span style="color:{color}; font-weight:bold;">{it[2] if it[1]==1.0 else it[3] if it[1]==0.5 else it[-1]}</span></div>', unsafe_allow_html=True)
