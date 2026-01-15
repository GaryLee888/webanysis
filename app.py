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
st.set_page_config(page_title="台股全方位決策系統", layout="wide")

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
    # 台股常見跳動單位簡化處理：四捨五入至 0.05 
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
st.title("🚀 台股全方位決策系統")

with st.sidebar:
    st.header("🔍 股票清單")
    default_vals = ["2330", "2317", "2454", "6223", "2603", "2881", "貝爾威勒", "", "", ""]
    queries = []
    
    # 輸入框併排調整：每列兩個框，寬度縮小
    for i in range(0, 10, 2):
        sc1, sc2 = st.columns(2)
        with sc1:
            v1 = st.text_input(f"{i+1}:", value=default_vals[i], key=f"in_{i}")
            if v1: queries.append(v1)
        with sc2:
            v2 = st.text_input(f"{i+2}:", value=default_vals[i+1], key=f"in_{i+1}")
            if v2: queries.append(v2)
            
    analyze_btn = st.button("啟動分析", type="primary", use_container_width=True)

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
            
            # 價格校準 0.05
            entry_raw = (curr['MA20'] + curr['BB_up']) / 2 if curr['Close'] <= curr['BB_up'] else curr['Close'] * 0.98
            entry_p = round_stock_price(entry_raw)
            sl_p = round_stock_price(entry_p - (float(curr['ATR']) * 2.2))
            tp_p = round_stock_price(entry_p + (entry_p - sl_p) * 2.0)

            # --- 計算指標清單 (為了先算出總分) ---
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
                ("量價配合", (1.0 if curr['Close'] >= prev['Close'] else 0.0), "穩健", "背離"),
                ("能量潮", (1.0 if curr['OBV'] > df['OBV'].mean() else 0.0), "集中", "渙散"),
                ("資金流向", (1.0 if curr['MFI'] > 50 else 0.0), "流入", "流出"),
                ("成交均量", (1.0 if curr['Volume'] > curr['VMA20'] else 0.0), "量增", "量縮"),
                ("多空勁道", (1.0 if curr['Close'] > curr['MA5'] else 0.0), "強勁", "偏弱"),
                ("乖離動能", (1.0 if curr['BIAS5'] > curr['BIAS20'] else 0.0), "轉強", "趨緩"),
                ("支撐位階", (1.0 if curr['Close'] > curr['MA20'] else 0.0), "站穩", "破線"),
                ("多空量比", (1.0 if curr['Vol_Ratio'] > 1 else 0.0), "買盤強", "賣壓大"),
                ("價格變動", (1.0 if curr['ROC'] > 0 else 0.0), "正向", "負向"),
                ("歷史位階", (1.0 if curr['SR_Rank'] > 0.5 else 0.0), "健康", "低迷"),
                ("[籌] 投信連買", (1.0 if chip_data and chip_data['it'] else 0.0), "佈局中", "無動作"),
                ("[籌] 外資波段", (1.0 if chip_data and chip_data['fg'] else 0.0), "加碼中", "調節中"),
                ("[籌] 法人集結", (1.0 if chip_data and chip_data['inst'] else 0.0), "共識買", "分散"),
                ("[籌] 攻擊量能", (1.0 if curr['Volume'] > curr['VMA20'] * 1.3 else 0.0), "爆量", "量縮"),
                ("[籌] 資金匯集", (1.0 if curr['OBV'] > df['OBV'].tail(5).mean() else 0.0), "匯入", "流出"),
                ("均線支撐", (1.0 if curr['Close'] > curr['MA10'] else 0.0), "強勁", "跌破")
            ]
            score = int((sum([it[1] for it in indicator_list]) / 25) * 100)

            # --- A. [新版面] 得分結果移到上方 ---
            st.subheader(f"📊 綜合診斷：{score} 分")
            if score >= 70: st.success("🚀 強勢標的")
            elif score >= 50: st.warning("⚖️ 穩健標的")
            else: st.error("⚠️ 觀望標的")

            # --- B. 價格數據卡片 ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("現價", f"{float(curr['Close']):.2f}")
            c2.metric("建議買點", f"{entry_p:.2f}")
            c3.metric("止損位", f"{sl_p:.2f}")
            c4.metric("獲利目標", f"{tp_p:.2f}")

            # --- C. K線圖 ---
            fig, ax = plt.subplots(figsize=(10, 4.5))
            df_p = df.tail(65)
            ax.plot(df_p.index, df_p['BB_up'], color='#e74c3c', ls='--', alpha=0.3)
            ax.plot(df_p.index, df_p['BB_low'], color='#27ae60', ls='--', alpha=0.3)
            ax.plot(df_p.index, df_p['Close'], color='#2c3e50', lw=2)
            ax.axhline(entry_p, color='#2980b9', ls='-', label='Entry')
            ax.axhline(sl_p, color='#c0392b', ls='--', label='SL')
            ax.axhline(tp_p, color='#27ae60', ls='--', label='TP')
            ax.set_title(f"{stock_name} ({sid}) 分析圖")
            st.pyplot(fig)

            # --- D. 25 項指標 (維持在原位) ---
            st.markdown("### 詳細指標診斷")
            ind_c1, ind_c2 = st.columns(2)
            for idx, it in enumerate(indicator_list):
                col = ind_c1 if idx < 13 else ind_c2
                icon = "🟢" if it[1] == 1.0 else "🟠" if it[1] == 0.5 else "🔴"
                status = it[2] if it[1] == 1.0 else (it[3] if it[1] == 0.5 else it[-1])
                color = "green" if it[1] == 1.0 else "orange" if it[1] == 0.5 else "red"
                col.markdown(f"{icon} {it[0]}: <span style='color:{color}; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)
