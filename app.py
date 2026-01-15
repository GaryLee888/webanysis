import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import twstock
import warnings
from FinMind.data import DataLoader

# 隱藏警告
warnings.filterwarnings("ignore")

# 頁面設定
st.set_page_config(page_title="台股全方位決策系統", layout="wide")

# --- 解決中文字體方塊問題 ---
def set_mpl_chinese():
    # 嘗試載入 Linux 伺服器常見的開源中文字體
    font_paths = fm.findSystemFonts()
    target_fonts = ['DejaVu Sans', 'Noto Sans CJK JP', 'Noto Sans TC', 'Arial Unicode MS']
    
    # 強制設定語系防止亂碼
    plt.rcParams['font.sans-serif'] = target_fonts + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 

set_mpl_chinese()

class StockEngine:
    def __init__(self):
        self.fm_api = DataLoader()
        self.special_mapping = {"貝爾威勒": "7861", "能率亞洲": "7777", "力旺": "3529", "朋程": "8255"}

    def fetch_data(self, sid):
        for suffix in [".TWO", ".TW"]:
            try:
                df = yf.download(f"{sid}{suffix}", period="1y", progress=False)
                if df is not None and not df.empty and len(df) > 5:
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
            it_val = summary['投信'].tail(3).sum() > 0 if '投信' in summary else False
            fg_val = summary['外資'].tail(5).sum() > 0 if '外資' in summary else False
            all_val = summary.tail(3).sum(axis=1).sum() > 0
            return {"it": it_val, "fg": fg_val, "inst": all_val}
        except: return None

# --- UI ---
st.title("🚀 台股全方位決策系統 (Mobile Web)")

with st.sidebar:
    st.header("清單設定")
    # 預設清單完全移植自原始程式
    default_vals = ["2330", "2317", "2454", "6223", "2603", "2881", "貝爾威勒", "", "", ""]
    queries = []
    for i in range(10):
        q = st.text_input(f"{i+1}:", value=default_vals[i], key=f"in_{i}")
        if q.strip(): queries.append(q.strip())
    analyze_btn = st.button("啟動分析", type="primary")

engine = StockEngine()

if analyze_btn:
    if not queries:
        st.warning("請至少輸入一個代碼")
    else:
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

                df_raw, ticker_str = engine.fetch_data(sid)
                if df_raw is None or len(df_raw) < 20:
                    st.error(f"無法抓取 {stock_name}({sid}) 數據")
                    continue

                df = engine.calculate_indicators(df_raw)
                chip_data = engine.fetch_chips(sid)
                curr = df.iloc[-1]
                prev = df.iloc[-2]
                
                # 策略建議價格
                entry_p = float((curr['MA20'] + curr['BB_up']) / 2 if curr['Close'] <= curr['BB_up'] else curr['Close'] * 0.98)
                sl_p = entry_p - (float(curr['ATR']) * 2.2)
                tp_p = entry_p + (entry_p - sl_p) * 2.0

                # 頂部數據卡片
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("現價", f"{float(curr['Close']):.2f}")
                c2.metric("建議買點", f"{entry_p:.2f}")
                c3.metric("止損位", f"{sl_p:.2f}")
                c4.metric("獲利目標", f"{tp_p:.2f}")

                # 圖表（移除中文字體依賴，改用英文 Label 以保證不亂碼，或使用系統支援字體）
                fig, ax = plt.subplots(figsize=(10, 5))
                df_p = df.tail(65)
                ax.plot(df_p.index, df_p['BB_up'], color='#e74c3c', ls='--', alpha=0.3)
                ax.plot(df_p.index, df_p['BB_low'], color='#27ae60', ls='--', alpha=0.3)
                ax.fill_between(df_p.index, df_p['BB_up'], df_p['BB_low'], color='#ecf0f1', alpha=0.2)
                ax.plot(df_p.index, df_p['Close'], color='#2c3e50', lw=2)
                ax.axhline(entry_p, color='#2980b9', ls='-', label='Entry')
                ax.axhline(sl_p, color='#c0392b', ls='--', label='SL')
                ax.axhline(tp_p, color='#27ae60', ls='--', label='TP')
                ax.set_title(f"Analysis: {stock_name} ({sid})")
                st.pyplot(fig)

                # --- 完全移植 25 項指標 ---
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
                    ("能量潮 (OBV)", (1.0 if curr['OBV'] > df['OBV'].mean() else 0.0), "集中", "渙散"),
                    ("資金流向 (MFI)", (1.0 if curr['MFI'] > 50 else 0.0), "流入", "流出"),
                    ("成交均量", (1.0 if curr['Volume'] > curr['VMA20'] else 0.0), "量增", "量縮"),
                    ("多空勁道", (1.0 if curr['Close'] > curr['MA5'] else 0.0), "強勁", "偏弱"),
                    ("乖離動能", (1.0 if curr['BIAS5'] > curr['BIAS20'] else 0.0), "轉強", "趨緩"),
                    ("支撐位階", (1.0 if curr['Close'] > curr['MA20'] else 0.0), "站穩", "破線"),
                    ("多空量比", (1.0 if curr['Vol_Ratio'] > 1 else 0.0), "買盤強", "賣壓大"),
                    ("價格變動 (ROC)", (1.0 if curr['ROC'] > 0 else 0.0), "正向", "負向"),
                    ("歷史位階", (1.0 if curr['SR_Rank'] > 0.5 else 0.0), "健康", "低迷"),
                    ("[籌] 投信連買", (1.0 if chip_data and chip_data['it'] else 0.0), "佈局中", "無動作"),
                    ("[籌] 外資波段", (1.0 if chip_data and chip_data['fg'] else 0.0), "加碼中", "調節中"),
                    ("[籌] 法人集結", (1.0 if chip_data and chip_data['inst'] else 0.0), "共識買", "分散"),
                    ("[籌] 攻擊量能", (1.0 if curr['Volume'] > curr['VMA20'] * 1.3 else 0.0), "爆量", "量縮"),
                    ("[籌] 資金匯集", (1.0 if curr['OBV'] > df['OBV'].tail(5).mean() else 0.0), "匯入", "流出"),
                    ("均線支撐 (MA10)", (1.0 if curr['Close'] > curr['MA10'] else 0.0), "支撐強", "跌破")
                ]

                total_pts = sum([it[1] for it in indicator_list])
                score = int((total_pts / 25) * 100)

                st.subheader(f"指標綜合診斷 ({score} 分)")
                
                # 顯示評級
                if score >= 70: st.success("🚀 強勢標的")
                elif score >= 50: st.warning("⚖️ 穩健標的")
                else: st.error("⚠️ 觀望標的")

                # 分兩欄顯示 25 項指標
                idx_cols = st.columns(2)
                for j, item in enumerate(indicator_list):
                    col = idx_cols[0] if j < 13 else idx_cols[1]
                    label = item[3] if item[1] == 1.0 else (item[4] if item[1] == 0.5 else item[-1])
                    icon = "🟢" if item[1] == 1.0 else "🟠" if item[1] == 0.5 else "🔴"
                    col.write(f"{icon} {item[0]}: **{label}**")
