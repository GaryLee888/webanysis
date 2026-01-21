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
st.set_page_config(page_title="台股 A.B.C 決策系統", layout="wide")

# --- CSS 修飾 ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #1c2833; color: #fcf3cf; }
    [data-testid="stSidebar"] .stTextInput label { display: none; }
    [data-testid="stSidebar"] .stTextInput, [data-testid="stSidebar"] .stButton {
        width: 130px !important; margin-left: 45px !important; margin-right: auto !important; padding: 0 !important;
    }
    [data-testid="stSidebar"] input {
        height: 35px !important; width: 130px !important; font-size: 1.3rem !important;
        text-align: center !important; border-radius: 2px !important; margin-bottom: 4px !important;
    }
    [data-testid="stSidebar"] button {
        background-color: #e67e22 !important; color: white !important; font-weight: bold !important;
        width: 130px !important; height: 35px !important; display: block !important;
        border-radius: 2px !important; border: none !important; line-height: 35px !important;
        padding: 0 !important; margin-top: 0px !important; margin-bottom: 8px !important; text-align: center !important;
    }
    .sidebar-title { color: #fcf3cf; text-align: center; width: 130px; margin-left: 45px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. 字體與價格修正設定 ---
def set_mpl_chinese():
    # 嘗試多種字體以相容不同系統 (Windows/Linux/Streamlit Cloud)
    fonts = ['Microsoft JhengHei', 'SimSun', 'Noto Sans CJK JP', 'DejaVu Sans']
    for f in fonts:
        plt.rcParams['font.sans-serif'] = [f] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 

set_mpl_chinese()

def round_stock_price(price):
    """依照台股升降單位規則修約 (2026 最新)"""
    if price < 10: return np.round(price, 2)
    elif price < 50: return np.round(price * 20) / 20
    elif price < 100: return np.round(price, 1)
    elif price < 500: return np.round(price * 2) / 2
    elif price < 1000: return np.round(price, 0)
    else: return np.round(price / 5) * 5

# --- 2. 核心分析引擎 ---
class StockEngine:
    def __init__(self):
        self.fm_api = DataLoader()
        self.special_mapping = {"貝爾威勒": "7861", "能率亞洲": "7777", "力旺": "3529", "朋程": "8255"}

    def fetch_data(self, sid):
        """強化版數據抓取：支援上市(.TW)與上櫃(.TWO)自動偵測"""
        for suffix in [".TW", ".TWO"]:
            try:
                ticker_str = f"{sid}{suffix}"
                df = yf.download(ticker_str, period="1y", progress=False, threads=False)
                
                # 處理 yfinance 0.2.x 版以後可能出現的 MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if df is not None and not df.empty and len(df) > 20:
                    return df, ticker_str
            except:
                continue
        return None, None

    def calculate_indicators(self, df):
        df = df.copy()
        win = 20
        # 均線
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(win).mean()
        
        # 布林通道
        std = df['Close'].rolling(win).std()
        df['BB_up'] = df['MA20'] + (std * 2)
        df['BB_low'] = df['MA20'] - (std * 2)
        df['BB_width'] = (df['BB_up'] - df['BB_low']) / df['MA20'].replace(0, 1)
        
        # ATR 與 KD
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        low_9 = df['Low'].rolling(9).min()
        high_9 = df['High'].rolling(9).max()
        df['K'] = ((df['Close'] - low_9) / (high_9 - low_9).replace(0, 1) * 100).ewm(com=2).mean()
        df['D'] = df['K'].ewm(com=2).mean()
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['MACD_hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
        
        # 其他指標
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['MFI'] = 50 + (df['Close'].diff().rolling(14).mean() * 10)
        df['VMA20'] = df['Volume'].rolling(win).mean()
        df['BIAS20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
        df['BIAS5'] = (df['Close'] - df['MA5']) / df['MA5'] * 100
        df['Vol_Ratio'] = (df['Volume'] / df['VMA20'].shift(1)).fillna(1)
        df['ROC'] = df['Close'].pct_change(12) * 100
        df['SR_Rank'] = (df['Close'] - df['Close'].rolling(60).min()) / (df['Close'].rolling(60).max() - df['Close'].rolling(60).min()).replace(0, 1)
        
        return df.ffill().bfill()

    def fetch_chips(self, sid):
        try:
            # 移除後置碼以便 FinMind 查詢
            clean_sid = sid.split('.')[0]
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
            df_chips = self.fm_api.taiwan_stock_institutional_investors(stock_id=clean_sid, start_date=start_date)
            if df_chips.empty: return None
            summary = df_chips.groupby(['date', 'name'])['buy'].sum().unstack().fillna(0)
            return {
                "it": summary['投信'].tail(3).sum() > 0 if '投信' in summary else False,
                "fg": summary['外資'].tail(5).sum() > 0 if '外資' in summary else False,
                "inst": summary.tail(3).sum(axis=1).sum() > 0
            }
        except: return None

# --- UI 介面 ---
st.title("🚀 台股 A.B.C 決策分析系統")

with st.sidebar:
    st.markdown("<h3 class='sidebar-title'>代碼/名稱</h3>", unsafe_allow_html=True)
    analyze_btn = st.button("啟動分析")
    
    default_vals = ["2330", "2317", "2454", "6223", "2603", "2881", "3529", "", "", ""]
    queries = []
    for i in range(10):
        val = st.text_input("", value=default_vals[i], key=f"in_{i}")
        if val.strip(): queries.append(val.strip())

engine = StockEngine()

if analyze_btn and queries:
    tabs = st.tabs([f" {q} " for q in queries])
    for i, query in enumerate(queries):
        with tabs[i]:
            # 代碼轉換邏輯
            sid = engine.special_mapping.get(query, query)
            stock_name = query
            if not sid.isdigit():
                found = False
                for code, info in twstock.codes.items():
                    if query == info.name:
                        sid, stock_name, found = code, info.name, True
                        break
                if not found:
                    st.error(f"找不到符合名稱: {query}"); continue
            elif sid in twstock.codes:
                stock_name = twstock.codes[sid].name

            # 抓取與計算
            df_raw, ticker = engine.fetch_data(sid)
            if df_raw is None:
                st.error(f"無法取得 {sid} 的行情數據，請檢查代碼是否正確。"); continue

            df = engine.calculate_indicators(df_raw)
            chip_data = engine.fetch_chips(sid)
            curr = df.iloc[-1]
            
            # --- A.B.C 關鍵點計算 ---
            # A 點 (Entry): 若股價在布林中軸上，取中軸與上軌平均；若噴出則取回測點
            raw_entry = (curr['MA20'] + curr['BB_up']) / 2 if curr['Close'] <= curr['BB_up'] else curr['Close'] * 0.97
            entry_p = round_stock_price(float(raw_entry))
            # B 點 (Stop Loss): ATR 停損法
            sl_p = round_stock_price(entry_p - (float(curr['ATR']) * 2.5))
            # C 點 (Take Profit): 風報比 1:2
            tp_p = round_stock_price(entry_p + (entry_p - sl_p) * 2.0)

            # --- 指標清單與評分 ---
            indicator_list = [
                ("均線趨勢", (1.0 if curr['Close'] > curr['MA20'] else 0.0), "多頭", "空頭"),
                ("KD動能", (1.0 if curr['K'] > curr['D'] else 0.0), "黃金交叉", "死亡交叉"),
                ("MACD柱狀", (1.0 if curr['MACD_hist'] > 0 else 0.0), "紅柱增長", "綠柱縮短"),
                ("RSI強弱", (1.0 if curr['RSI'] > 50 else 0.0), "強勢區", "弱勢區"),
                ("布林位階", (1.0 if curr['Close'] > curr['MA20'] else 0.0), "中軸上方", "中軸下方"),
                ("乖離率", (1.0 if abs(curr['BIAS20']) < 10 else 0.0), "安全區", "乖離過大"),
                ("資金流向", (1.0 if curr['MFI'] > 50 else 0.0), "流入", "流出"),
                ("[籌碼] 法人", (1.0 if chip_data and chip_data['inst'] else 0.0), "有買盤", "無量"),
                ("[籌碼] 投信", (1.0 if chip_data and chip_data['it'] else 0.0), "佈局中", "無動作")
            ]
            score = int((sum([it[1] for it in indicator_list]) / len(indicator_list)) * 100)
            
            # --- 儀表板顯示 ---
            st.markdown(f"### 📊 {stock_name} ({sid}) 診斷：{score} 分")
            c1, c2, c3, c4 = st.columns(4)
            
            def metric_box(label, val, color, is_price=True):
                fmt = ".2f" if val < 100 else ".1f" if val < 500 else ".0f"
                display_val = f"{val:{fmt}}" if is_price else val
                return f"<div style='border-left:5px solid {color}; padding-left:10px;'><p style='color:gray;margin:0;'>{label}</p><h2 style='margin:0;color:{color};'>{display_val}</h2></div>"

            c1.markdown(metric_box("目前股價", float(curr['Close']), "#2c3e50"), unsafe_allow_html=True)
            c2.markdown(metric_box("A. 建議買點", entry_p, "#2980b9"), unsafe_allow_html=True)
            c3.markdown(metric_box("B. 止損位", sl_p, "#27ae60"), unsafe_allow_html=True)
            c4.markdown(metric_box("C. 獲利目標", tp_p, "#e74c3c"), unsafe_allow_html=True)

            # --- 技術圖表 ---
            st.markdown("---")
            fig, ax = plt.subplots(figsize=(12, 5))
            df_p = df.tail(60)
            ax.plot(df_p.index, df_p['Close'], color='black', lw=2, label='收盤價')
            ax.plot(df_p.index, df_p['MA20'], color='orange', ls='--', alpha=0.7, label='20MA')
            ax.fill_between(df_p.index, df_p['BB_up'], df_p['BB_low'], color='gray', alpha=0.1, label='布林通道')
            
            # 標註 ABC 點
            ax.axhline(entry_p, color='#2980b9', ls=':', alpha=0.8)
            ax.axhline(sl_p, color='#27ae60', ls=':', alpha=0.8)
            ax.axhline(tp_p, color='#e74c3c', ls=':', alpha=0.8)
            
            ax.set_title(f"{stock_name} 近期走勢與策略參考點")
            ax.legend(loc='upper left')
            st.pyplot(fig)

            # 詳細指標
            st.markdown("#### 詳細指標狀態")
            ind_c1, ind_c2, ind_c3 = st.columns(3)
            for idx, it in enumerate(indicator_list):
                target_col = [ind_c1, ind_c2, ind_c3][idx % 3]
                icon = "✅" if it[1] == 1.0 else "⚪"
                target_col.write(f"{icon} {it[0]}: **{it[2] if it[1]==1.0 else it[3]}**")
