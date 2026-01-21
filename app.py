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
st.set_page_config(page_title="🚀 精準台股決策系統", layout="wide")

# --- CSS 修飾 ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #1c2833; color: #fcf3cf; }
    [data-testid="stSidebar"] .stTextInput label { display: none; }
    [data-testid="stSidebar"] .stTextInput, [data-testid="stSidebar"] .stButton {
        width: 130px !important; margin-left: 45px !important; margin-right: auto !important; padding: 0 !important;
    }
    [data-testid="stSidebar"] input {
        height: 35px !important; width: 130px !important; font-size: 1.1rem !important;
        text-align: center !important; border-radius: 4px !important; margin-bottom: 4px !important;
    }
    [data-testid="stSidebar"] button {
        background-color: #e67e22 !important; color: white !important; font-weight: bold !important;
        width: 130px !important; height: 40px !important; display: block !important;
        border-radius: 4px !important; border: none !important; margin-top: 10px !important;
    }
    .sidebar-title { color: #fcf3cf; text-align: center; width: 130px; margin-left: 45px; margin-bottom: 15px; font-size: 1.2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. 環境設定與效能優化函數 ---
def set_mpl_chinese():
    # 嘗試多種中文字體以確保在不同系統都能正常顯示
    fonts = ['msjh.ttc', 'msjh.ttf', 'NotoSansCJK-Regular.ttc']
    font_found = False
    for f_path in fonts:
        if os.path.exists(f_path):
            fe = fm.FontEntry(fname=f_path, name='CustomFont')
            fm.fontManager.ttflist.insert(0, fe)
            plt.rcParams['font.sans-serif'] = ['CustomFont']
            font_found = True
            break
    if not font_found:
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False 

set_mpl_chinese()

def round_stock_price(price):
    """依照台股最新升降單位規則修約"""
    if price < 10: return np.round(price, 2)
    elif price < 50: return np.round(price * 20) / 20
    elif price < 100: return np.round(price, 1)
    elif price < 500: return np.round(price * 2) / 2
    elif price < 1000: return np.round(price, 0)
    else: return np.round(price / 5) * 5

# --- 2. 數據獲取與分析引擎 (含快取) ---
class StockEngine:
    def __init__(self):
        self.fm_api = DataLoader()
        self.special_mapping = {"貝爾威勒": "7861", "能率亞洲": "7777", "力旺": "3529", "朋程": "8255"}

    @st.cache_data(ttl=3600) # 緩存1小時，減少API呼叫次數
    def get_stock_info(_self, query):
        """完全比對名稱抓取代碼"""
        sid = _self.special_mapping.get(query, query)
        stock_name = query
        
        if not sid.isdigit():
            found = False
            for code, info in twstock.codes.items():
                if query == info.name: # 修改點：使用 == 進行完全比對
                    sid = code
                    stock_name = info.name
                    found = True
                    break
            if not found: return None, None
        elif sid in twstock.codes:
            stock_name = twstock.codes[sid].name
        return sid, stock_name

    @st.cache_data(ttl=1800)
    def fetch_data(_self, sid):
        """資料抓取備援機制：Yahoo -> FinMind"""
        # 嘗試 Yahoo Finance
        for suffix in [".TW", ".TWO"]:
            try:
                df = yf.download(f"{sid}{suffix}", period="1y", progress=False)
                if df is not None and not df.empty and len(df) > 20:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    return df, f"{sid}{suffix}"
            except: continue
        
        # 備援：嘗試 FinMind
        try:
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
            df_fm = _self.fm_api.taiwan_stock_daily(stock_id=sid, start_date=start_date)
            if not df_fm.empty:
                df_fm = df_fm.rename(columns={'date': 'Date', 'open': 'Open', 'max': 'High', 'min': 'Low', 'close': 'Close', 'Trading_Volume': 'Volume'})
                df_fm['Date'] = pd.to_datetime(df_fm['Date'])
                df_fm.set_index('Date', inplace=True)
                return df_fm, sid
        except: pass
        
        return None, None

    def calculate_indicators(self, df):
        df = df.copy()
        # 原有指標計算邏輯保留並優化
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df['MA20'] = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
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
        df['VMA20'] = df['Volume'].rolling(20).mean()
        df['BIAS20'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
        df['Vol_Ratio'] = (df['Volume'] / df['VMA20'].shift(1)).fillna(1)
        df['ROC'] = df['Close'].pct_change(12) * 100
        df['SR_Rank'] = (df['Close'] - df['Close'].rolling(60).min()) / (df['Close'].rolling(60).max() - df['Close'].rolling(60).min()).replace(0, 1)
        
        return df.fillna(method='ffill').fillna(method='bfill')

    @st.cache_data(ttl=3600)
    def fetch_chips(_self, sid):
        try:
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=45)).strftime('%Y-%m-%d')
            df_chips = _self.fm_api.taiwan_stock_institutional_investors(stock_id=sid, start_date=start_date)
            if df_chips.empty: return None
            summary = df_chips.groupby(['date', 'name'])['buy'].sum().unstack().fillna(0)
            return {
                "it": summary['投信'].tail(3).sum() > 0 if '投信' in summary else False,
                "fg": summary['外資'].tail(5).sum() > 0 if '外資' in summary else False,
                "inst": summary.tail(3).sum(axis=1).sum() > 0
            }
        except: return None

# --- UI 介面 ---
st.title("🚀 台股全能分析與決策系統")

with st.sidebar:
    st.markdown("<h3 class='sidebar-title'>股票代碼/名稱</h3>", unsafe_allow_html=True)
    analyze_btn = st.button("啟動分析")
    
    default_vals = ["2330", "2317", "2454", "6223", "2603", "2881", "貝爾威勒", "", "", ""]
    queries = []
    for i in range(10):
        val = st.text_input("", value=default_vals[i], key=f"in_{i}")
        if val.strip(): queries.append(val.strip())

engine = StockEngine()

if analyze_btn and queries:
    tabs = st.tabs([f" {q} " for q in queries])
    for i, query in enumerate(queries):
        with tabs[i]:
            sid, stock_name = engine.get_stock_info(query)
            if not sid:
                st.error(f"❌ 無法識別股票: {query} (請確認名稱完全符合)")
                continue

            df_raw, ticker = engine.fetch_data(sid)
            if df_raw is None:
                st.error(f"⚠️ 資料抓取失敗: {sid}")
                continue

            df = engine.calculate_indicators(df_raw)
            chip_data = engine.fetch_chips(sid)
            curr = df.iloc[-1]
            
            # --- 買賣策略 ---
            raw_entry = (curr['MA20'] + curr['BB_up']) / 2 if curr['Close'] <= curr['BB_up'] else curr['Close'] * 0.98
            entry_p = round_stock_price(float(raw_entry))
            sl_p = round_stock_price(entry_p - (float(curr['ATR']) * 2.2))
            tp_p = round_stock_price(entry_p + (entry_p - sl_p) * 2.0)

            # 得分邏輯
            indicator_list = [
                ("均線趨勢", (1.0 if curr['Close'] > curr['MA20'] else 0.0), "多頭", "空頭"),
                ("軌道位階", (1.0 if curr['Close'] > curr['BB_up'] else 0.5 if curr['Close'] > curr['MA20'] else 0.0), "上位", "中位", "下位"),
                ("KD動能", (1.0 if curr['K'] > curr['D'] else 0.0), "向上", "向下"),
                ("MACD趨勢", (1.0 if curr['MACD_hist'] > 0 else 0.0), "紅柱", "綠柱"),
                ("RSI強弱", (1.0 if curr['RSI'] > 50 else 0.0), "強勢", "弱勢"),
                ("籌碼投信", (1.0 if chip_data and chip_data['it'] else 0.0), "佈局中", "無動作"),
                ("籌碼外資", (1.0 if chip_data and chip_data['fg'] else 0.0), "加碼中", "調節中")
                # 可依此類推增加至原有的 25 項
            ]
            score = int((sum([it[1] for it in indicator_list]) / len(indicator_list)) * 100)
            
            # 顯示結果
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.metric(label=f"{stock_name} ({sid}) 診斷得分", value=f"{score} 分")
            with col_b:
                rating = "🚀 強勢進攻" if score >= 70 else "⚖️ 穩健持平" if score >= 50 else "⚠️ 保守觀望"
                st.markdown(f"### 評等：{rating}")

            # --- 數據卡片 ---
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            for col, (label, val, color) in zip([c1, c2, c3, c4], [("現價", curr['Close'], "#2c3e50"), ("買點", entry_p, "#2980b9"), ("止損", sl_p, "green"), ("目標", tp_p, "red")]):
                col.markdown(f"<div style='background:#f8f9f9; padding:15px; border-radius:10px; border-left:5px solid {color}'>"
                             f"<p style='margin:0; color:gray;'>{label}</p>"
                             f"<h2 style='margin:0; color:{color};'>{val:.2f}</h2></div>", unsafe_allow_html=True)

            # --- 強化版繪圖 ---
            st.markdown("### 📈 技術走勢與策略參考")
            fig, ax = plt.subplots(figsize=(12, 6))
            df_p = df.tail(60)
            
            # 布林通道陰影區
            ax.fill_between(df_p.index, df_p['BB_up'], df_p['BB_low'], color='gray', alpha=0.1, label='布林通道')
            ax.plot(df_p.index, df_p['MA20'], color='orange', lw=1, ls='--', label='20MA')
            
            # 收盤價曲線 (使用階梯色或動態視覺)
            ax.plot(df_p.index, df_p['Close'], color='#1c2833', lw=2.5, label='收盤價')
            
            # 策略線
            ax.axhline(entry_p, color='#2980b9', ls='-', lw=1.5, alpha=0.8, label='策略買點')
            ax.axhline(sl_p, color='green', ls='--', lw=1.2, alpha=0.6, label='停損參考')
            ax.axhline(tp_p, color='red', ls='--', lw=1.2, alpha=0.6, label='目標參考')
            
            ax.set_title(f"{stock_name} ({sid}) 60日走勢圖", fontsize=14)
            ax.legend(loc='upper left', frameon=True)
            ax.grid(axis='y', alpha=0.3)
            
            st.pyplot(fig)

            # 詳細診斷清單
            with st.expander("🔍 查看詳細技術指標診斷"):
                ind_c1, ind_c2 = st.columns(2)
                for idx, it in enumerate(indicator_list):
                    col = ind_c1 if idx % 2 == 0 else ind_c2
                    icon = "🟢" if it[1] == 1.0 else "🟡" if it[1] == 0.5 else "🔴"
                    col.write(f"{icon} **{it[0]}**: {it[2] if it[1] == 1.0 else it[3]}")
