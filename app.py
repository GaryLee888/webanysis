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

# --- 1. 頁面與風格配置 ---
st.set_page_config(page_title="PRO-Quant 專業決策系統", layout="wide")

# 注入高級感 CSS
st.markdown("""
    <style>
    /* 全域背景與字體 */
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* 側邊欄美化 */
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .sidebar-title { color: #58a6ff; font-weight: 800; font-size: 1.2rem; text-align: center; margin-bottom: 20px; }
    
    /* 專業卡片設計 */
    .metric-card {
        background: #1c2128;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-label { color: #8b949e; font-size: 0.85rem; margin-bottom: 5px; }
    .metric-value { font-size: 1.8rem; font-weight: bold; font-family: 'Courier New', monospace; }
    
    /* 指標膠囊 */
    .indicator-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 2px;
    }
    .tag-bull { background-color: #238636; color: #ffffff; }
    .tag-bear { background-color: #da3633; color: #ffffff; }
    .tag-neutral { background-color: #6e7681; color: #ffffff; }

    /* Tabs 美化 */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 4px 4px 0 0;
        padding: 5px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 核心邏輯 (繼承並優化原本 Engine) ---
# [保留您原本的 round_stock_price 和 StockEngine 類別，此處省略以節省篇幅]

# --- 3. UI 呈現邏輯 ---
def display_analysis(query, engine):
    sid = engine.special_mapping.get(query, query)
    # ... (此處保留原有的數據獲取與計算邏輯) ...
    # 假設已獲取 df, score, rating, curr, entry_p, sl_p, tp_p, indicator_list
    
    # --- 頁面頂部：診斷分數 ---
    score_color = "#238636" if score >= 70 else "#d29922" if score >= 50 else "#da3633"
    st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; background: #1c2128; padding: 20px; border-radius: 10px; border-left: 5px solid {score_color};">
            <div>
                <h2 style="margin:0;">{stock_name} <span style="font-size:1rem; color:#8b949e;">({sid})</span></h2>
                <p style="margin:0; color:#8b949e;">{rating} | { "多空共鳴，適合順勢操作" if score >= 70 else "格局穩定，建議分批佈局" if score >= 50 else "訊號疲弱，建議保守觀望"}</p>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.9rem; color:#8b949e;">綜合評分</span><br>
                <span style="font-size: 3rem; font-weight: bold; color: {score_color};">{score}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.write(" ")

    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.subheader("🎯 關鍵價位")
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        
        # 使用卡片式呈現
        metrics = [
            ("目前現價", curr['Close'], "#e0e0e0", c1),
            ("建議進場", entry_p, "#58a6ff", c2),
            ("止損防線", sl_p, "#ff7b72", c3),
            ("獲利目標", tp_p, "#7ee787", c4)
        ]
        
        for label, val, color, col in metrics:
            col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color};">{val:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            col.write(" ")

        st.subheader("🔍 指標矩陣")
        # 改用更緊湊的佈局
        ind_col1, ind_col2 = st.columns(2)
        for idx, (name, val, pos, neg, *extra) in enumerate(indicator_list):
            target_col = ind_col1 if idx % 2 == 0 else ind_col2
            tag_class = "tag-bull" if val == 1.0 else "tag-neutral" if val == 0.5 else "tag-bear"
            text = pos if val == 1.0 else (extra[0] if val == 0.5 else neg)
            target_col.markdown(f"""
                <div style="margin-bottom:8px;">
                    <span style="color:#8b949e; font-size:0.85rem;">{name}</span><br>
                    <span class="indicator-tag {tag_class}">{text}</span>
                </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.subheader("📈 技術趨勢")
        # 圖表美化
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0e1117')
        ax.set_facecolor('#0e1117')
        
        df_p = df.tail(60)
        # 繪製布林通道
        ax.fill_between(df_p.index, df_p['BB_up'], df_p['BB_low'], color='#58a6ff', alpha=0.05)
        ax.plot(df_p.index, df_p['BB_up'], color='#30363d', lw=0.8, ls='--')
        ax.plot(df_p.index, df_p['BB_low'], color='#30363d', lw=0.8, ls='--')
        
        # 繪製收盤價與均線
        ax.plot(df_p.index, df_p['Close'], color='#e0e0e0', lw=2.5, label='Close')
        ax.plot(df_p.index, df_p['MA20'], color='#d29922', lw=1, alpha=0.8, label='MA20')
        
        # 標註價位線
        ax.axhline(entry_p, color='#58a6ff', ls='-', lw=1.5, alpha=0.6)
        ax.text(df_p.index[0], entry_p, f' 買點 {entry_p}', color='#58a6ff', va='bottom')
        
        # 座標軸美化
        ax.tick_params(colors='#8b949e', which='both')
        for spine in ax.spines.values(): spine.set_color('#30363d')
        ax.grid(color='#30363d', linestyle=':', alpha=0.5)
        
        st.pyplot(fig)
        
        # 新增量能分析區塊
        st.markdown("""
            <div style="background: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d;">
                <h4 style="margin-top:0; color:#8b949e;">📊 籌碼動能觀測</h4>
                <p style="font-size:0.9rem;">目前成交量較 20 日均量變化：<b style="color:#58a6ff;">{:.2f}%</b></p>
            </div>
        """.format((curr['Vol_Ratio']-1)*100), unsafe_allow_html=True)

# --- 主程式進入點 ---
# ... (串聯 sidebar 點擊與上述 display_analysis 函數)
