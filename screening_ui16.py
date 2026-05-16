"""Streamlit UI for screening0515.py

執行方式:
    streamlit run screening_ui16.py
"""

# ── 頂層 import ───────────────────────────────────────────────────────────
import json
import tempfile
from pathlib import Path

import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from screening0515 import (
    run_screening,
    get_stock_history,          
    _calc_kd_series,            
    PRESETS,
    PASS_SCORE, LOOKBACK_DAYS, IT_MIN_BUY_DAYS, FI_MIN_BUY_DAYS,
    KD_LOOKBACK, KD_LOW_FROM, KD_HIGH_CAP_NOW,
    MIN_AVG_VOL_LOTS, ATR_MAX_PCT,
    CACHE_DIR,
)

# ── 自選股持久化 ───────────────────────────────────────────────────────────
WATCHLIST_FILE = "watchlist.json"

def load_watchlist() -> list:
    if Path(WATCHLIST_FILE).exists():
        with open(WATCHLIST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_watchlist(wl: list) -> None:
    with open(WATCHLIST_FILE, 'w', encoding='utf-8') as f:
        json.dump(wl, f, ensure_ascii=False, indent=2)

# ── 交易筆記持久化 (🔥 修正版：避免 Widget Cleanup 刪除機制) ───────────────────
NOTES_FILE = "notes.json"

def load_notes() -> dict:
    if Path(NOTES_FILE).exists():
        with open(NOTES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_note(sid: str, text: str) -> None:
    notes = load_notes()
    if text.strip():
        notes[sid] = text.strip()
    elif sid in notes:
        del notes[sid]
    with open(NOTES_FILE, 'w', encoding='utf-8') as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

def _note_on_change_cb(sid: str) -> None:
    """Widget 內容變動時直接將新文字寫入記憶體與 json 檔案中"""
    val = st.session_state.get(f"note_input_{sid}", "")
    st.session_state.trading_notes[sid] = val
    save_note(sid, val)

# ── [新增] 法人資料讀取器 (供技術圖表渲染副圖) ──────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def get_stock_institutional(stock_id: str) -> pd.DataFrame:
    """讀取最新的三大法人快取並過濾指定標的"""
    files = sorted(CACHE_DIR.glob('institutional_*.parquet'))
    if not files:
        return pd.DataFrame()
    try:
        df = pd.read_parquet(files[-1])
        df['stock_id'] = df['stock_id'].astype(str)
        df_stock = df[df['stock_id'] == str(stock_id)].copy()
        df_stock['date'] = pd.to_datetime(df_stock['date'])
        return df_stock
    except Exception:
        return pd.DataFrame()

# ── 大盤 RS 快取 ───────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _load_twii_cached() -> pd.DataFrame:
    try:
        data = yf.download("^TWII", period="2y", auto_adjust=True, progress=False, threads=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if not data.empty and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        return data
    except Exception:
        return pd.DataFrame()

# ── 頁面設定 ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="台股選股", page_icon="📊", layout="wide")
st.title("📊 台股選股工具")
st.caption("拖滑桿調參數 → 按開始選股 → 結果可下載匯入 嘉實 / 精誠。")

# ── Cache 新鮮度警示 & 雲端更新 ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def _get_cache_max_date():
    files = sorted(CACHE_DIR.glob('daily_*.parquet'))
    if not files:
        return None
    try:
        df = pd.read_parquet(files[-1], columns=['date'])
    except Exception:
        return None
    if df.empty:
        return None
    return pd.to_datetime(df['date']).max()

_cache_date = _get_cache_max_date()
if _cache_date is None:
    st.error("⚠ 找不到 daily 快取，請先執行 `python fetch_cache.py` 拉資料")
else:
    _today    = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None).normalize()
    _age      = (_today - _cache_date.normalize()).days
    _date_str = _cache_date.strftime('%Y-%m-%d')
    if _age == 0:
        st.success(f"📅 cache 最新日期: {_date_str}(今天)")
    elif _age <= 2:
        st.info(f"📅 cache 最新日期: {_date_str}({_age} 天前)")
    elif _age <= 7:
        st.warning(f"📅 cache 最新日期: {_date_str}({_age} 天前)— 建議更新快取")
    else:
        st.error(f"📅 cache 最新日期: {_date_str}({_age} 天前) ⚠ 過舊，請先更新")

    st.divider()
    st.subheader("🔄 資料更新")
    st.caption("雲端環境專用：點擊後從網路抓取最新台股資料。")

    if st.session_state.get('show_update_success'):
        st.toast("✅ 雲端資料檢查/更新完成！", icon="🎉")
        st.session_state.show_update_success = False

    if st.button("📥 抓取今日最新資料", type="secondary", use_container_width=True):
        st.toast("⏳ 系統已收到請求，開始比對資料...", icon="🤖") 
        with st.spinner("正在執行資料更新... (若轉圈超過 2 分鐘，可能是主機記憶體不足崩潰)"):
            try:
                import subprocess
                import sys
                result = subprocess.run([sys.executable, "fetch_cache.py"], capture_output=True, text=True, check=True)
                st.cache_data.clear()
                st.session_state.show_update_success = True
                st.rerun()
            except subprocess.CalledProcessError as e:
                st.error("❌ 雲端腳本執行失敗！")
                st.code(e.stderr)
            except Exception as e:
                st.error(f"❌ 發生未知的錯誤：{e}")

# ── Session state 初始化 ────────────────────────────────────────────────────
DEFAULTS = {
    'pass_score': PASS_SCORE, 'lookback_days': LOOKBACK_DAYS, 'it_min_buy_days': IT_MIN_BUY_DAYS,
    'fi_min_buy_days': FI_MIN_BUY_DAYS, 'kd_lookback': KD_LOOKBACK, 'kd_low_from': KD_LOW_FROM,
    'kd_high_cap_now': KD_HIGH_CAP_NOW, 'min_avg_vol_lots': MIN_AVG_VOL_LOTS, 'atr_max_pct': ATR_MAX_PCT,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

_state_defaults = {
    'result_df': None, 'result_files': {}, 'result_meta': None, 'target_sid': None, 'last_df_selection': [],
}
for k, v in _state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# 🔥 核心修正：將筆記載入獨立字典，不受 Widget 銷毀影響
if 'trading_notes' not in st.session_state:
    st.session_state.trading_notes = load_notes()

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()

# ── 全域股名對照表 ─────────────────────────────────────────────────────────
@st.cache_data
def get_ui_name_map() -> dict:
    files = sorted(CACHE_DIR.glob('info_*.parquet'))
    if not files:
        return {}
    try:
        df_info = pd.read_parquet(files[-1])
        df_info = df_info.drop_duplicates(subset='stock_id', keep='last')
        return df_info.set_index('stock_id')['stock_name'].astype(str).to_dict()
    except Exception:
        return {}

ui_name_map = get_ui_name_map()

def apply_preset(name: str) -> None:
    for k, v in DEFAULTS.items(): st.session_state[k] = v
    for k, v in PRESETS[name].items(): st.session_state[k] = v

def labeled(base: str, key: str) -> str:
    return f"{base} ●" if st.session_state.get(key) != DEFAULTS[key] else base

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("🎯 快速套用組合")
    st.button("預設",     on_click=apply_preset, args=('default',),   use_container_width=True)
    st.button("多頭寬鬆", on_click=apply_preset, args=('bull',),      use_container_width=True)
    st.button("空頭嚴格", on_click=apply_preset, args=('bear',),      use_container_width=True)
    st.button("KD 起漲",  on_click=apply_preset, args=('kd_start',),  use_container_width=True)

    st.divider()
    st.subheader("📌 核心 / KD")
    st.slider(labeled("過關門檻 PASS_SCORE", 'pass_score'), 1, 10, key='pass_score')
    st.slider(labeled("KD 觀察區間 (天)", 'kd_lookback'), 1, 30, key='kd_lookback')
    st.slider(labeled("KD 低檔啟動門檻", 'kd_low_from'), 10, 50, key='kd_low_from')
    st.slider(labeled("KD 今日上限", 'kd_high_cap_now'), 50, 95, key='kd_high_cap_now')

    st.divider()
    st.subheader("📌 預篩 / 法人")
    st.slider(labeled("20 日均量下限 (張)", 'min_avg_vol_lots'), 100, 2000, key='min_avg_vol_lots', step=50)
    st.slider(labeled("ATR% 上限", 'atr_max_pct'), 5.0, 25.0, key='atr_max_pct', step=0.5)
    st.slider(labeled("法人觀察天數", 'lookback_days'), 3, 10, key='lookback_days')
    st.slider(labeled("投信最少買超日", 'it_min_buy_days'), 1, 5, key='it_min_buy_days')
    st.slider(labeled("外資最少買超日", 'fi_min_buy_days'), 1, 5, key='fi_min_buy_days')

    st.divider()
    run_clicked = st.button("▶ 開始選股", type='primary', use_container_width=True)

    st.divider()
    st.subheader("⭐ 我的自選股")
    if not st.session_state.watchlist:
        st.caption("目前無自選股。")
    else:
        for w_sid in st.session_state.watchlist:
            c1, c2 = st.columns([3, 1])
            w_name = ui_name_map.get(str(w_sid), "")
            label  = f"📈 {w_sid} {w_name}".strip()
            if c1.button(label, key=f"view_{w_sid}", use_container_width=True):
                st.session_state.target_sid = w_sid
                st.rerun()
            if c2.button("❌", key=f"del_{w_sid}", help="移除"):
                st.session_state.watchlist.remove(w_sid)
                save_watchlist(st.session_state.watchlist)
                st.rerun()

# ── 大盤狀態橫幅 ───────────────────────────────────────────────────────────
def show_market_banner(meta: dict) -> None:
    if not meta: return
    base = meta.get('base_pass_score')
    eff  = meta.get('effective_pass_score')
    if not meta.get('market_data_ok'):
        st.warning(f"🟡 大盤資料抓取失敗，本次無 RS 計分 | 過關門檻 {base} → **{eff}**")
        return
    bullish, twii_now, twii_ma, change = meta.get('market_bullish', True), meta.get('twii_now'), meta.get('twii_ma'), meta.get('twii_lookback_change', 0.0)
    state_txt = "🟢 多頭(站上季線)" if bullish else "🔴 空頭(跌破季線)"
    thr_txt   = f"過關門檻 {base}" if base == eff else f"過關門檻 {base} → **{eff}**"
    msg = f"{state_txt}  |  {thr_txt}  |  TWII {twii_now:,.0f} / MA60 {twii_ma:,.0f} | 近 20 日 {change:+.2f}%"
    st.success(msg) if bullish else st.warning(msg)

if run_clicked:
    with st.spinner("選股中..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            df, file_paths, meta = run_screening(
                pass_score=st.session_state.pass_score, lookback_days=st.session_state.lookback_days,
                it_min_buy_days=st.session_state.it_min_buy_days, fi_min_buy_days=st.session_state.fi_min_buy_days,
                kd_lookback=st.session_state.kd_lookback, kd_low_from=st.session_state.kd_low_from,
                kd_high_cap_now=st.session_state.kd_high_cap_now, min_avg_vol_lots=st.session_state.min_avg_vol_lots,
                atr_max_pct=st.session_state.atr_max_pct, output_dir=Path(tmpdir),
            )
            files_bytes: dict = {}
            for ftype, fpath in file_paths.items():
                if Path(fpath).exists():
                    with open(fpath, 'rb') as fh: files_bytes[ftype] = (Path(fpath).name, fh.read())
        st.session_state.result_df = df
        st.session_state.result_files = files_bytes
        st.session_state.result_meta = meta
    st.success("✅ 完成，看下方結果")

df = st.session_state.result_df
files_bytes = st.session_state.result_files or {}
meta = st.session_state.result_meta

if meta: show_market_banner(meta)
st.divider()

col_list, col_chart = st.columns([0.45, 0.55])

with col_list:
    if df is None: st.info("👈 請先按左側「開始選股」產生最新清單。")
    elif len(df) == 0: st.error("❌ 沒有標的達標")
    else:
        st.subheader(f"📋 結果({len(df)} 檔)")
        fc1, fc2 = st.columns(2)
        with fc1:
            score_min, score_max = int(df['總分'].min()), int(df['總分'].max())
            score_threshold = st.slider("總分 ≥", score_min, score_max, score_min) if score_min != score_max else score_min
        with fc2:
            industries = sorted([i for i in df['產業'].dropna().unique() if i])
            selected_industries = st.multiselect("產業篩選", industries, default=industries)

        filtered = df[(df['總分'] >= score_threshold) & (df['產業'].isin(selected_industries))]
        event = st.dataframe(filtered, use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")

        st.divider()
        st.subheader("📥 下載匯入檔")
        dl_cols = st.columns(3)
        MIME_TYPES = {'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'dsl': 'application/octet-stream', 'xls': 'application/vnd.ms-excel'}
        LABELS = {'xlsx': '📥 Excel', 'dsl': '📥 嘉實', 'xls': '📥 精誠'}
        for i, ftype in enumerate(['xlsx', 'dsl', 'xls']):
            if ftype in files_bytes:
                fname, data = files_bytes[ftype]
                dl_cols[i].download_button(LABELS[ftype], data, fname, MIME_TYPES[ftype], use_container_width=True)
            else:
                dl_cols[i].button(LABELS[ftype] + " (無)", disabled=True, use_container_width=True)

        current_sel = event.selection.rows
        if current_sel != st.session_state.last_df_selection:
            st.session_state.last_df_selection = current_sel
            if current_sel: st.session_state.target_sid = str(filtered.iloc[current_sel[0]]['代號'])

# ── K 線圖區 ───────────────────────────────────────────────────────────────
with col_chart:
    if not st.session_state.target_sid:
        st.info("👈 請點擊左側列表或自選股，開始量化決策分析。")
    else:
        sid = str(st.session_state.target_sid)
        row_data = None
        if df is not None and len(df) > 0:
            if sid in df['代號'].astype(str).values:
                row_data = df[df['代號'].astype(str) == sid].iloc[0]

        sname = row_data['名稱'] if row_data is not None else ui_name_map.get(sid, "")
        tab1, tab2, tab3 = st.tabs(["📈 技術分析", "📊 籌碼/基本面", "📝 交易筆記"])
        hist_daily = get_stock_history(sid)

        with tab1:
            tcol1, tcol2, tcol3 = st.columns([2, 1.5, 1])
            tcol1.markdown(f"### {sid} {sname}")
            timeframe = tcol2.radio("週期", ["日K", "週K"], horizontal=True, label_visibility="collapsed")
            theme_choice = tcol3.radio("主題", ["深色", "淺色"], horizontal=True, label_visibility="collapsed")

            is_in_watchlist = sid in st.session_state.watchlist
            if tcol1.button("⭐ 移除自選" if is_in_watchlist else "⭐ 加入自選", key=f"add_{sid}"):
                if is_in_watchlist: st.session_state.watchlist.remove(sid)
                else: st.session_state.watchlist.append(sid)
                save_watchlist(st.session_state.watchlist)
                st.rerun()

            chart_theme, bg_color, grid_color = ('plotly_dark', '#111', '#333') if theme_choice == "深色" else ('plotly_white', 'white', '#EEE')

            if not hist_daily.empty:
                high_col = 'max' if 'max' in hist_daily.columns else 'high'
                low_col  = 'min' if 'min' in hist_daily.columns else 'low'
                vol_col  = 'Trading_Volume' if 'Trading_Volume' in hist_daily.columns else 'volume'

                if timeframe == "週K":
                    hist = hist_daily.set_index('date').resample('W-FRI').agg({'open': 'first', high_col: 'max', low_col: 'min', 'close': 'last', vol_col: 'sum'}).dropna().reset_index()
                    hist = hist.rename(columns={high_col: 'max', low_col: 'min'})
                else:
                    hist = hist_daily.rename(columns={high_col: 'max', low_col: 'min'}).copy()

                # ── [整合] 讀取並排列法人買賣超數據 ──
                inst_stock = get_stock_institutional(sid)
                if not inst_stock.empty:
                    inst_stock['net_lots'] = (inst_stock['buy'] - inst_stock['sell']) / 1000.0
                    inst_pivot = inst_stock.pivot_table(index='date', columns='name', values='net_lots', aggfunc='sum').reset_index()
                    for c in ['Foreign_Investor', 'Investment_Trust']:
                        if c not in inst_pivot.columns: inst_pivot[c] = 0.0
                    
                    if timeframe == "週K":
                        inst_pivot = inst_pivot.set_index('date').resample('W-FRI').sum().reset_index()
                    else:
                        inst_pivot['date'] = pd.to_datetime(inst_pivot['date'])
                    
                    hist['date'] = pd.to_datetime(hist['date'])
                    hist = hist.merge(inst_pivot, on='date', how='left').fillna(0)
                else:
                    hist['Foreign_Investor'], hist['Investment_Trust'] = 0.0, 0.0

                twii_all = _load_twii_cached()
                if not twii_all.empty:
                    m_data = twii_all[((twii_all.index >= hist['date'].min()) & (twii_all.index <= hist['date'].max()))]
                    if timeframe == "週K" and not m_data.empty: m_data = m_data.resample('W-FRI').agg({'Close': 'last'}).dropna()
                else: m_data = pd.DataFrame()

                if not m_data.empty and 'Close' in m_data.columns:
                    m_close = m_data['Close'].reindex(hist.set_index('date').index).ffill()
                    if len(m_close) > 0 and m_close.iloc[0] > 0: hist['Market_Norm'] = m_close.values / m_close.iloc[0] * 100

                hist['MA5'], hist['MA20'], hist['MA60'] = hist['close'].rolling(5).mean(), hist['close'].rolling(20).mean(), hist['close'].rolling(60).mean()
                k_list, d_list = _calc_kd_series(hist['max'], hist['min'], hist['close'])
                hist['K'] = [x if x is not None else float('nan') for x in k_list]
                hist['D'] = [x if x is not None else float('nan') for x in d_list]

                kd_low_thr = st.session_state.kd_low_from
                kd_cross = ((hist['K'] > hist['D']) & (hist['K'].shift(1) <= hist['D'].shift(1)) & (hist['K'] < kd_low_thr))
                signal_dates, signal_prices = hist[kd_cross]['date'], hist[kd_cross]['min'] * 0.95

                # 頂部卡片顯示
                latest_close = hist_daily['close'].iloc[-1]
                prev_close = hist_daily['close'].iloc[-2] if len(hist_daily) > 1 else latest_close
                change_val = latest_close - prev_close
                m1, m2, m3 = st.columns(3)
                m1.metric("日收盤價", f"{latest_close:.2f}", f"{change_val:+.2f} ({change_val/prev_close*100:+.2f}%)")
                m2.metric("日成交量", f"{hist_daily[vol_col].iloc[-1]/1000:,.0f} 張")
                m3.metric("選股總分", f"{row_data['總分']} 分" if row_data is not None else "未入選")

                # 🎨 大改版：由 3 層擴展為 4 層子圖 (技術/成交量/法人/KD)
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.42, 0.12, 0.18, 0.28])

                fig.add_trace(go.Candlestick(x=hist['date'], open=hist['open'], high=hist['max'], low=hist['min'], close=hist['close'], name='K線', increasing_fillcolor='red', increasing_line_color='red', decreasing_fillcolor='green', decreasing_line_color='green'), row=1, col=1)
                if 'Market_Norm' in hist.columns:
                    fig.add_trace(go.Scatter(x=hist['date'], y=hist['Market_Norm'] * (hist['close'].iloc[0] / 100), line=dict(color='rgba(150,150,150,0.5)', width=1, dash='dot'), name='大盤 RS'), row=1, col=1)

                for col_ma, color, lbl in [('MA5', 'orange', 'MA5'), ('MA20', 'purple', 'MA20'), ('MA60', 'green', 'MA60')]:
                    fig.add_trace(go.Scatter(x=hist['date'], y=hist[col_ma], line=dict(color=color, width=1.2), name=lbl), row=1, col=1)

                if not signal_dates.empty:
                    fig.add_trace(go.Scatter(x=signal_dates, y=signal_prices, mode='markers', marker=dict(symbol='triangle-up', color='lime', size=12, line=dict(width=1, color='darkgreen')), name='KD金叉訊號'), row=1, col=1)

                if len(hist) > 60:
                    fig.add_hline(y=hist['max'].iloc[-61:-1].max(), line_dash="dot", line_color="orange", annotation_text="60日壓", annotation_position="top left", row=1, col=1)
                
                # 新股防護：MA20 NaN 防摔
                ma20_last = hist['MA20'].iloc[-1]
                defense_y = hist['min'].tail(10).min() if pd.isna(ma20_last) else max(ma20_last * 0.98, hist['min'].tail(10).min())
                fig.add_hline(y=defense_y, line_dash="dash", line_color="red", annotation_text="🚨 防守", annotation_position="bottom right", row=1, col=1)

                if row_data is not None:
                    fig.add_annotation(x=hist['date'].iloc[-1], y=hist['min'].iloc[-1], text="🔥 訊號觸發", showarrow=True, arrowhead=1, arrowcolor="red", ay=30, row=1, col=1)

                # 第二層：成交量
                vol_colors = ['red' if c >= o else 'green' for c, o in zip(hist['close'], hist['open'])]
                fig.add_trace(go.Bar(x=hist['date'], y=hist[vol_col], marker_color=vol_colors, name='成交量'), row=2, col=1)

                # 🔥 第三層：[新增] 法人買賣超柱狀圖 (barmode='group' 並列顯示)
                fig.add_trace(go.Bar(x=hist['date'], y=hist['Investment_Trust'], marker_color='#FF4B4B', name='投信(張)'), row=3, col=1)
                fig.add_trace(go.Bar(x=hist['date'], y=hist['Foreign_Investor'], marker_color='#FACA44', name='外資(張)'), row=3, col=1)

                # 第四層：KD 訊號線
                fig.add_trace(go.Scatter(x=hist['date'], y=hist['K'], line=dict(color='blue', width=1.2), name='K'), row=4, col=1)
                fig.add_trace(go.Scatter(x=hist['date'], y=hist['D'], line=dict(color='orange', width=1.2), name='D'), row=4, col=1)
                fig.add_hline(y=80, line_dash="dash", line_color="gray", line_width=0.5, row=4, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color="gray", line_width=0.5, row=4, col=1)

                all_dates = pd.date_range(start=hist['date'].min(), end=hist['date'].max())
                missing_dates = [d.strftime("%Y-%m-%d") for d in all_dates if d not in hist['date'].dt.normalize().values]

                fig.update_layout(
                    template=chart_theme, paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                    height=850, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10),
                    hovermode='x unified', showlegend=False, barmode='group' # 啟用並列柱狀圖模式
                )
                # 修正：將 bgcolor 放入 rangeselector 的 dict() 內部
                fig.update_xaxes(
                    rangeselector=dict(
                        buttons=[
                            dict(count=30, label="1月", step="day", stepmode="backward"), 
                            dict(count=90, label="3月", step="day", stepmode="backward"), 
                            dict(count=180, label="半年", step="day", stepmode="backward"), 
                            dict(step="all", label="全部")
                        ],
                        bgcolor='#333' if theme_choice == "深色" else '#EEE'
                    ), 
                    row=1, col=1
                )

                config = {'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawrect', 'eraseshape'], 'displayModeBar': True, 'displaylogo': False, 'scrollZoom': True}
                st.plotly_chart(fig, use_container_width=True, config=config)
            else: st.warning("暫無歷史數據。")

        with tab2:
            st.subheader("📊 詳細數據分析")
            if row_data is not None:
                c1, c2 = st.columns(2)
                c1.write(f"**💎 籌碼數據**\n- 投信淨額: {row_data['投信5日淨額(張)']} 張\n- 外資淨額: {row_data['外資5日淨額(張)']} 張\n- 籌碼共振: {'✅' if row_data['★籌碼共振(大戶↑散戶↓)'] else '❌'}")
                c2.write(f"**📈 基本與波動**\n- 產業: {row_data['產業']}\n- 營收增率: {row_data['最新月營收增率(%)']}%\n- ATR%: {row_data['ATR%']}%")
            else: st.info("💡 該股未出現在本次選股名單中，暫無籌碼與評分數據。")

        with tab3:
            st.subheader("📝 交易筆記")
            st.caption("內容自動儲存，切換股票或刷新頁面後皆不會遺失。")
            
            # 🔥 核心優化：直接將 sid 綁入 on_change 參數，保證隔離不遺失
            st.text_area(
                f"紀錄對 {sid} {sname} 的看法…",
                value=st.session_state.trading_notes.get(sid, ""),
                key=f"note_input_{sid}",
                on_change=_note_on_change_cb,
                args=(sid,),
                height=180,
            )
            
            # 📥 [新增] 筆記備份下載系統
            st.divider()
            st.subheader("💾 筆記資料庫管理")
            notes_json_str = json.dumps(st.session_state.trading_notes, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 一鍵匯出/下載所有交易筆記 (JSON)",
                data=notes_json_str,
                file_name=f"taiwan_stock_notes_{pd.Timestamp.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )