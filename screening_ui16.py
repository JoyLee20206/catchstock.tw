"""Streamlit UI for screening0515.py

執行方式:
    streamlit run screening_ui16.py
"""

# ── 頂層 import ───────────────────────────────────────────────────────────
import os
import json
import tempfile
import textwrap
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from performance import compute_performance
from industry_rotation import compute_industry_rotation
import plotly.express as px # 用於繪製績效直方圖

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

# 共用模組
from ai_helper import call_openrouter_ai, AI_MODELS, get_api_key
from cache_status import cache_freshness
from picks_history import load_history, compute_hot_picks

# ── 自選股與交易筆記持久化 ────────────────────────────────────────────────
WATCHLIST_FILE = "cache/watchlist.json"
NOTES_FILE = "cache/notes.json"

def load_watchlist() -> list:
    if Path(WATCHLIST_FILE).exists():
        with open(WATCHLIST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_watchlist(wl: list) -> None:
    Path("cache").mkdir(exist_ok=True)
    with open(WATCHLIST_FILE, 'w', encoding='utf-8') as f:
        json.dump(wl, f, ensure_ascii=False, indent=2)

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
    Path("cache").mkdir(exist_ok=True)
    with open(NOTES_FILE, 'w', encoding='utf-8') as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

def _note_on_change_cb(sid: str) -> None:
    val = st.session_state.get(f"note_input_{sid}", "")
    st.session_state.trading_notes[sid] = val
    save_note(sid, val)

# ── 資料快取讀取 ───────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def get_stock_institutional(stock_id: str) -> pd.DataFrame:
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

@st.cache_data(ttl=300, show_spinner=False)
def _cached_stock_history(sid: str, n_days: int = 200) -> pd.DataFrame:
    return get_stock_history(sid, n_days)

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

# ── 頁面設定與快取狀態橫幅 ──────────────────────────────────────────────────
st.set_page_config(page_title="台股選股", page_icon="📊", layout="wide")
st.title("📊 台股選股工具")
st.caption("拖滑桿調參數 → 按開始選股 → 結果可下載匯入 嘉實 / 精誠的看盤軟體。")

@st.cache_data(ttl=300)
def _get_cache_max_date():
    files = sorted(CACHE_DIR.glob('daily_*.parquet'))
    if not files: return None
    try:
        df = pd.read_parquet(files[-1], columns=['date'])
        if df.empty: return None
        return pd.to_datetime(df['date']).max()
    except Exception: return None

_cache_date = _get_cache_max_date()
_freshness = cache_freshness(_cache_date)
_level_fn = {
    "missing": st.error, "ok": st.success, "info": st.info,
    "warn": st.warning, "error": st.error,
}
_level_fn[_freshness["level"]](f"📅 {_freshness['msg']}")

if _freshness["level"] != "missing":
    st.markdown("#### 🔄 資料更新") 
    st.caption("雲端環境專用：點擊後從網路抓取最新台股資料。")

    if st.session_state.get('show_update_success'):
        st.toast("✅ 雲端資料檢查/更新完成！", icon="🎉")
        st.session_state.show_update_success = False

    if st.button("📥 抓取今日最新資料", type="secondary", use_container_width=True):
        st.toast("⏳ 系統已收到請求，開始比對資料...", icon="🤖") 
        with st.spinner("正在執行資料更新..."):
            try:
                import subprocess, sys
                subprocess.run([sys.executable, "fetch_cache.py"], capture_output=True, text=True, check=True)
                st.cache_data.clear()
                st.session_state.show_update_success = True
                st.rerun()
            except Exception as e:
                st.error(f"❌ 發生錯誤：{e}")

# ── Session state 初始化 ────────────────────────────────────────────────────
DEFAULTS = {
    'pass_score': PASS_SCORE, 'lookback_days': LOOKBACK_DAYS, 'it_min_buy_days': IT_MIN_BUY_DAYS,
    'fi_min_buy_days': FI_MIN_BUY_DAYS, 'kd_lookback': KD_LOOKBACK, 'kd_low_from': KD_LOW_FROM,
    'kd_high_cap_now': KD_HIGH_CAP_NOW, 'min_avg_vol_lots': MIN_AVG_VOL_LOTS, 'atr_max_pct': ATR_MAX_PCT,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

_state_defaults = {'result_df': None, 'result_files': {}, 'result_meta': None, 'target_sid': None, 'last_df_selection': []}
for k, v in _state_defaults.items():
    if k not in st.session_state: st.session_state[k] = v

if 'trading_notes' not in st.session_state: st.session_state.trading_notes = load_notes()
if 'watchlist' not in st.session_state: st.session_state.watchlist = load_watchlist()
if 'ai_cache' not in st.session_state: st.session_state.ai_cache = {} 

def apply_preset(name: str) -> None:
    for k, v in DEFAULTS.items(): st.session_state[k] = v
    for k, v in PRESETS[name].items(): st.session_state[k] = v

def labeled(base: str, key: str) -> str:
    return f"{base} ●" if st.session_state.get(key) != DEFAULTS[key] else base

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.subheader("📱 介面設定")
    is_mobile = st.checkbox("切換為手機版排版 (垂直排列)", value=False)
    
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
            if c2.button("❌", key=f"del_{w_sid}"):
                st.session_state.watchlist.remove(w_sid)
                save_watchlist(st.session_state.watchlist)
                st.rerun()

# ── 大盤狀態與選股執行 ─────────────────────────────────────────────────────
def show_market_banner(meta: dict) -> None:
    if not meta: return
    base = meta.get('base_pass_score')
    eff  = meta.get('effective_pass_score')
    if not meta.get('market_data_ok'):
        st.warning(f"🟡 大盤資料抓取失敗，本次無 RS 計分 | 過關門檻 {base} → **{eff}**")
        return
    bullish      = meta.get('market_bullish', True)
    twii_now_raw = meta.get('twii_now')
    twii_ma_raw  = meta.get('twii_ma')
    change_raw   = meta.get('twii_lookback_change')
    change       = float(change_raw) if pd.notna(change_raw) else 0.0
    state_txt    = "📈 多頭(站上季線)" if bullish else "📉 空頭(跌破季線)"
    twii_now_s   = f"{twii_now_raw:,.0f}" if pd.notna(twii_now_raw) else "N/A"
    twii_ma_s    = f"{twii_ma_raw:,.0f}"  if pd.notna(twii_ma_raw)  else "N/A"
    thr_txt = f"過關門檻 {base} → **{eff}**" if base != eff else f"過關門檻 {base}"
    msg = f"{state_txt}  |  {thr_txt}  |  TWII {twii_now_s} / MA60 {twii_ma_s} | 近 20 日 {change:+.2f}%"
    
    # 💡 修正處：改回標準的 if-else 寫法，避免觸發 Streamlit Magic 的解析 Bug
    if bullish:
        st.success(msg)
    else:
        st.warning(msg)

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

# ── 結果顯示區 ─────────────────────────────────────────────────────────────
df = st.session_state.result_df
files_bytes = st.session_state.result_files or {}
meta = st.session_state.result_meta

if meta: show_market_banner(meta)

# ── 歷史數據大局觀(產業輪動與策略績效) ──
@st.cache_data(ttl=300, show_spinner=False)
def _load_hot_picks_cached(top_n: int = 10):
    hist = load_history()
    return compute_hot_picks(hist, top_n=top_n), len(hist)

_hot, _hist_days = _load_hot_picks_cached(top_n=10)
hist_data = load_history()

with st.expander("🌍 大局觀：產業輪動 & 策略績效驗證", expanded=False):
    tab_ind, tab_perf = st.tabs(["🔄 產業輪動 (近7日)", "📈 策略績效 (近30日)"])
    
    with tab_ind:
        rotations = compute_industry_rotation(hist_data, recent_days=7, prev_days=7)
        if rotations:
            st.caption("觀察資金正在流入 (↑) 或流出 (↓) 哪些板塊")
            rot_cols = st.columns(4)
            for i, r in enumerate(rotations[:8]):
                trend = "↑" if r['direction'] == 'up' else ("↓" if r['direction'] == 'down' else "平")
                color = "normal" if trend == "平" else ("inverse" if trend == "↓" else "normal")
                rot_cols[i % 4].metric(r['industry'], f"{r['recent_count']} 檔次", f"{trend} (上週 {r['prev_count']})", delta_color=color)
        else:
            st.info("資料不足，無法計算產業輪動。")

    with tab_perf:
        perf = compute_performance(hist_data, CACHE_DIR, n_days_list=(5, 10, 20))
        overall = perf.get("overall", {})
        if "win_rate_5d" in overall:
            c1, c2, c3 = st.columns(3)
            c1.metric("5日後勝率", f"{overall['win_rate_5d']*100:.1f}%")
            c2.metric("5日平均報酬", f"{overall['avg_return_5d']:+.2f}%")
            c3.metric("總樣本數", f"{overall['n_5d']} 筆")
            
            samples = [s['return_5d'] for s in perf['samples'] if s.get('return_5d') is not None]
            if samples:
                fig_hist = px.histogram(x=samples, nbins=20, labels={'x':'5日報酬率(%)', 'y':'次數'}, title="入選後 5 日報酬分佈")
                fig_hist.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("目前歷史資料不足以計算 5 日後績效。")

if _hot:
    with st.expander(f"🔥 過去 {_hist_days} 日入選熱度榜(TOP 10)", expanded=False):
        hot_rows = []
        for r in _hot:
            sid_str = str(r["sid"])
            hot_rows.append({
                "★今日": "✅" if r["in_latest"] else "", "代號": sid_str, "名稱": ui_name_map.get(sid_str, ""),
                "入選天數": f"{r['hits']} / {r['total_days']}", "最長連續": f"{r['max_streak']} 日",
                "目前連續": f"{r['active_streak']} 日" if r['active_streak'] >= 1 else "—",
            })
        st.dataframe(pd.DataFrame(hot_rows), use_container_width=True, hide_index=True)

st.divider()

# 🚀 升級：動態排版切換 (支援手機版)
if is_mobile:
    col_list = st.container()
    st.divider()
    col_chart = st.container()
    chart_height = 500
else:
    col_list, col_chart = st.columns([0.45, 0.55])
    chart_height = 850

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

        codes_csv = ",".join(filtered['代號'].astype(str).tolist())
        if codes_csv:
            st.caption(f"📋 已篩選 {len(filtered)} 檔 — 點右上角圖示一鍵複製")
            st.code(codes_csv, language=None)

        current_sel = event.selection.rows
        if current_sel != st.session_state.last_df_selection:
            st.session_state.last_df_selection = current_sel
            if current_sel: st.session_state.target_sid = str(filtered.iloc[current_sel[0]]['代號'])

if (not st.session_state.target_sid) and (df is not None) and (len(df) > 0):
    st.session_state.target_sid = str(df.iloc[0]['代號'])

with col_chart:
    if not st.session_state.target_sid:
        st.info("👈 請點擊左側列表或自選股，開始量化決策分析。")
    else:
        sid = str(st.session_state.target_sid)
        row_data = None
        if df is not None and len(df) > 0 and sid in df['代號'].astype(str).values:
            row_data = df[df['代號'].astype(str) == sid].iloc[0]

        sname = str(row_data['名稱']) if row_data is not None and pd.notna(row_data['名稱']) else ui_name_map.get(sid, "")
        
        # 🚀 升級：新增「資金計算」Tab
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 技術分析", "📊 籌碼/基本面", "🧠 AI 虛擬點評", "📝 交易筆記", "🧮 資金計算"])
        hist_daily = _cached_stock_history(sid)

        with tab1:
            # 🚀 升級：進場節奏燈號判斷
            entry_signal = "🔵 觀察訊號剛初期 (下一兩天再看)"
            if not hist_daily.empty:
                latest_close = hist_daily['close'].iloc[-1]
                ma20_last = hist_daily['close'].rolling(20).mean().iloc[-1]
                k_list, _ = _calc_kd_series(hist_daily['max'] if 'max' in hist_daily.columns else hist_daily['high'], 
                                            hist_daily['min'] if 'min' in hist_daily.columns else hist_daily['low'], 
                                            hist_daily['close'])
                k_last = next((k for k in reversed(k_list) if k is not None), 50) if k_list else 50
                
                if pd.notna(ma20_last):
                    dist_ma20 = (latest_close - ma20_last) / ma20_last
                    if k_last > 80 or dist_ma20 > 0.15:
                        entry_signal = "🟡 追高有風險、等待拉回 (訊號強但 K 值超買或乖離過大)"
                    elif k_last < 60 and dist_ma20 < 0.08:
                        entry_signal = "🟢 可進場 (訊號明確、位階合理、未超買)"

            st.info(f"**進場節奏：** {entry_signal}")

            tcol1, tcol2, tcol3, tcol4 = st.columns([1.5, 1.5, 1.5, 1])
            tcol1.markdown(f"### {sid} {sname}")
            timeframe = tcol2.radio("週期", ["日K", "週K"], horizontal=True, label_visibility="collapsed")
            zoom_choice = tcol3.radio("縮放", ["1月", "3月", "半年", "全部"], horizontal=True, index=3, label_visibility="collapsed")
            theme_choice = tcol4.radio("主題", ["深色", "淺色"], horizontal=True, label_visibility="collapsed")

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

                inst_stock = get_stock_institutional(sid)
                if not inst_stock.empty:
                    inst_stock['net_lots'] = (inst_stock['buy'] - inst_stock['sell']) / 1000.0
                    inst_pivot = inst_stock.pivot_table(index='date', columns='name', values='net_lots', aggfunc='sum').reset_index()
                    for c in ['Foreign_Investor', 'Investment_Trust']:
                        if c not in inst_pivot.columns: inst_pivot[c] = 0.0
                    if timeframe == "週K": inst_pivot = inst_pivot.set_index('date').resample('W-FRI').sum().reset_index()
                    else: inst_pivot['date'] = pd.to_datetime(inst_pivot['date'])
                    hist['date'] = pd.to_datetime(hist['date'])
                    hist = hist.merge(inst_pivot, on='date', how='left').fillna(0)
                else:
                    hist['Foreign_Investor'], hist['Investment_Trust'] = 0.0, 0.0

                hist['MA5'], hist['MA20'], hist['MA60'] = hist['close'].rolling(5).mean(), hist['close'].rolling(20).mean(), hist['close'].rolling(60).mean()
                k_list, d_list = _calc_kd_series(hist['max'], hist['min'], hist['close'])
                if k_list is None: hist['K'], hist['D'] = float('nan'), float('nan')
                else:
                    hist['K'] = [x if x is not None else float('nan') for x in k_list]
                    hist['D'] = [x if x is not None else float('nan') for x in d_list]

                kd_low_thr = st.session_state.kd_low_from
                kd_cross = ((hist['K'] > hist['D']) & (hist['K'].shift(1) <= hist['D'].shift(1)) & (hist['K'] < kd_low_thr))
                signal_dates, signal_prices = hist[kd_cross]['date'], hist[kd_cross]['min'] * 0.95

                latest_close = hist_daily['close'].iloc[-1]
                prev_close = hist_daily['close'].iloc[-2] if len(hist_daily) > 1 else latest_close
                m1, m2, m3 = st.columns(3)
                m1.metric("日收盤價", f"{latest_close:.2f}", f"{latest_close - prev_close:+.2f} ({(latest_close - prev_close)/prev_close*100:+.2f}%)")
                m2.metric("日成交量", f"{hist_daily[vol_col].iloc[-1]/1000:,.0f} 張")
                m3.metric("選股總分", f"{row_data['總分']} 分" if row_data is not None else "未入選")

                fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.42, 0.12, 0.18, 0.28])
                fig.add_trace(go.Candlestick(x=hist['date'], open=hist['open'], high=hist['max'], low=hist['min'], close=hist['close'], name='K線', increasing_fillcolor='red', increasing_line_color='red', decreasing_fillcolor='green', decreasing_line_color='green'), row=1, col=1)

                for col_ma, color in [('MA5', 'orange'), ('MA20', 'purple'), ('MA60', 'green')]:
                    fig.add_trace(go.Scatter(x=hist['date'], y=hist[col_ma], line=dict(color=color, width=1.2), name=col_ma), row=1, col=1)

                if not signal_dates.empty:
                    fig.add_trace(go.Scatter(x=signal_dates, y=signal_prices, mode='markers', marker=dict(symbol='triangle-up', color='lime', size=12), name='KD金叉'), row=1, col=1)

                vol_colors = ['red' if c >= o else 'green' for c, o in zip(hist['close'], hist['open'])]
                fig.add_trace(go.Bar(x=hist['date'], y=hist[vol_col], marker_color=vol_colors, name='成交量'), row=2, col=1)
                fig.add_trace(go.Bar(x=hist['date'], y=hist['Investment_Trust'], marker_color='#FF4B4B', name='投信(張)'), row=3, col=1)
                fig.add_trace(go.Bar(x=hist['date'], y=hist['Foreign_Investor'], marker_color='#FACA44', name='外資(張)'), row=3, col=1)
                fig.add_trace(go.Scatter(x=hist['date'], y=hist['K'], line=dict(color='blue', width=1.2), name='K'), row=4, col=1)
                fig.add_trace(go.Scatter(x=hist['date'], y=hist['D'], line=dict(color='orange', width=1.2), name='D'), row=4, col=1)

                fig.update_layout(template=chart_theme, paper_bgcolor=bg_color, plot_bgcolor=bg_color, height=chart_height, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                if zoom_choice != "全部":
                    days_map = {"1月": 30, "3月": 90, "半年": 180}
                    start_date = pd.to_datetime(hist['date'].iloc[-1]) - pd.Timedelta(days=days_map[zoom_choice])
                    fig.update_xaxes(range=[start_date, hist['date'].iloc[-1]])
                
                st.plotly_chart(fig, use_container_width=True)
            else: st.warning("暫無歷史數據。")

        with tab2:
            st.subheader("📊 詳細數據分析")
            if row_data is not None:
                c1, c2 = st.columns(2)
                c1.write(f"**💎 籌碼數據**\n- 投信淨額: {row_data['投信5日淨額(張)']} 張\n- 外資淨額: {row_data['外資5日淨額(張)']} 張\n- 籌碼共振: {'✅' if row_data['★籌碼共振(大戶↑散戶↓)'] else '❌'}")
                c2.write(f"**📈 基本與波動**\n- 產業: {row_data['產業']}\n- 營收增率: {row_data['最新月營收增率(%)']}%\n- ATR%: {row_data['ATR%']}%")
            else: st.info("💡 該股未出現在本次選股名單中。")

        with tab3:
            st.subheader("🧠 OpenRouter AI 虛擬操盤分析師")
            if not bool(get_api_key()): st.warning("⚠️ 未偵測到 API_KEY")
            elif row_data is None: st.info("💡 該股未達標，暫無數據供 AI 點評。")
            else:
                model_options = ["自動輪替 (推薦)"] + [m["name"] for m in AI_MODELS]
                selected_model = st.selectbox("🤖 選擇 AI 模型", model_options)
                cache_key = (sid, selected_model)

                if cache_key in st.session_state.ai_cache:
                    st.info(st.session_state.ai_cache[cache_key][1])
                    if st.button("🔄 重新生成"): 
                        del st.session_state.ai_cache[cache_key]
                        st.rerun()
                else:
                    if st.button("🚀 啟動 AI 深度量化解析", use_container_width=True):
                        prompt = textwrap.dedent(f"""\
                            你是一位資深量化分析師。請用繁體中文寫一份約 150 字的專業診斷，不要使用 Markdown。
                            股票:{sid} {sname}
                            總分:{row_data['總分']}/10
                            """)
                        with st.spinner("AI 分析中..."):
                            m_try = None if selected_model == "自動輪替 (推薦)" else [m for m in AI_MODELS if m["name"] == selected_model]
                            m_name, res = call_openrouter_ai(prompt, models=m_try, max_tokens=400)
                        if m_name:
                            st.session_state.ai_cache[cache_key] = (m_name, res)
                            st.rerun()

        with tab4:
            st.subheader("📝 交易筆記")
            st.text_area(f"對 {sid} 的看法…", value=st.session_state.trading_notes.get(sid, ""), key=f"note_input_{sid}", on_change=_note_on_change_cb, args=(sid,), height=180)
            
        # 🚀 升級：資金管理計算器
        with tab5:
            st.subheader("🧮 固定風險資金管理計算器")
            st.caption("根據您願意承受的單筆虧損，計算出合理的買進張數。")
            st.markdown("公式： $$買進張數 = \\frac{總資金 \\times (風險\\% \\div 100)}{(買進價 - 停損價) \\times 1000}$$")
            
            latest_price = hist_daily['close'].iloc[-1] if not hist_daily.empty else 0.0
            
            calc_c1, calc_c2 = st.columns(2)
            capital = calc_c1.number_input("總操作資金 (元)", min_value=10000, value=1000000, step=100000)
            risk_pct = calc_c2.number_input("單筆最大風險 (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
            
            buy_price = calc_c1.number_input("預計買進價", value=float(latest_price))
            stop_loss = calc_c2.number_input("防守停損價", value=float(latest_price * 0.95))
            
            if buy_price > stop_loss:
                risk_amount = capital * (risk_pct / 100)
                risk_per_share = buy_price - stop_loss
                shares_to_buy = risk_amount / (risk_per_share * 1000)
                
                st.success(f"🛡️ 單筆允許虧損金額：**{risk_amount:,.0f} 元**")
                st.info(f"📦 建議買進張數：**{shares_to_buy:.2f} 張** (約需投入 {shares_to_buy * buy_price * 1000:,.0f} 元)")
            else:
                st.error("停損價必須低於買進價！")