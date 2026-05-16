"""Streamlit UI for screening0515.py

執行方式:
    streamlit run screening_ui.py

第一次使用要先安裝依賴:
    pip install streamlit plotly yfinance

修正記錄:
  #1 補上缺漏的 get_stock_history (screening0515.py 已同步補定義)
  #2 「訊號觸發」標記只在股票確實入選時才顯示
  #3 圖表 KD 改用 screening0515._calc_kd_series (正確的 2/3 平滑)
  #4 KD 金叉偵測門檻改讀 session_state.kd_low_from (不再硬寫 30)
  #5 files_bytes 改用 `or {}` 防 None
  #6 雙重 import 合併為一個 import 區塊
  #7 yfinance 移至頂層 import；大盤 RS 改走快取函式
  #8 大盤 RS 下載加 @st.cache_data(ttl=3600) 避免每次切股都重打 API
  #9 交易筆記改寫入 notes.json，頁面重整後不遺失
"""

# ── 頂層 import (Fix #6 #7: 雙重 import 合併，yfinance 移到頂層) ──────────
import json
import tempfile
from pathlib import Path

import pandas as pd
import yfinance as yf                                                   # Fix #7
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from screening0515 import (                                              # Fix #1 #6
    run_screening,
    get_stock_history,          # Fix #1: 現在 screening0515.py 已有此定義
    _calc_kd_series,            # Fix #3: 引入正確的 KD 計算函式
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

# ── 交易筆記持久化 (Fix #9) ────────────────────────────────────────────────
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

def _note_on_change_cb() -> None:
    """text_area on_change：自動把最新內容寫進 notes.json。"""
    sid = st.session_state.get('_note_editing_sid')
    if sid:
        save_note(sid, st.session_state.get(f"note_{sid}", ""))

# ── 大盤 RS 快取 (Fix #8: 每小時才重打 API，不再每次切股都下載) ──────────
@st.cache_data(ttl=3600, show_spinner=False)
def _load_twii_cached() -> pd.DataFrame:
    """下載 2 年 TWII 並快取；重整頁面 / 切換股票不重複請求。"""
    try:
        data = yf.download("^TWII", period="2y", auto_adjust=True,
                           progress=False, threads=False)
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
st.caption("拖滑桿調參數 → 按開始選股 → 結果可下載匯入 嘉實 / 精誠。"
           "策略核心(法人/營收/籌碼週數)維持程式預設，要改請編輯 screening0515.py。")

# ── Cache 新鮮度警示 ───────────────────────────────────────────────────────
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
    _today    = pd.Timestamp.now().normalize()
    _age      = (_today - _cache_date.normalize()).days
    _date_str = _cache_date.strftime('%Y-%m-%d')
    if _age == 0:
        st.success(f"📅 cache 最新日期: {_date_str}(今天)")
    elif _age <= 2:
        st.info(f"📅 cache 最新日期: {_date_str}({_age} 天前)")
    elif _age <= 7:
        st.warning(f"📅 cache 最新日期: {_date_str}({_age} 天前)— 建議跑 `python fetch_cache.py` 更新")
    else:
        st.error(f"📅 cache 最新日期: {_date_str}({_age} 天前) ⚠ 過舊，請先更新")

# ── Session state 初始化 (Fix #11: 各 key 獨立初始化，不包在一個 if 裡) ──
DEFAULTS = {
    'pass_score':       PASS_SCORE,
    'lookback_days':    LOOKBACK_DAYS,
    'it_min_buy_days':  IT_MIN_BUY_DAYS,
    'fi_min_buy_days':  FI_MIN_BUY_DAYS,
    'kd_lookback':      KD_LOOKBACK,
    'kd_low_from':      KD_LOW_FROM,
    'kd_high_cap_now':  KD_HIGH_CAP_NOW,
    'min_avg_vol_lots': MIN_AVG_VOL_LOTS,
    'atr_max_pct':      ATR_MAX_PCT,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

_state_defaults = {
    'result_df':          None,
    'result_files':       {},      # Fix #5: 預設 dict，不是 None
    'result_meta':        None,
    'target_sid':         None,
    'last_df_selection':  [],
    '_note_editing_sid':  None,
}
for k, v in _state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Fix #9: 把已儲存的筆記預先塞進 session_state，widget 第一次渲染就有內容
if 'notes_loaded' not in st.session_state:
    for note_sid, note_text in load_notes().items():
        key = f"note_{note_sid}"
        if key not in st.session_state:
            st.session_state[key] = note_text
    st.session_state.notes_loaded = True

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

# ── Preset 套用 ────────────────────────────────────────────────────────────
def apply_preset(name: str) -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    for k, v in PRESETS[name].items():
        st.session_state[k] = v

def labeled(base: str, key: str) -> str:
    return f"{base} ●" if st.session_state.get(key) != DEFAULTS[key] else base

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("🎯 快速套用組合")
    st.button("預設",     on_click=apply_preset, args=('default',),   use_container_width=True)
    st.button("多頭寬鬆", on_click=apply_preset, args=('bull',),      use_container_width=True,
              help="PASS_SCORE 6, KD 視窗較寬，適合多頭環境放寬選股")
    st.button("空頭嚴格", on_click=apply_preset, args=('bear',),      use_container_width=True,
              help="PASS_SCORE 8, KD 短視窗，流動性門檻提高，適合熊市保守選")
    st.button("KD 起漲",  on_click=apply_preset, args=('kd_start',),  use_container_width=True,
              help="專攻 KD 低檔金叉訊號，PASS_SCORE 較寬，KD 條件適中")

    st.divider()
    st.subheader("📌 核心 / KD")
    st.slider(labeled("過關門檻 PASS_SCORE", 'pass_score'), 1, 10, key='pass_score',
              help="滿分 10。大盤跌破季線自動 +1；^TWII 抓取失敗自動 -1")
    st.slider(labeled("KD 觀察區間 (天)", 'kd_lookback'), 1, 30, key='kd_lookback',
              help="近 N 個交易日內是否發生低檔金叉。短=要剛起漲；長=容許更早的訊號")
    st.slider(labeled("KD 低檔啟動門檻", 'kd_low_from'), 10, 50, key='kd_low_from',
              help="交叉當天 K 須 < 此值，證明從低檔啟動。低=越嚴格")
    st.slider(labeled("KD 今日上限", 'kd_high_cap_now'), 50, 95, key='kd_high_cap_now',
              help="今日 K 超過此值代表已超買，即使曾低檔金叉也不算")

    st.divider()
    st.subheader("📌 預篩 / 法人")
    st.slider(labeled("20 日均量下限 (張)", 'min_avg_vol_lots'), 100, 2000, key='min_avg_vol_lots', step=50,
              help="流動性過濾，低於此值直接剔除(不計分)")
    st.slider(labeled("ATR% 上限", 'atr_max_pct'), 5.0, 25.0, key='atr_max_pct', step=0.5,
              help="ATR(14)/現價 超過此值視為波動過大、飆股，直接剔除")
    st.slider(labeled("法人觀察天數", 'lookback_days'), 3, 10, key='lookback_days',
              help="投信/外資累計買超的觀察視窗")
    st.slider(labeled("投信最少買超日", 'it_min_buy_days'), 1, 5, key='it_min_buy_days',
              help="觀察期內投信至少幾日買超才算訊號")
    st.slider(labeled("外資最少買超日", 'fi_min_buy_days'), 1, 5, key='fi_min_buy_days',
              help="觀察期內外資至少幾日買超")

    st.divider()
    st.caption("● = 已改自預設")
    run_clicked = st.button("▶ 開始選股", type='primary', use_container_width=True)

    # ── 自選股清單 ──
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
    if not meta:
        return
    base = meta.get('base_pass_score')
    eff  = meta.get('effective_pass_score')
    if not meta.get('market_data_ok'):
        st.warning(f"🟡 大盤資料抓取失敗，本次無 RS 計分 | 過關門檻 {base} → **{eff}**(自動 -1 補償)")
        return
    bullish   = meta.get('market_bullish', True)
    twii_now  = meta.get('twii_now')
    twii_ma   = meta.get('twii_ma')
    change    = meta.get('twii_lookback_change', 0.0)
    state_txt = "🟢 多頭(站上季線)" if bullish else "🔴 空頭(跌破季線)"
    thr_txt   = f"過關門檻 {base}" if base == eff else f"過關門檻 {base} → **{eff}**(空頭自動 +1)"
    twii_txt  = f"TWII {twii_now:,.0f} / MA60 {twii_ma:,.0f} | 近 20 日 {change:+.2f}%"
    msg = f"{state_txt}  |  {thr_txt}  |  {twii_txt}"
    if bullish:
        st.success(msg)
    else:
        st.warning(msg)

# ── 選股執行 ───────────────────────────────────────────────────────────────
if run_clicked:
    with st.spinner("選股中，請稍候(約 30~60 秒)..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            df, file_paths, meta = run_screening(
                pass_score       = st.session_state.pass_score,
                lookback_days    = st.session_state.lookback_days,
                it_min_buy_days  = st.session_state.it_min_buy_days,
                fi_min_buy_days  = st.session_state.fi_min_buy_days,
                kd_lookback      = st.session_state.kd_lookback,
                kd_low_from      = st.session_state.kd_low_from,
                kd_high_cap_now  = st.session_state.kd_high_cap_now,
                min_avg_vol_lots = st.session_state.min_avg_vol_lots,
                atr_max_pct      = st.session_state.atr_max_pct,
                output_dir       = Path(tmpdir),
            )
            files_bytes: dict = {}
            for ftype, fpath in file_paths.items():
                if Path(fpath).exists():
                    with open(fpath, 'rb') as fh:
                        files_bytes[ftype] = (Path(fpath).name, fh.read())
        st.session_state.result_df    = df
        st.session_state.result_files = files_bytes    # Fix #5: 一定是 dict
        st.session_state.result_meta  = meta
    st.success("✅ 完成，看下方結果")

# ── 結果顯示區 ─────────────────────────────────────────────────────────────
df          = st.session_state.result_df
files_bytes = st.session_state.result_files or {}     # Fix #5: 防 None
meta        = st.session_state.result_meta

if meta:
    show_market_banner(meta)
st.divider()

col_list, col_chart = st.columns([0.45, 0.55])

with col_list:
    if df is None:
        st.info("👈 請先按左側「開始選股」產生最新清單。")
    elif len(df) == 0:
        st.error("❌ 沒有標的達標，試試降低 PASS_SCORE 或按「多頭寬鬆」preset")
    else:
        st.subheader(f"📋 結果({len(df)} 檔)")

        fc1, fc2 = st.columns(2)
        with fc1:
            score_min, score_max = int(df['總分'].min()), int(df['總分'].max())
            score_threshold = (
                st.slider("總分 ≥", score_min, score_max, score_min)
                if score_min != score_max else score_min
            )
        with fc2:
            industries = sorted([i for i in df['產業'].dropna().unique() if i])
            selected_industries = st.multiselect("產業篩選", industries, default=industries)

        filtered = df[
            (df['總分'] >= score_threshold) &
            (df['產業'].isin(selected_industries))
        ]

        event = st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        st.divider()
        st.subheader("📥 下載匯入檔")
        st.caption("⚠ 下載的是完整結果，不套用上方篩選")
        dl_cols = st.columns(3)
        MIME_TYPES = {
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'dsl':  'application/octet-stream',
            'xls':  'application/vnd.ms-excel',
        }
        LABELS = {'xlsx': '📥 Excel', 'dsl': '📥 嘉實', 'xls': '📥 精誠'}
        for i, ftype in enumerate(['xlsx', 'dsl', 'xls']):
            if ftype in files_bytes:
                fname, data = files_bytes[ftype]
                dl_cols[i].download_button(
                    LABELS[ftype], data, fname, MIME_TYPES[ftype], use_container_width=True,
                )
            else:
                dl_cols[i].button(LABELS[ftype] + " (無)", disabled=True, use_container_width=True)

        # 表格點擊偵測
        current_sel = event.selection.rows
        if current_sel != st.session_state.last_df_selection:
            st.session_state.last_df_selection = current_sel
            if current_sel:
                st.session_state.target_sid = str(filtered.iloc[current_sel[0]]['代號'])

# ── K 線圖區 ───────────────────────────────────────────────────────────────
with col_chart:
    if not st.session_state.target_sid:
        st.info("👈 請點擊左側列表或自選股，開始量化決策分析。")
    else:
        sid = str(st.session_state.target_sid)

        # 從選股結果找 row_data（自選股若未入選則 None）
        row_data = None
        if df is not None and len(df) > 0:
            df['代號_str'] = df['代號'].astype(str)
            if sid in df['代號_str'].values:
                row_data = df[df['代號_str'] == sid].iloc[0]

        sname = row_data['名稱'] if row_data is not None else ui_name_map.get(sid, "")

        tab1, tab2, tab3 = st.tabs(["📈 技術分析", "📊 籌碼/基本面", "📝 交易筆記"])

        hist_daily = get_stock_history(sid)    # Fix #1: 呼叫已正確定義的函式

        with tab1:
            tcol1, tcol2, tcol3 = st.columns([2, 1.5, 1])
            tcol1.markdown(f"### {sid} {sname}")
            timeframe    = tcol2.radio("週期", ["日K", "週K"], horizontal=True,
                                       label_visibility="collapsed")
            theme_choice = tcol3.radio("主題", ["深色", "淺色"], horizontal=True,
                                       label_visibility="collapsed")

            is_in_watchlist = sid in st.session_state.watchlist
            if tcol1.button("⭐ 移除自選" if is_in_watchlist else "⭐ 加入自選",
                            key=f"add_{sid}"):
                if is_in_watchlist:
                    st.session_state.watchlist.remove(sid)
                else:
                    st.session_state.watchlist.append(sid)
                save_watchlist(st.session_state.watchlist)
                st.rerun()

            chart_theme = 'plotly_dark' if theme_choice == "深色" else 'plotly_white'
            bg_color    = '#111'  if theme_choice == "深色" else 'white'
            grid_color  = '#333'  if theme_choice == "深色" else '#EEE'

            if not hist_daily.empty:
                # 欄位名稱相容 (bonus 小修: max/min 或 high/low 都正確處理)
                high_col = 'max'  if 'max'  in hist_daily.columns else 'high'
                low_col  = 'min'  if 'min'  in hist_daily.columns else 'low'
                # [新增] 動態判斷成交量欄位
                vol_col  = 'Trading_Volume' if 'Trading_Volume' in hist_daily.columns else 'volume'

                if timeframe == "週K":
                    hist = (
                        hist_daily.set_index('date')
                        .resample('W-FRI')
                        # [修改] 將 'Trading_Volume' 換成 vol_col
                        .agg({'open': 'first', high_col: 'max', low_col: 'min',
                              'close': 'last', vol_col: 'sum'}) 
                        .dropna()
                        .reset_index()
                    )
                    hist = hist.rename(columns={high_col: 'max', low_col: 'min'})
                else:
                    hist = hist_daily.rename(columns={high_col: 'max', low_col: 'min'}).copy()

                # ── 大盤 RS (Fix #8: 使用快取，不再每切一股就重打 API) ──────
                twii_all = _load_twii_cached()
                if not twii_all.empty:
                    mask   = ((twii_all.index >= hist['date'].min()) &
                              (twii_all.index <= hist['date'].max()))
                    m_data = twii_all[mask]
                    if timeframe == "週K":
                        m_data = m_data.resample('W-FRI').agg({'Close': 'last'}).dropna()
                else:
                    m_data = pd.DataFrame()

                hist['Stock_Norm'] = hist['close'] / hist['close'].iloc[0] * 100
                if not m_data.empty and 'Close' in m_data.columns:
                    m_close = m_data['Close'].reindex(hist.set_index('date').index).ffill()
                    if len(m_close) > 0 and m_close.iloc[0] > 0:
                        hist['Market_Norm'] = m_close.values / m_close.iloc[0] * 100

                # ── 技術指標 ─────────────────────────────────────────────────
                hist['MA5']  = hist['close'].rolling(5).mean()
                hist['MA20'] = hist['close'].rolling(20).mean()
                hist['MA60'] = hist['close'].rolling(60).mean()

                # Fix #3: 改用 screening0515._calc_kd_series (正確的 2/3 平滑)
                k_list, d_list = _calc_kd_series(hist['max'], hist['min'], hist['close'])
                hist['K'] = [x if x is not None else float('nan') for x in k_list]
                hist['D'] = [x if x is not None else float('nan') for x in d_list]

                # Fix #4: KD 金叉門檻讀 session_state，不再硬寫 30
                kd_low_thr = st.session_state.kd_low_from
                kd_cross   = (
                    (hist['K'] > hist['D']) &
                    (hist['K'].shift(1) <= hist['D'].shift(1)) &
                    (hist['K'] < kd_low_thr)
                )
                signal_dates  = hist[kd_cross]['date']
                signal_prices = hist[kd_cross]['min'] * 0.95

                # 指標數字
                latest_close = hist_daily['close'].iloc[-1]
                prev_close   = (hist_daily['close'].iloc[-2]
                                if len(hist_daily) > 1 else latest_close)
                change       = latest_close - prev_close
                m1, m2, m3   = st.columns(3)
                m1.metric("日收盤價", f"{latest_close:.2f}",
                          f"{change:+.2f} ({change/prev_close*100:+.2f}%)")
                m2.metric("日成交量",
                          f"{hist_daily[vol_col].iloc[-1]/1000:,.0f} 張")
                score_display = (f"{row_data['總分']} 分"
                                 if row_data is not None else "未入選")
                m3.metric("選股總分", score_display)

                # ── Plotly 圖表 ──────────────────────────────────────────────
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.5, 0.15, 0.35],
                )

                fig.add_trace(go.Candlestick(
                    x=hist['date'],
                    open=hist['open'], high=hist['max'],
                    low=hist['min'],   close=hist['close'],
                    name='K線',
                    increasing_fillcolor='red',   increasing_line_color='red',
                    decreasing_fillcolor='green', decreasing_line_color='green',
                ), row=1, col=1)

                if 'Market_Norm' in hist.columns:
                    fig.add_trace(go.Scatter(
                        x=hist['date'],
                        y=hist['Market_Norm'] * (hist['close'].iloc[0] / 100),
                        line=dict(color='rgba(150,150,150,0.5)', width=1, dash='dot'),
                        name='大盤 RS',
                    ), row=1, col=1)

                for col_ma, color, lbl in [
                    ('MA5', 'orange', 'MA5'),
                    ('MA20', 'purple', 'MA20'),
                    ('MA60', 'green', 'MA60'),
                ]:
                    fig.add_trace(go.Scatter(
                        x=hist['date'], y=hist[col_ma],
                        line=dict(color=color, width=1.2), name=lbl,
                    ), row=1, col=1)

                if not signal_dates.empty:
                    fig.add_trace(go.Scatter(
                        x=signal_dates, y=signal_prices,
                        mode='markers',
                        marker=dict(symbol='triangle-up', color='lime', size=12,
                                    line=dict(width=1, color='darkgreen')),
                        name='KD金叉訊號',
                    ), row=1, col=1)

                if len(hist) > 60:
                    fig.add_hline(
                        y=hist['max'].iloc[-61:-1].max(),
                        line_dash="dot", line_color="orange",
                        annotation_text="60日壓", annotation_position="top left",
                        row=1, col=1,
                    )
                fig.add_hline(
                    y=max(hist['MA20'].iloc[-1] * 0.98, hist['min'].tail(10).min()),
                    line_dash="dash", line_color="red",
                    annotation_text="🚨 防守", annotation_position="bottom right",
                    row=1, col=1,
                )

                # Fix #2: 「訊號觸發」只在股票確實入選本次選股時才顯示
                if row_data is not None:
                    fig.add_annotation(
                        x=hist['date'].iloc[-1],
                        y=hist['min'].iloc[-1],
                        text="🔥 訊號觸發",
                        showarrow=True, arrowhead=1, arrowcolor="red", ay=30,
                        row=1, col=1,
                    )

                vol_colors = ['red' if c >= o else 'green'
                              for c, o in zip(hist['close'], hist['open'])]
                fig.add_trace(
                    go.Bar(x=hist['date'], y=hist[vol_col], marker_color=vol_colors, name='成交量'),
                    row=2, col=1,
                )

                fig.add_trace(go.Scatter(x=hist['date'], y=hist['K'],
                                         line=dict(color='blue',   width=1.2), name='K'),
                              row=3, col=1)
                fig.add_trace(go.Scatter(x=hist['date'], y=hist['D'],
                                         line=dict(color='orange', width=1.2), name='D'),
                              row=3, col=1)
                fig.add_hline(y=80, line_dash="dash", line_color="gray",
                              line_width=0.5, row=3, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color="gray",
                              line_width=0.5, row=3, col=1)

                all_dates     = pd.date_range(start=hist['date'].min(),
                                              end=hist['date'].max())
                missing_dates = [d.strftime("%Y-%m-%d") for d in all_dates
                                 if d not in hist['date'].dt.normalize().values]

                fig.update_layout(
                    template=chart_theme, paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                    height=800, xaxis_rangeslider_visible=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                    hovermode='x unified', showlegend=False,
                )
                fig.update_xaxes(
                    gridcolor=grid_color,
                    rangebreaks=[dict(values=missing_dates)],
                    type="date",
                )
                fig.update_xaxes(
                    rangeselector=dict(
                        buttons=[
                            dict(count=30,  label="1月",  step="day", stepmode="backward"),
                            dict(count=90,  label="3月",  step="day", stepmode="backward"),
                            dict(count=180, label="半年", step="day", stepmode="backward"),
                            dict(step="all", label="全部"),
                        ],
                        bgcolor='#333' if theme_choice == "深色" else '#EEE',
                    ),
                    row=1, col=1,
                )
                fig.update_yaxes(gridcolor=grid_color, side='right')
                fig.update_yaxes(fixedrange=False, row=1, col=1)

                config = {
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath',
                                            'drawrect', 'eraseshape'],
                    'displayModeBar': True,
                    'displaylogo':    False,
                    'scrollZoom':     True,
                }
                st.plotly_chart(fig, use_container_width=True, config=config)
            else:
                st.warning("暫無歷史數據。")

        with tab2:
            st.subheader("📊 詳細數據分析")
            if row_data is not None:
                c1, c2 = st.columns(2)
                c1.write(
                    f"**💎 籌碼數據**\n"
                    f"- 投信淨額: {row_data['投信5日淨額(張)']} 張\n"
                    f"- 外資淨額: {row_data['外資5日淨額(張)']} 張\n"
                    f"- 籌碼共振: {'✅' if row_data['★籌碼共振(大戶↑散戶↓)'] else '❌'}"
                )
                c2.write(
                    f"**📈 基本與波動**\n"
                    f"- 產業: {row_data['產業']}\n"
                    f"- 營收增率: {row_data['最新月營收增率(%)']}%\n"
                    f"- ATR%: {row_data['ATR%']}%"
                )
            else:
                st.info("💡 該股未出現在本次選股名單中，暫無籌碼與評分數據。")

        with tab3:
            # Fix #9: 筆記自動持久化到 notes.json，頁面重整不遺失
            st.subheader("📝 交易筆記")
            st.caption("內容自動儲存，重新整理頁面後不會遺失。")
            st.session_state['_note_editing_sid'] = sid    # 讓回呼知道在編哪檔
            st.text_area(
                f"紀錄對 {sid} {sname} 的看法…",
                key=f"note_{sid}",
                on_change=_note_on_change_cb,
                height=200,
            )
            if st.button("💾 手動儲存", key=f"save_note_{sid}"):
                save_note(sid, st.session_state.get(f"note_{sid}", ""))
                st.success("已儲存！")
