"""Streamlit UI for screening0515.py

執行方式:
    streamlit run screening_ui16.py
"""

# ── 頂層 import ───────────────────────────────────────────────────────────
import os
import json
import tempfile
from pathlib import Path

import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from google import genai  # 🧠 新增：導入最新版 Gemini 套件

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

# 🧠 新增：連線到系統保險箱讀取 Gemini 金鑰 (使用最新寫法)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    ai_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    ai_client = None

# ── 自選股與交易筆記持久化 ────────────────────────────────────────────────
# 🐛 已修正：路徑改為 cache 資料夾，確保網頁與 Telegram 小助理資料完全同步！
WATCHLIST_FILE = "cache/watchlist.json"
NOTES_FILE = "cache/notes.json"

def load_watchlist() -> list:
    if Path(WATCHLIST_FILE).exists():
        with open(WATCHLIST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_watchlist(wl: list) -> None:
    # 確保 cache 資料夾存在
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
    """Widget 內容變動時直接將新文字寫入記憶體與 json 檔案中"""
    val = st.session_state.get(f"note_input_{sid}", "")
    st.session_state.trading_notes[sid] = val
    save_note(sid, val)

# ── 資料快取讀取 ───────────────────────────────────────────────────────────
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
if _cache_date is None:
    st.error("⚠ 找不到 daily 快取，請先執行 `python fetch_cache.py` 拉資料")
else:
    _today = pd.Timestamp.now(tz="Asia/Taipei").replace(tzinfo=None).normalize()
    _age        = (_today - _cache_date.normalize()).days
    _date_str   = _cache_date.strftime('%Y-%m-%d')
    _is_weekend = _today.weekday() >= 5  # 5=週六, 6=週日

    # 增加一個「收盤後」的判斷 (15:00 之後才算真正的今天)
    now_hour = pd.Timestamp.now(tz="Asia/Taipei").hour    
    if _age == 0:
        if now_hour < 15:
            st.info(f"📅 目前資料為 {_date_str}。今日盤後數據預計 15:30 自動更新。")
        else:
            st.success(f"📅 cache 最新日期: {_date_str} (今日最新數據)")           
    elif _is_weekend and _age <= 2:
        st.success(f"📅 cache 最新日期: {_date_str} (週末未開盤，此已為最新交易日)")
    elif _age <= 5:
        st.info(f"📅 cache 最新日期: {_date_str} ({_age} 天前 - 註：若遇國定假日未開盤，此即為最新資料)")
    else:
        st.error(f"📅 cache 最新日期: {_date_str} ({_age} 天前) ⚠ 過舊，請先更新")

    # 雲端更新按鈕區塊
    st.markdown("#### 🔄 資料更新") 
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

# ── 策略邏輯導覽 (FAQ) ─────────────────────────────────────────────────────
with st.expander("💡 策略邏輯導覽 (FAQ) — 核心量化規則說明", expanded=False):
    st.markdown("""
    ### 🎯 第一區：快速套用組合
    不知道參數怎麼調？這裡為你準備了四套大腦：
    * **預設 (Default)**：系統原始設定（門檻 7 分），適合一般盤勢下的穩健選股。
    * **多頭寬鬆 (Bull)**：過關門檻降為 6 分，放寬 KD 觀察視窗。適合大盤**強勢上漲**時，避免因條件過於嚴苛而漏掉提早發動的潛力股。
    * **空頭嚴格 (Bear)**：過關門檻提升至 8 分，縮短 KD 視窗並要求高流動性。適合大盤**大跌或震盪**時，嚴格挑選籌碼最集中、最抗跌的防禦標的。
    * **KD 起漲 (KD Start)**：專注於尋找「低檔剛發生黃金交叉」的技術面訊號。適合用來抓跌深反彈，或是波段起漲的第一根轉折點。

    ---

    ### 📌 第二區：核心 / KD (計分與技術面)
    * **過關門檻 (PASS_SCORE)**：滿分 10 分。計分包含：法人買超、大戶增散戶減、資減券增、技術突破、KD金叉、營收成長與大盤相對強弱。
        * 🛡️ **動態防禦機制**：若大盤跌破季線，系統會**自動將門檻 +1 分**（空頭從嚴），幫你過濾掉崩盤時容易補跌的弱勢股。
    * **KD 觀察區間**：往回看 N 天，尋找「曾經」發生低檔金叉的股票。設短專抓剛發動，設長容許起漲後稍微休息的股票。
    * **KD 低檔啟動門檻**：金叉當天的 K 值必須小於此數值，確保這是一檔「從谷底翻揚」的股票，越低越嚴格。
    * **KD 今日上限**：如果今天的 K 值已經大於此數值，代表短線已過熱（超買），即便曾低檔起漲也會被剔除，避免追高被套牢。

    ---

    ### 📌 第三區：預篩 / 法人 (基本保護與籌碼追蹤)
    * **20 日均量與 ATR% 上限**：第一線防護網。均量過濾掉買得到賣不掉的「冷門股」；ATR% 則過濾掉每天上沖下洗、過度投機的「妖股」。
    * **法人最少買超日（累計 vs 連續）**：
        * 💡 **這是一個超大重點**：本系統計算的是**「累計天數」**而非「嚴格連續天數」。
        * 例如觀察 7 天、最少買超 5 天，代表只要這 7 天內有任何 5 天外資或投信站在買方（且總淨額為正）即成立。這能有效包容法人在吃貨過程中「進三退一」的洗盤動作，避免因單日調節而錯失波段飆股。
    * **什麼是「★籌碼共振」？**
        * 當同週期內觸發「大戶持股上升（主力吃貨）」且「散戶持股下降（籌碼沉澱）」，即達成共振。
        * 雖然不額外加分，但它是本系統**最高優先級的排序基準**！總分相同時，共振成立（✅）的股票會優先排在最上方。

    ---

    ### 🔄 第四區：資料更新與系統狀態
    * **為什麼週末點擊「抓取最新資料」，日期沒有變成今天？**
        * 台股在週末與國定假日是不開盤的。如果今天是週六，最新交易日停留在「本週五」是完全正確的狀態。此時畫面的提示會智能轉為綠色，告訴你「週末未開盤，此已為最新交易日」，不需重複抓取。
    """)

# ── Session state 初始化 ────────────────────────────────────────────────────
DEFAULTS = {
    'pass_score': PASS_SCORE, 'lookback_days': LOOKBACK_DAYS, 'it_min_buy_days': IT_MIN_BUY_DAYS,
    'fi_min_buy_days': FI_MIN_BUY_DAYS, 'kd_lookback': KD_LOOKBACK, 'kd_low_from': KD_LOW_FROM,
    'kd_high_cap_now': KD_HIGH_CAP_NOW, 'min_avg_vol_lots': MIN_AVG_VOL_LOTS, 'atr_max_pct': ATR_MAX_PCT,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

_state_defaults = {
    'result_df': None, 'result_files': {}, 'result_meta': None, 'target_sid': None, 'last_df_selection': [],
}
for k, v in _state_defaults.items():
    if k not in st.session_state: st.session_state[k] = v

if 'trading_notes' not in st.session_state: st.session_state.trading_notes = load_notes()
if 'watchlist' not in st.session_state: st.session_state.watchlist = load_watchlist()
if 'ai_cache' not in st.session_state: st.session_state.ai_cache = {}  # 🧠 新增：用來記住每一檔股票算出來的 AI 評語

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
    st.slider(labeled("過關門檻 PASS_SCORE", 'pass_score'), 1, 10, key='pass_score',
              help="滿分 10 分，達標才算入榜。\n\n計分包含：法人買超、大戶增散戶減、資減券增、技術突破、KD金叉、營收成長與大盤相對強弱。\n\n💡 系統會智能判斷：若大盤破季線會自動 +1 分（空頭從嚴）。")
    st.slider(labeled("KD 觀察區間 (天)", 'kd_lookback'), 1, 30, key='kd_lookback',
              help="往回看 N 天，尋找『曾經』發生低檔金叉的股票。\n\n- 設短一點：專抓剛發動的。\n- 設長一點：容許起漲後稍微休息的股票。")
    st.slider(labeled("KD 低檔啟動門檻", 'kd_low_from'), 10, 50, key='kd_low_from',
              help="金叉當天 K 值必須 < 此數值。\n\n用來確保這是一檔『從谷底翻揚』的股票，而不是在高檔糾纏。數值越低越嚴格。")
    st.slider(labeled("KD 今日上限", 'kd_high_cap_now'), 50, 95, key='kd_high_cap_now',
              help="如果今天的 K 值已經超過此數值，代表短線已過熱（超買）。\n\n即使它前幾天曾低檔起漲也會被剔除，避免追高被套牢。")

    st.divider()
    st.subheader("📌 預篩 / 法人")
    st.slider(labeled("20 日均量下限 (張)", 'min_avg_vol_lots'), 100, 2000, key='min_avg_vol_lots', step=50,
              help="基本流動性保護。\n\n如果設定 500，代表過去一個月平均每天成交不到 500 張的股票會直接被丟掉，避免買得到賣不掉的冷門股。")
    st.slider(labeled("ATR% 上限", 'atr_max_pct'), 5.0, 25.0, key='atr_max_pct', step=0.5,
              help="用來過濾波動極端的『妖股』。\n\n設定 10% 代表該股每天上沖下洗的振幅約在 10% 內。超過此值視為過度投機，直接剔除保平安。")
    st.slider(labeled("法人觀察天數", 'lookback_days'), 3, 10, key='lookback_days',
              help="我們往回看法人籌碼的『總天數區間』。\n\n這個區間就是底下『最少買超日』的統計母體。")
    st.slider(labeled("投信最少買超日", 'it_min_buy_days'), 1, 5, key='it_min_buy_days',
              help="💡 注意：這是『累計』而非連續！\n\n例如觀察 7 天、最少 5 天，代表只要這 7 天內有 5 天買超且總淨額為正即可。這樣能包容投信『進三退一』的洗盤手法。")
    st.slider(labeled("外資最少買超日", 'fi_min_buy_days'), 1, 5, key='fi_min_buy_days',
              help="💡 注意：這是『累計』而非連續！\n\n代表在觀察天數內，總共有幾天買超且總淨額為正。這能容許外資偶爾單日大賣調節，只要整體趨勢站在買方就能抓出。")

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

# ── 大盤狀態與選股執行 ─────────────────────────────────────────────────────
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

# ── K 線與技術圖表區 ────────────────────────────────────────────────────────
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
        
        # 🧠 修改處：新增第四個 AI 分頁
        tab1, tab2, tab3, tab4 = st.tabs(["📈 技術分析", "📊 籌碼/基本面", "🧠 AI 虛擬點評", "📝 交易筆記"])
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

                # 法人買賣超數據整合
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

                # 4 層子圖 (技術/成交量/法人/KD)
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
                
                # 新股防護：MA20 NaN 防呆
                ma20_last = hist['MA20'].iloc[-1]
                defense_y = hist['min'].tail(10).min() if pd.isna(ma20_last) else max(ma20_last * 0.98, hist['min'].tail(10).min())
                fig.add_hline(y=defense_y, line_dash="dash", line_color="red", annotation_text="🚨 防守", annotation_position="bottom right", row=1, col=1)

                if row_data is not None:
                    fig.add_annotation(x=hist['date'].iloc[-1], y=hist['min'].iloc[-1], text="🔥 訊號觸發", showarrow=True, arrowhead=1, arrowcolor="red", ay=30, row=1, col=1)

                # 第二層：成交量
                vol_colors = ['red' if c >= o else 'green' for c, o in zip(hist['close'], hist['open'])]
                fig.add_trace(go.Bar(x=hist['date'], y=hist[vol_col], marker_color=vol_colors, name='成交量'), row=2, col=1)

                # 第三層：法人買賣超柱狀圖
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
                    hovermode='x unified', showlegend=False, barmode='group' 
                )
                fig.update_xaxes(gridcolor=grid_color, rangebreaks=[dict(values=missing_dates)], type="date")
                
                # Plotly rangeselector dict 修正
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
                
                fig.update_yaxes(gridcolor=grid_color, side='right')
                fig.update_yaxes(fixedrange=False, row=1, col=1)

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

        # 🧠 新增：第三個分頁的 AI 深度診斷邏輯 (OpenRouter 免綁卡版)
        with tab3:
            st.subheader("🧠 OpenRouter AI 虛擬操盤分析師")
            
            # 從環境變數或 Streamlit Secrets 讀取 OpenRouter Key
            OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
            
            if not OPENROUTER_API_KEY:
                st.warning("⚠️ 網頁系統未偵測到環境變數 `OPENROUTER_API_KEY`。請先在 Streamlit Cloud Secrets 設定金鑰！")
            elif row_data is None:
                st.info("💡 該股未出現在本次排行榜中（未達過關門檻），暫無量化核心數據供 AI 進行深度點評。")
            else:
                if sid in st.session_state.ai_cache:
                    st.markdown("### 📋 歷史診斷報告記憶檔")
                    st.info(st.session_state.ai_cache[sid])
                    if st.button("🔄 重新生成最新點評", key=f"re_ai_{sid}"):
                        del st.session_state.ai_cache[sid]
                        st.rerun()
                else:
                    st.caption("點擊下方按鈕將會把本系統算出的 10 大雷達數據餵給免綁卡的 OpenRouter 免費模型。")
                    if st.button("🚀 啟動 AI 深度量化解析", key=f"btn_ai_{sid}", use_container_width=True):
                        with st.spinner("AI 分析師正在閱讀籌碼量化數據中，請稍候..."):
                            try:
                                import requests
                                prompt = f"""
                                你是一位台灣股市的資深量化操盤分析師。請根據以下這檔個股由系統算出的核心籌碼與技術數據，用繁體中文寫出一份約 150 字的專業投資診斷。
                                一針見血指出當前籌碼面（法人大戶）與技術基本面的優缺點，並給出具體的短線操盤與策略建議（多空方向）。
                                請不要使用 Markdown 語法中的星號 (**) 加粗，以純文字順暢表達即可。
                                
                                【個股數據報告】
                                股票標的：{sid} {sname}
                                當前股價：{row_data['現價']} 元
                                綜合量化總分：{row_data['總分']} / 10 分
                                產業板塊：{row_data['產業']}
                                20日均成交量：{row_data['20日均量(張)']} 張
                                歷史波動度 (ATR%)：{row_data['ATR%']}%
                                三大法人5日籌碼：投信總計 {row_data['投信5日淨額(張)']} 張 / 外資總計 {row_data['外資5日淨額(張)']} 張
                                主力與散戶狀態：{'主力大戶吃貨且散戶退場（籌碼共振成立）' if row_data['★籌碼共振(大戶↑散戶↓)'] == 1 else '籌碼尚未集中（共振未成立）'}
                                最新營收表現：年增/月增率 {row_data['最新月營收增率(%)']}% (採用模式: {row_data['營收模式']})
                                相對大盤強度 (RS)：{'表現超越大盤（多頭領頭羊）' if row_data['RS優於大盤'] == 1 else '弱於大盤走勢'}
                                """
                                
                                headers = {
                                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                                    "Content-Type": "application/json"
                                }
                                
                                # 🎯 1. 換成速度極快、不塞車的 Llama 3.3 大腦
                                payload = {
                                    "model": "qwen/qwen-2.5-7b-instruct:free",  # 🎯 換成 7B 版本，這個通道幾乎永遠暢通
                                    "messages": [{"role": "user", "content": prompt}]
                                }
                                
                                resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20)
                                resp_json = resp.json()
                                
                                # 🛡️ 2. 鋼鐵防禦機制：有拿到 choices 才解包，沒拿到就印出真實錯誤
                                if 'choices' in resp_json:
                                    ai_result = resp_json['choices'][0]['message']['content'].strip()
                                    st.session_state.ai_cache[sid] = ai_result
                                    st.rerun()
                                elif 'error' in resp_json:
                                    st.error(f"❌ OpenRouter 拒絕請求原因：{resp_json['error'].get('message')}")
                                else:
                                    st.error(f"❌ 發生未知異常，完整回應內容：{resp_json}")
                                    
                            except Exception as e:
                                st.error(f"❌ 呼叫 OpenRouter AI 發生異常：{str(e)}")

        with tab4:
            st.subheader("📝 交易筆記")
            st.caption("內容自動儲存，切換股票或刷新頁面後皆不會遺失。")
            
            st.text_area(
                f"紀錄對 {sid} {sname} 的看法…",
                value=st.session_state.trading_notes.get(sid, ""),
                key=f"note_input_{sid}",
                on_change=_note_on_change_cb,
                args=(sid,),
                height=180,
            )
            
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