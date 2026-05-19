"""Streamlit UI for screening0515.py

執行方式:
    streamlit run screening_ui16.py
"""

# ── 頂層 import ───────────────────────────────────────────────────────────
import os
import sys
import json
import re
import time
import subprocess
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
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

# 共用模組(與 telegram_notify.py 共用同一份邏輯)
from ai_helper import call_openrouter_ai, AI_MODELS, get_api_key
from cache_status import cache_freshness
from picks_history import load_history, compute_hot_picks
from data_health import check_data_health
from industry_rotation import compute_industry_rotation
from performance import compute_performance
from backtest import run_backtest, SIGNAL_LABELS

# ── 自選股與交易筆記持久化 ────────────────────────────────────────────────
# 🐛 已修正：路徑改為 cache 資料夾，確保網頁與 Telegram 小助理資料完全同步！
WATCHLIST_FILE = "cache/watchlist.json"
NOTES_FILE = "cache/notes.json"

def load_watchlist() -> list:
    """讀取自選股。優先順序:URL query param > 檔案 > 空清單。
    Streamlit Cloud 檔案系統 reboot 會被洗掉,改用 query_param 跨 reboot 持久化;
    使用者只要書籤 URL 就能保留自選股(URL 不會超過上限,幾十支股票完全沒問題)。
    """
    try:
        qp_wl = st.query_params.get("wl", "")
        if qp_wl:
            return [s.strip() for s in qp_wl.split(",") if s.strip()]
    except Exception:
        pass
    if Path(WATCHLIST_FILE).exists():
        try:
            with open(WATCHLIST_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_watchlist(wl: list) -> None:
    """雙寫:URL query param(跨 reboot 持久化) + 檔案(session 加速)。"""
    try:
        if wl:
            st.query_params["wl"] = ",".join(str(s) for s in wl)
        elif "wl" in st.query_params:
            del st.query_params["wl"]
    except Exception:
        pass
    Path("cache").mkdir(exist_ok=True)
    with open(WATCHLIST_FILE, 'w', encoding='utf-8') as f:
        json.dump(wl, f, ensure_ascii=False, indent=2)


# ── AI 進場節奏標籤 ────────────────────────────────────────────────────
# AI 在點評最後輸出 [節奏] 可進場 / 拉回再進 / 觀察
# 用 regex 抽出來與正文分離,UI 顯示為彩色 pill。
def extract_verdict(ai_text: str):
    """從 AI 回應拆出 (body, verdict)。verdict 為 '可進場'/'拉回再進'/'觀察' 或 None。"""
    if not ai_text:
        return ai_text, None
    m = re.search(r"\[節奏\]\s*(可進場|拉回再進|觀察)", ai_text)
    verdict = m.group(1) if m else None
    body = re.sub(r"\n?\s*\[節奏\][^\n]*", "", ai_text).strip()
    return body, verdict


# 進場節奏 → 顏色/icon 對映(台股紅綠慣例:可進場用紅、觀察用灰、拉回再進用黃)
VERDICT_STYLE = {
    "可進場":   {"icon": "🟢", "color": "#16a34a", "bg": "rgba(22,163,74,0.10)",  "label": "可進場"},
    "拉回再進": {"icon": "🟡", "color": "#ca8a04", "bg": "rgba(202,138,4,0.10)",  "label": "拉回再進"},
    "觀察":     {"icon": "🔵", "color": "#2563eb", "bg": "rgba(37,99,235,0.10)",  "label": "觀察"},
}


def render_verdict_pill(verdict: str) -> None:
    """渲染進場節奏 pill(用 markdown HTML)。"""
    if verdict not in VERDICT_STYLE:
        return
    s = VERDICT_STYLE[verdict]
    st.markdown(
        f"""<div style="
            display: inline-block;
            background: {s['bg']};
            color: {s['color']};
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
            margin: 4px 0;
            border: 1px solid {s['color']};
        ">{s['icon']} 進場節奏 · {s['label']}</div>""",
        unsafe_allow_html=True
    )


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

@st.cache_data(ttl=300, show_spinner=False)
def _cached_stock_history(sid: str, n_days: int = 200) -> pd.DataFrame:
    """包一層 @st.cache_data，同一檔股票在同一次 session 只讀一次 parquet。"""
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
# Level → Streamlit 顯示元件對映(顏色一致由 cache_status 控制)
_level_fn = {
    "missing": st.error,
    "ok":      st.success,
    "info":    st.info,
    "warn":    st.warning,
    "error":   st.error,
}
_level_fn[_freshness["level"]](f"📅 {_freshness['msg']}")

if _freshness["level"] != "missing":

    # 雲端更新按鈕區塊
    st.markdown("#### 🔄 資料更新") 
    st.caption("雲端環境專用:點擊後從網路抓取最新台股資料。")

    # ── 顯示 daily 快取的「實際抓取時間」(mtime),讓使用者判斷需不需要重抓 ──
    try:
        daily_files = sorted(CACHE_DIR.glob('daily_*.parquet'))
        if daily_files:
            _daily_mtime = datetime.fromtimestamp(daily_files[-1].stat().st_mtime)
            st.caption(f"📅 日 K 最後抓取:**{_daily_mtime.strftime('%m/%d %H:%M')}**")
    except Exception:
        pass

    # ── 盤中提醒:< 14:00 抓的是即時價,警告使用者 ──
    _now = datetime.now()
    _is_market_hours = (
        _now.weekday() < 5 and
        ((_now.hour == 9 and _now.minute >= 0) or
         (10 <= _now.hour < 13) or
         (_now.hour == 13 and _now.minute <= 30))
    )
    if _is_market_hours:
        st.warning(
            "⏰ **目前盤中**,抓的會是『即時價』而非『收盤價』。\n"
            "**建議 14:00 後再強制重抓**,才會拿到完整收盤資料。"
        )

    if st.session_state.get('show_update_success'):
        st.toast("✅ 雲端資料檢查/更新完成!", icon="🎉")
        st.session_state.show_update_success = False
    if st.session_state.get('show_force_daily_success'):
        st.toast("✅ 日 K 強制更新完成!", icon="🔄")
        st.session_state.show_force_daily_success = False

    # ── 兩顆按鈕並排顯示(左:標準抓取 / 右:強制更新日 K) ──
    FORCE_DAILY_COOLDOWN = 120  # 2 分鐘
    _last_force_ts = st.session_state.get('last_force_daily_ts', 0)
    _elapsed = time.time() - _last_force_ts
    _on_cooldown = _elapsed < FORCE_DAILY_COOLDOWN

    _btn_col1, _btn_col2 = st.columns(2)

    with _btn_col1:
        _click_fetch = st.button(
            "📥 抓取今日最新資料", type="secondary", use_container_width=True,
            help="標準模式:若各類資源今天已抓過就略過。預設排程跑的就是這個。"
        )

    with _btn_col2:
        if _on_cooldown:
            _remaining = int(FORCE_DAILY_COOLDOWN - _elapsed)
            st.button(
                f"🔄 強制更新日 K(冷卻 {_remaining} 秒)",
                disabled=True, use_container_width=True
            )
            _click_force = False
        else:
            _click_force = st.button(
                "🔄 強制更新日 K", type="primary", use_container_width=True,
                help=(
                    "無視當日 cache,重新抓取**所有股票**的日 K 資料。\n\n"
                    "**用途**:盤中已經跑過、要拿到真正收盤價時(14:00 後再按)。\n"
                    "**只重抓 daily**:法人/融資券/營收/大戶 仍沿用今日 cache,不浪費 API 額度。\n"
                    "**冷卻**:2 分鐘內只能按一次。"
                )
            )

    # ── 按鈕 1:標準抓取(略過已有當日 cache) ──
    if _click_fetch:
        st.toast("⏳ 系統已收到請求,開始比對資料...", icon="🤖")
        with st.spinner("正在執行資料更新... (若轉圈超過 2 分鐘,可能是主機記憶體不足崩潰)"):
            try:
                result = subprocess.run([sys.executable, "fetch_cache.py"],
                                        capture_output=True, text=True, check=True)
                st.cache_data.clear()
                st.session_state.show_update_success = True
                st.rerun()
            except subprocess.CalledProcessError as e:
                st.error("❌ 雲端腳本執行失敗!")
                st.code(e.stderr)
            except Exception as e:
                st.error(f"❌ 發生未知的錯誤:{e}")

    # ── 按鈕 2:強制更新日 K(無視 cache,只重抓 daily) ──
    if _click_force:
        with st.status("正在強制重抓 daily K 線...", expanded=True) as _status:
            try:
                st.write("⏳ 啟動 `fetch_cache.py --force-daily`,預估 1~3 分鐘...")
                result = subprocess.run(
                    [sys.executable, "fetch_cache.py", "--force-daily"],
                    capture_output=True, text=True, timeout=600
                )
                if result.returncode == 0:
                    # 只顯示最後 8 行 log,避免訊息太長
                    _last_lines = result.stdout.strip().split('\n')[-8:]
                    st.code('\n'.join(_last_lines))
                    st.session_state.last_force_daily_ts = time.time()
                    st.session_state.show_force_daily_success = True
                    # 清掉所有 cache_data,讓 UI 重新讀新檔
                    st.cache_data.clear()
                    _status.update(label="✅ daily K 線已更新", state="complete")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ 抓取失敗(returncode={result.returncode})")
                    st.code(result.stderr[-500:] if result.stderr else "(無錯誤輸出)")
                    _status.update(label="❌ 失敗", state="error")
            except subprocess.TimeoutExpired:
                st.error("⏱️ 抓取逾時(超過 10 分鐘),已中止。可能是 yfinance 異常。")
                _status.update(label="❌ 逾時", state="error")
            except Exception as e:
                st.error(f"❌ 例外:{e}")
                _status.update(label="❌ 例外", state="error")

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
    # ── 裝置模式切換(影響欄位排列與圖表高度) ──
    if 'mobile_mode' not in st.session_state:
        st.session_state.mobile_mode = False
    st.session_state.mobile_mode = st.toggle(
        "📱 手機模式",
        value=st.session_state.mobile_mode,
        help="手機模式:列表與圖表改為上下排列、K 線圖縮小、隱藏次要欄位"
    )
    st.divider()

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
    bullish      = meta.get('market_bullish', True)
    twii_now_raw = meta.get('twii_now')
    twii_ma_raw  = meta.get('twii_ma')
    change_raw   = meta.get('twii_lookback_change')
    # Bug B 修正：用 pd.notna 同時擋 None 與 NaN
    change       = float(change_raw) if pd.notna(change_raw) else 0.0
    state_txt    = "📈 多頭(站上季線)" if bullish else "📉 空頭(跌破季線)"
    twii_now_s   = f"{twii_now_raw:,.0f}" if pd.notna(twii_now_raw) else "N/A"
    twii_ma_s    = f"{twii_ma_raw:,.0f}"  if pd.notna(twii_ma_raw)  else "N/A"
    if base is None or eff is None:
        thr_txt = "過關門檻未知"
    elif base == eff:
        thr_txt = f"過關門檻 {base}"
    else:
        thr_txt = f"過關門檻 {base} → **{eff}**"
    msg = f"{state_txt}  |  {thr_txt}  |  TWII {twii_now_s} / MA60 {twii_ma_s} | 近 20 日 {change:+.2f}%"
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

# ── 7 日入選熱度榜(資料來自 cache/previous_picks.json,由 Telegram 每日推播寫入) ──
@st.cache_data(ttl=300, show_spinner=False)
def _load_hot_picks_cached(top_n: int = 10):
    """快取 5 分鐘,避免每次 rerun 都重讀 JSON。"""
    hist = load_history()
    return compute_hot_picks(hist, top_n=top_n), len(hist)

_hot, _hist_days = _load_hot_picks_cached(top_n=10)
if _hot:
    with st.expander(f"🔥 過去 {_hist_days} 日入選熱度榜(TOP 10)", expanded=False):
        st.caption("追蹤近期持續上榜的強勢股 — 連續出現次數越多,代表趨勢延續性越強。")
        # 用 dataframe 顯示,讓使用者可以排序與複製
        hot_rows = []
        for r in _hot:
            sid_str = str(r["sid"])
            name = ui_name_map.get(sid_str, "")
            hot_rows.append({
                "★今日": "✅" if r["in_latest"] else "",
                "代號":  sid_str,
                "名稱":  name,
                "入選天數":   f"{r['hits']} / {r['total_days']}",
                "最長連續":   f"{r['max_streak']} 日",
                "目前連續":   f"{r['active_streak']} 日" if r['active_streak'] >= 1 else "—",
            })
        st.dataframe(pd.DataFrame(hot_rows), use_container_width=True, hide_index=True)


# ── 產業輪動追蹤(對比最近 7 日 vs 前 7 日各產業上榜次數) ──────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _load_industry_rotation_cached():
    hist = load_history()
    return compute_industry_rotation(hist, recent_days=7, prev_days=7)

_rotation = _load_industry_rotation_cached()
if _rotation:
    with st.expander("🔄 產業輪動追蹤(近 7 日 vs 前 7 日)", expanded=False):
        st.caption("資金流向觀察 — 上榜次數驟增的產業代表主流轉換,可加重押注;驟減則注意避開。")
        rot_rows = []
        for r in _rotation[:10]:
            arrow = {"up": "🔺", "down": "🔻", "flat": "—"}[r["direction"]]
            change_str = f"{r['change']:+d}" if r["change"] != 0 else "0"
            rot_rows.append({
                "產業": r["industry"],
                "近 7 日": r["recent_count"],
                "前 7 日": r["prev_count"],
                "變化": f"{arrow} {change_str}",
            })
        st.dataframe(pd.DataFrame(rot_rows), use_container_width=True, hide_index=True)


# ── 策略績效追蹤(對歷史 picks 算入選後 N 日報酬) ───────────────────────
@st.cache_data(ttl=600, show_spinner="計算策略績效中…")
def _load_performance_cached():
    hist = load_history()
    return compute_performance(hist, CACHE_DIR, n_days_list=(5, 10, 20))

if _hist_days >= 5:  # 至少有 5 天歷史才有意義
    with st.expander("📊 策略績效追蹤(過去入選後續報酬)", expanded=False):
        st.caption("回答「我這套系統真的有用嗎?」— 對每筆歷史選股,從 daily 快取算出後續 N 日報酬。")
        perf = _load_performance_cached()
        overall = perf.get("overall", {})
        if not overall:
            st.info("資料尚不足以計算績效,建議累積更多天數的選股紀錄後再來看。")
        else:
            # 三檔指標卡片
            for n_days in (5, 10, 20):
                key_n   = f"n_{n_days}d"
                key_win = f"win_rate_{n_days}d"
                key_avg = f"avg_return_{n_days}d"
                if key_n not in overall:
                    continue
                st.markdown(f"**入選後 {n_days} 個交易日**")
                m_cols = st.columns(4)
                m_cols[0].metric("樣本", f"{overall[key_n]} 筆")
                m_cols[1].metric("勝率", f"{overall[key_win]*100:.0f}%")
                avg_ret = overall[key_avg]
                m_cols[2].metric("平均報酬", f"{avg_ret:+.2f}%",
                                 delta_color="normal" if avg_ret >= 0 else "inverse")
                med_ret = overall.get(f"median_return_{n_days}d", 0)
                m_cols[3].metric("中位數", f"{med_ret:+.2f}%")

            # 分數區間表
            by_score = perf.get("by_score", {})
            if by_score:
                st.divider()
                st.markdown("**各分數區間 5 日勝率**")
                rows = []
                for score in sorted(by_score.keys(), reverse=True):
                    s = by_score[score]
                    rows.append({
                        "分數": f"{score} 分",
                        "樣本數": s.get("n_5d", 0),
                        "勝率":   f"{s.get('win_rate_5d', 0)*100:.0f}%" if "win_rate_5d" in s else "—",
                        "平均報酬": f"{s.get('avg_return_5d', 0):+.2f}%" if "avg_return_5d" in s else "—",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── 訊號回測:對 daily/法人 parquet 掃描三大技術訊號歷史報酬 ─────────────
@st.cache_data(ttl=900, show_spinner="跑回測中(掃描 180 天歷史訊號)…")
def _run_backtest_cached(signals_tuple: tuple, hold_days: int, date_filter: str, combine_mode: str):
    """signals_tuple: 用 tuple 才能被 cache_data hash。"""
    if date_filter == "all":
        date_range = None
    else:
        ndays = 90 if date_filter == "90d" else 30
        end = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None).normalize()
        start = end - pd.Timedelta(days=ndays)
        date_range = (start, end)
    return run_backtest(CACHE_DIR, signal=list(signals_tuple), hold_days=hold_days,
                        date_range=date_range, combine_mode=combine_mode)


with st.expander("🔬 訊號回測(過去 180 天歷史掃描)", expanded=False):
    st.caption(
        "對 daily 快取掃描 5 種訊號的歷史觸發點,算「進場後 N 日報酬」── "
        "回答「哪個訊號真的有 alpha?該在計分系統加重?」可複選做組合測試。"
    )

    sig_options = list(SIGNAL_LABELS.keys())
    sig_choice_multi = st.multiselect(
        "選擇訊號(可複選)", sig_options,
        default=["breakout"],
        format_func=lambda k: SIGNAL_LABELS[k],
        key="bt_signals",
        help="多選時依「合併模式」交集 / 聯集 — 例:選『外資+投信』模式 AND = 兩家同日都買超才算"
    )

    bc1, bc2, bc3 = st.columns(3)
    combine_mode_choice = bc1.radio(
        "合併模式", ["and", "or"], horizontal=True, key="bt_combine",
        format_func=lambda k: "AND 交集(都觸發)" if k == "and" else "OR 聯集(任一觸發)",
        help="只選 1 個訊號時兩者效果一樣;選 2+ 才有意義"
    )
    hold_choice = bc2.selectbox(
        "持有天數", [5, 10, 20], index=1, key="bt_hold",
        help="進場後幾個交易日賣出"
    )
    period_choice = bc3.selectbox(
        "樣本期間",
        ["all", "90d", "30d"],
        format_func=lambda k: {"all": "全部(~180 日)", "90d": "近 90 日", "30d": "近 30 日"}[k],
        key="bt_period"
    )

    if not sig_choice_multi:
        st.info("👆 請至少勾選一個訊號才能跑回測")
        _bt = {"trades": pd.DataFrame(), "stats": {"n": 0}, "all_signals_stats": {}}
    else:
        _bt = _run_backtest_cached(tuple(sig_choice_multi), hold_choice, period_choice, combine_mode_choice)
    # 下方明細表/CSV 命名用的單一 label
    sig_choice = "_".join(sig_choice_multi) + ("_AND" if combine_mode_choice == "and" else "_OR") if sig_choice_multi else "none"

    if _bt.get("error"):
        st.error(f"⚠️ {_bt['error']}")
    elif _bt['stats'].get('n', 0) == 0:
        st.info("此訊號在所選期間內無觸發紀錄。試試擴大期間或換訊號。")
    else:
        stats = _bt['stats']
        # ── 4 張指標卡 ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("樣本數", f"{stats['n']:,}", "次觸發")
        # 台股慣例:勝率紅色、平均報酬看正負決定紅綠
        m2.metric("勝率", f"{stats['win_rate']*100:.0f}%",
                  f"{int(stats['win_rate']*stats['n'])} 賺")
        avg = stats['avg_return']
        # 用 delta 顯示中位數;台股紅漲綠跌,但 streamlit metric 預設綠正紅負,改 inverse 反轉
        m3.metric("平均報酬", f"{avg:+.2f}%",
                  f"中位數 {stats['median_return']:+.2f}%",
                  delta_color="inverse")
        m4.metric("最差單筆", f"{stats['min_return']:+.2f}%",
                  f"最佳 {stats['max_return']:+.2f}%",
                  delta_color="inverse")

        # ── 三訊號對照圖 ──
        st.divider()
        st.markdown(f"**三訊號 {hold_choice} 日勝率 / 平均報酬對照**")
        st.caption("紅色 = 勝率(左軸,%) / 灰色 = 平均報酬(右軸,%)")

        all_stats = _bt['all_signals_stats']
        # 整理成 DataFrame 給 plotly 雙軸
        chart_data = []
        for sig_key in sig_options:
            s = all_stats.get(sig_key, {"n": 0})
            if s.get('n', 0) > 0:
                chart_data.append({
                    "訊號": SIGNAL_LABELS[sig_key].split('(')[0],  # 簡短名稱
                    "勝率": s['win_rate'] * 100,
                    "平均報酬": s['avg_return'],
                    "樣本數": s['n'],
                })
        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            fig_bt = make_subplots(specs=[[{"secondary_y": True}]])
            fig_bt.add_trace(
                go.Bar(name="勝率", x=chart_df['訊號'], y=chart_df['勝率'],
                       marker_color='#A32D2D',
                       text=[f"{v:.0f}%" for v in chart_df['勝率']],
                       textposition='outside'),
                secondary_y=False
            )
            fig_bt.add_trace(
                go.Bar(name="平均報酬", x=chart_df['訊號'], y=chart_df['平均報酬'],
                       marker_color='#888780',
                       text=[f"{v:+.1f}%" for v in chart_df['平均報酬']],
                       textposition='outside'),
                secondary_y=True
            )
            fig_bt.update_layout(
                barmode='group',
                height=320,
                margin=dict(l=10, r=10, t=20, b=10),
                showlegend=False,
                template=chart_theme if 'chart_theme' in dir() else 'plotly',
            )
            fig_bt.update_yaxes(title_text="勝率 (%)", secondary_y=False, range=[0, max(chart_df['勝率'].max()*1.15, 70)])
            fig_bt.update_yaxes(title_text="平均報酬 (%)", secondary_y=True)
            st.plotly_chart(fig_bt, use_container_width=True)

            # ── 提示:哪個訊號最強 ──
            best_wr = max(all_stats.items(),
                          key=lambda kv: kv[1].get('win_rate', 0) if kv[1].get('n', 0) > 0 else -1)
            best_ret = max(all_stats.items(),
                           key=lambda kv: kv[1].get('avg_return', -999) if kv[1].get('n', 0) > 0 else -999)
            if best_wr[1].get('n', 0) > 0:
                hints = []
                hints.append(f"勝率最高 — **{SIGNAL_LABELS[best_wr[0]].split('(')[0]}** ({best_wr[1]['win_rate']*100:.0f}%)")
                if best_ret[0] != best_wr[0]:
                    hints.append(f"報酬最高 — **{SIGNAL_LABELS[best_ret[0]].split('(')[0]}** ({best_ret[1]['avg_return']:+.2f}%)")
                st.info("💡 " + " / ".join(hints) + " — 可考慮在計分系統內加重對應訊號的權重。")

        # ── 觸發明細表 ──
        trades = _bt['trades']
        if not trades.empty:
            st.divider()
            top_n_bt = min(50, len(trades))
            st.markdown(f"**觸發明細(顯示前 {top_n_bt} 筆 / 共 {len(trades):,} 筆)**")

            display_df = trades.head(top_n_bt).copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df['名稱'] = display_df['stock_id'].map(ui_name_map).fillna('')
            display_df['進場'] = display_df['entry_close'].apply(lambda x: f"{x:,.1f}")
            display_df['出場'] = display_df['exit_close'].apply(lambda x: f"{x:,.1f}")
            # 台股紅漲綠跌的著色用 emoji 代替(streamlit dataframe 著色需 styler 較重)
            display_df['報酬'] = display_df['return_pct'].apply(
                lambda x: f"🔴 {x:+.2f}%" if x > 0.05 else (f"🟢 {x:+.2f}%" if x < -0.05 else f"⚪ {x:+.2f}%")
            )
            st.dataframe(
                display_df[['date', 'stock_id', '名稱', '進場', '出場', '報酬']]
                    .rename(columns={'date': '觸發日', 'stock_id': '代號'}),
                use_container_width=True, hide_index=True,
            )

            # CSV 下載
            csv_bytes = trades.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "📥 下載完整明細 CSV",
                csv_bytes,
                file_name=f"backtest_{sig_choice}_{hold_choice}d.csv",
                mime="text/csv",
                use_container_width=False,
            )


# ── 資料健康度警告(只在 warn/error 級才顯示) ──
@st.cache_data(ttl=300, show_spinner=False)
def _load_health_cached():
    return check_data_health(CACHE_DIR)

_health = _load_health_cached()
if _health.get("level") in ("warn", "error"):
    _msg_fn = st.error if _health["level"] == "error" else st.warning
    _msg_fn(f"⚠️ **{_health['summary']}** — " + " / ".join(_health.get("issues", [])[:3]))


st.divider()

# 手機模式:上下排列(列表在上、K 線在下);桌機模式:左右排列
_is_mobile = st.session_state.get('mobile_mode', False)
if _is_mobile:
    col_list = st.container()
    col_chart = st.container()
else:
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

        # 一鍵複製股號清單(套用篩選後,可直接貼進券商看盤軟體的群組建立)
        codes_csv = ",".join(filtered['代號'].astype(str).tolist())
        if codes_csv:
            st.caption(f"📋 已篩選 {len(filtered)} 檔 — 點右上角圖示一鍵複製,可貼進券商看盤軟體建立群組")
            st.code(codes_csv, language=None)

        current_sel = event.selection.rows
        if current_sel != st.session_state.last_df_selection:
            st.session_state.last_df_selection = current_sel
            if current_sel: st.session_state.target_sid = str(filtered.iloc[current_sel[0]]['代號'])

# ── K 線與技術圖表區 ────────────────────────────────────────────────────────
# 若有選股結果但 target_sid 還沒設,預設指向第一檔(總分最高),讓使用者一進來就有東西看
if (not st.session_state.target_sid) and (df is not None) and (len(df) > 0):
    st.session_state.target_sid = str(df.iloc[0]['代號'])

with col_chart:
    if not st.session_state.target_sid:
        st.info("👈 請點擊左側列表或自選股，開始量化決策分析。")
    else:
        sid = str(st.session_state.target_sid)
        row_data = None
        if df is not None and len(df) > 0:
            if sid in df['代號'].astype(str).values:
                row_data = df[df['代號'].astype(str) == sid].iloc[0]

        # Issue 2 修正：名稱為 NaN 時 fallback 到 ui_name_map，避免顯示「nan」穿透到圖表/AI/筆記
        if row_data is not None:
            _raw_name = row_data['名稱']
            sname = str(_raw_name) if pd.notna(_raw_name) else ui_name_map.get(sid, "")
        else:
            sname = ui_name_map.get(sid, "")
        
        # 🧠 修改處：新增第四個 AI 分頁
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 技術分析", "📊 籌碼/基本面", "🧠 AI 虛擬點評", "📝 交易筆記", "💰 資金管理"
        ])
        hist_daily = _cached_stock_history(sid)

        with tab1:
            tcol1, tcol2, tcol3, tcol4 = st.columns([1.5, 1.5, 1.5, 1])
            tcol1.markdown(f"### {sid} {sname}")
            timeframe = tcol2.radio("週期", ["日K", "週K"], horizontal=True, label_visibility="collapsed")
            # 改用 Streamlit radio 控制縮放範圍，繞開 Plotly rangeselector + rangebreaks 的相容性問題
            zoom_choice = tcol3.radio("縮放", ["1月", "3月", "半年", "全部"], horizontal=True, index=3, label_visibility="collapsed", key=f"zoom_{sid}")
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
                # Bug 1 修正：_calc_kd_series 在歷史不足 10 筆時回傳 (None, None),
                # 不防護的話新股或剛恢復交易股會讓 list comprehension 拋 TypeError
                k_list, d_list = _calc_kd_series(hist['max'], hist['min'], hist['close'])
                if k_list is None:
                    hist['K'] = float('nan')
                    hist['D'] = float('nan')
                else:
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
                    # Issue 3 修正：週K 模式下實際是 60 週(~1.2 年)而非 60 日,正確標示避免誤導
                    pressure_label = "60週壓" if timeframe == "週K" else "60日壓"
                    fig.add_hline(y=hist['max'].iloc[-61:-1].max(), line_dash="dot", line_color="orange", annotation_text=pressure_label, annotation_position="top left", row=1, col=1)
                
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
                    height=550 if _is_mobile else 850,
                    xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10),
                    hovermode='x unified', showlegend=False, barmode='group' 
                )
                fig.update_xaxes(gridcolor=grid_color, rangebreaks=[dict(values=missing_dates)], type="date")

                # 用 Streamlit radio 控制縮放：直接設定 xaxis range，
                # 比 Plotly 內建 rangeselector 可靠（rangeselector + rangebreaks 在 Plotly 有已知 bug）
                if zoom_choice != "全部":
                    days_map = {"1月": 30, "3月": 90, "半年": 180}
                    end_date   = pd.to_datetime(hist['date'].iloc[-1])
                    start_date = end_date - pd.Timedelta(days=days_map[zoom_choice])
                    fig.update_xaxes(range=[start_date, end_date])
                
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

        # 🧠 第三個分頁的 AI 深度診斷邏輯(多模型可選 + 自動 fallback)
        with tab3:
            st.subheader("🧠 OpenRouter AI 虛擬操盤分析師")

            # 共用 ai_helper 的 key 讀取邏輯,確保「警告偵測」與「實際呼叫」用同一把 key
            _has_api_key = bool(get_api_key())

            if not _has_api_key:
                st.warning("⚠️ 網頁系統未偵測到環境變數 `OPENROUTER_API_KEY`。請先在 Streamlit Cloud Secrets 設定金鑰!")
            elif row_data is None:
                st.info("💡 該股未出現在本次排行榜中（未達過關門檻），暫無量化核心數據供 AI 進行深度點評。")
            else:
                # 模型選擇下拉選單：「自動輪替」優先，再列出個別模型
                model_options = ["自動輪替 (推薦)"] + [m["name"] for m in AI_MODELS]
                selected_model = st.selectbox(
                    "🤖 選擇 AI 模型",
                    model_options,
                    key=f"ai_model_{sid}",
                    help="選「自動輪替」會依序試 DeepSeek → Qwen → Gemini → Llama → GPT-OSS,第一個成功的就用。"
                )

                # cache 用 (sid, model) 當 key,換模型可重新生成而不必清舊的
                cache_key = (sid, selected_model)

                if cache_key in st.session_state.ai_cache:
                    cached_model, cached_text = st.session_state.ai_cache[cache_key]
                    st.markdown(f"### 📋 診斷報告({cached_model})")
                    body, verdict = extract_verdict(cached_text)
                    if verdict:
                        render_verdict_pill(verdict)
                    st.info(body if verdict else cached_text)
                    if st.button("🔄 重新生成", key=f"re_ai_{sid}_{selected_model}"):
                        del st.session_state.ai_cache[cache_key]
                        st.rerun()
                else:
                    st.caption("把系統算出的量化雷達數據與技術位階一起餵給免費模型,產生 ~150 字深度診斷。")
                    if st.button("🚀 啟動 AI 深度量化解析", key=f"btn_ai_{sid}_{selected_model}", use_container_width=True):

                        # ── 技術位階:從 hist_daily 算 MA20 / MA60 / K 值,讓 AI 點評更具體 ──
                        position_ctx = ""
                        if hist_daily is not None and not hist_daily.empty and len(hist_daily) >= 20:
                            try:
                                high_col_ai = 'max' if 'max' in hist_daily.columns else 'high'
                                low_col_ai  = 'min' if 'min' in hist_daily.columns else 'low'
                                close_s     = hist_daily['close']
                                latest_p    = float(close_s.iloc[-1])
                                ma20_p      = float(close_s.tail(20).mean())
                                ma60_p      = float(close_s.tail(60).mean()) if len(close_s) >= 60 else None

                                # K 值(略過 None 取最近一個有效值)
                                k_last_p = None
                                try:
                                    k_list, _ = _calc_kd_series(hist_daily[high_col_ai], hist_daily[low_col_ai], close_s)
                                    k_last_p = next((k for k in reversed(k_list) if k is not None), None)
                                except Exception:
                                    pass

                                parts = [f"目前股價 {latest_p:.1f}"]
                                parts.append(f"{'站上' if latest_p > ma20_p else '跌破'} MA20({ma20_p:.1f})")
                                if ma60_p is not None:
                                    parts.append(f"{'站上' if latest_p > ma60_p else '跌破'} MA60({ma60_p:.1f})")
                                if k_last_p is not None:
                                    k_desc = "超買" if k_last_p > 80 else ("超賣" if k_last_p < 20 else "中性")
                                    parts.append(f"K值 {k_last_p:.0f}({k_desc})")
                                position_ctx = "技術位階:" + "、".join(parts) + "。"
                            except Exception as _pe:
                                print(f"位階計算失敗(忽略): {_pe}")

                        # 用 textwrap.dedent 清掉每行開頭的縮排,避免 24 spaces 進入 prompt
                        # 浪費 token 並可能影響 AI 對結構的理解
                        # ──「進場節奏」分析:本系統選出的標的本質上都是偏多的(已通過 10 項正向篩選)
                        # 因此判斷的維度是「現在該不該動手」,而非「多空方向」
                        prompt = textwrap.dedent(f"""\
                            你是台灣股市量化分析師。以下個股已通過本系統 10 項正向篩選(法人買、大戶增、KD 起漲、量價突破、營收成長等),基本面已偏多。
                            請根據「技術位階」與「籌碼成熟度」判斷『進場節奏』,用繁體中文寫出約 120 字的進場建議。
                            不要使用 Markdown 語法,以純文字順暢表達。

                            【個股數據報告】
                            股票標的:{sid} {sname}
                            當前股價:{row_data['現價']} 元
                            綜合量化總分:{row_data['總分']} / 10 分
                            產業板塊:{row_data['產業']}
                            20日均成交量:{row_data['20日均量(張)']} 張
                            歷史波動度 (ATR%):{row_data['ATR%']}%
                            三大法人5日籌碼:投信總計 {row_data['投信5日淨額(張)']} 張 / 外資總計 {row_data['外資5日淨額(張)']} 張
                            主力與散戶狀態:{'主力大戶吃貨且散戶退場(籌碼共振成立)' if row_data['★籌碼共振(大戶↑散戶↓)'] == 1 else '籌碼尚未集中(共振未成立)'}
                            最新營收表現:年增/月增率 {row_data['最新月營收增率(%)']}% (採用模式: {row_data['營收模式']})
                            相對大盤強度 (RS):{'表現超越大盤(多頭領頭羊)' if row_data['RS優於大盤'] == 1 else '弱於大盤走勢'}
                            {position_ctx}

                            請在分析最後另起一行,只輸出『進場節奏標籤』,從以下三選一:
                            [節奏] 可進場   ← K值未過熱(<70)、未連續大漲、訊號明確、可現在分批進場
                            [節奏] 拉回再進 ← K值超買(>80)或近期已大漲、追高風險高、建議等回檔
                            [節奏] 觀察    ← 訊號剛開始、籌碼未完全確認、再觀察 1-2 天再決定
                            """)

                        # 決定要試的模型清單:
                        # - 自動輪替 → models=None,讓 ai_helper 套 PREFERRED_AI 環境變數排序
                        # - 指定單一模型 → 傳 [單一 dict] 強制只試這個
                        if selected_model == "自動輪替 (推薦)":
                            models_to_try = None
                            spinner_msg = "AI 分析師正在輪替模型中,請稍候..."
                        else:
                            models_to_try = [m for m in AI_MODELS if m["name"] == selected_model]
                            spinner_msg = f"{selected_model} 分析師正在閱讀籌碼數據..."

                        with st.spinner(spinner_msg):
                            model_name, result = call_openrouter_ai(
                                prompt, models=models_to_try, max_tokens=400
                            )

                        if model_name:
                            st.session_state.ai_cache[cache_key] = (model_name, result)
                            st.rerun()
                        else:
                            st.error("❌ 所有 AI 模型皆失敗,請稍後再試。原因請看終端機 log。")

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

        # ── tab5 資金管理計算器 ─ 進場價預設帶該股現價 ──
        with tab5:
            st.subheader(f"💰 {sid} {sname} 部位計算")
            st.caption("先決定能虧多少 → 反推該買幾張。避免憑感覺下單導致單筆風險過大。")

            # 進場價預設值:row_data 有就用現價,沒有就 fallback 100
            try:
                _default_entry = float(row_data['現價']) if row_data is not None and pd.notna(row_data.get('現價')) else 100.0
            except Exception:
                _default_entry = 100.0

            # 把使用者「總資金/單筆風險/整張交易」三個偏好存進 session,
            # 切換不同股票時設定維持,只有「進場價」會隨股票變
            if 'fund_total' not in st.session_state:
                st.session_state.fund_total = 100.0
            if 'fund_risk_pct' not in st.session_state:
                st.session_state.fund_risk_pct = 2.0
            if 'fund_use_lot' not in st.session_state:
                st.session_state.fund_use_lot = True

            fc1, fc2, fc3 = st.columns(3)
            fund_total = fc1.number_input(
                "總資金(萬)", min_value=10.0, max_value=10000.0,
                value=st.session_state.fund_total, step=10.0, key="fund_total_input",
                help="你拿來操作股票的本金總額(不是身家)"
            )
            st.session_state.fund_total = fund_total

            risk_pct = fc2.number_input(
                "單筆風險 %", min_value=0.5, max_value=10.0,
                value=st.session_state.fund_risk_pct, step=0.5, key="fund_risk_input",
                help="這一筆交易最多容忍虧損占總資金的比例。\n保守:1~2%,積極:3~5%。\n**這是你的紀律,不該隨股票改變。**"
            )
            st.session_state.fund_risk_pct = risk_pct

            stop_loss_pct = fc3.number_input(
                "停損 %", min_value=2.0, max_value=20.0,
                value=7.0, step=0.5, key=f"fund_stop_{sid}",
                help="從進場價跌多少 % 認賠出場。\n建議參考:跌破 MA20 通常 5~8%、跌破 MA60 通常 10~12%"
            )

            fc4, fc5 = st.columns(2)
            entry_price = fc4.number_input(
                "進場價(元)", min_value=1.0, max_value=100000.0,
                value=_default_entry, step=0.5, key=f"fund_entry_{sid}",
                help=f"預設帶入 {sid} 目前現價 {_default_entry:.1f} 元"
            )

            use_default_lot = fc5.checkbox(
                "整張交易(1 張 = 1000 股)", value=st.session_state.fund_use_lot,
                key="fund_lot_check"
            )
            st.session_state.fund_use_lot = use_default_lot

            # ── 計算 ──
            risk_amount = fund_total * 10000 * (risk_pct / 100)      # 容忍虧損金額(元)
            loss_per_share = entry_price * (stop_loss_pct / 100)     # 每股虧損(元)

            if loss_per_share > 0:
                max_shares = risk_amount / loss_per_share
                if use_default_lot:
                    max_lots = int(max_shares // 1000)
                    actual_position = max_lots * 1000 * entry_price
                    actual_risk = max_lots * 1000 * loss_per_share
                else:
                    max_lots = max_shares
                    actual_position = max_shares * entry_price
                    actual_risk = max_shares * loss_per_share

                position_pct = actual_position / (fund_total * 10000) * 100 if fund_total > 0 else 0
                stop_price = entry_price * (1 - stop_loss_pct / 100)

                st.divider()
                st.markdown("##### 📋 建議部位")
                rc1, rc2, rc3 = st.columns(3)
                if use_default_lot:
                    rc1.metric("建議買進", f"{max_lots} 張", f"= {max_lots*1000:,} 股" if max_lots > 0 else "資金不足")
                else:
                    rc1.metric("建議買進", f"{max_shares:.0f} 股")
                rc2.metric("動用資金", f"{actual_position:,.0f} 元", f"占總資金 {position_pct:.1f}%")
                rc3.metric("最大虧損", f"{actual_risk:,.0f} 元", f"= {risk_pct:.1f}% 總資金")

                st.markdown(
                    f"📍 進場價 **{entry_price:.1f}** 元 → 停損價 **{stop_price:.1f}** 元 "
                    f"(跌幅 {stop_loss_pct:.1f}%)"
                )

                # ── 警示 ──
                if use_default_lot and max_lots == 0:
                    needed_capital = (1000 * entry_price) / (position_pct / 100 if position_pct > 0 else 0.5)
                    st.warning(
                        f"⚠️ 依此風險設定無法買進 1 整張(需動用 {1000 * entry_price:,.0f} 元 "
                        f"= {1000 * entry_price / (fund_total * 10000) * 100:.1f}% 總資金,超出風險上限)。\n\n"
                        f"建議:**提高單筆風險**、**縮小停損**、或**勾掉整張交易改買零股**。"
                    )
                elif position_pct > 50:
                    st.warning(
                        f"⚠️ 單檔部位占比 {position_pct:.0f}% 過高,建議分散到 2~3 檔以降低集中度風險。"
                    )
                elif use_default_lot and max_lots > 0:
                    st.success(
                        f"✅ 部位符合紀律,可下單。"
                        f"記得設定停損價 **{stop_price:.1f}** 元,跌破即出場不戀棧。"
                    )

                # ── 小教學 ──
                with st.expander("💡 看不懂?資金管理 1 分鐘速懂", expanded=False):
                    st.markdown(
                        f"""
**為什麼要算這個?**

很多人是這樣決定的:「我有 50 萬,2603 一張 7.5 萬,我來個 5 張好了!」── 完全沒考慮「**萬一跌了會虧多少**」。

資金管理計算器**反過來算**:**先決定能虧多少,反推該買幾張**。

**3 個關鍵概念**

1. **單筆風險 %**(填 {risk_pct}%)= 這筆交易最多虧總資金的多少%。**這是紀律,不該隨股票改變**。保守 1~2%,積極 3~5%。
2. **停損 %**(填 {stop_loss_pct}%)= 從進場價跌多少 % 出場。跟單筆風險不同 ── 一個是**佔本金**,一個是**佔股價**。
3. **進場價**(填 {entry_price:.1f})= 你打算買的價位。**預設帶入這檔股票的現價**。

**讀懂建議部位**

- **{max_lots if use_default_lot else int(max_shares)} 張**:在你的紀律內最多能買的量
- **動用資金 {actual_position:,.0f} 元**:這筆要花多少錢
- **最大虧損 {actual_risk:,.0f} 元**:最壞情況虧多少(就是 {risk_pct}% 總資金)

**新手心法**:**永遠把單筆風險設 2%**,其他照實填,**計算器叫你買幾張就買幾張**,不要硬加碼。長期下來才能活到大勝那一次。
                        """
                    )
