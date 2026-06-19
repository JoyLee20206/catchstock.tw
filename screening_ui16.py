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
import html
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
from data_health import cache_freshness
from picks_history import load_history, compute_hot_picks
from data_health import check_data_health
from industry_rotation import compute_industry_rotation
from performance import (
    compute_performance, compute_equity_curve, backtest_market_filter,
    attribute_signals, backtest_exit_rules, compute_per_stock_performance,
    check_system_health,
)
from backtest import run_backtest, SIGNAL_LABELS, build_signal_matrices
from market_sentiment import (
    compute_sentiment,
    load_sentiment_history,
    backtest_sentiment,
    persist_sentiment_history,
    persist_fi_history,
    persist_retail_history,
)

# ── 自選股與交易筆記持久化 ────────────────────────────────────────────────
# 路徑統一以 CACHE_DIR 為基準,避免散落各處改 cache 位置時遺漏
WATCHLIST_FILE = str(CACHE_DIR / "watchlist.json")
NOTES_FILE     = str(CACHE_DIR / "notes.json")

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


# 乖離率「過熱黃燈」門檻(%)— 個股波動比大盤大,故比大盤位階的 ±8% 寬。
# 進場節奏 pill 與選股結果表共用同一組門檻(避免兩處數字不一致)。
BIAS_MA20_HOT = 12.0   # 股價高於 MA20 超過此 % → 短中期延伸過大
BIAS_MA60_HOT = 20.0   # 股價高於季線 MA60 超過此 % → 中期追高風險

# 🎯 籌碼抄底雷達:選股結果表標記「今天同時觸發這幾個訊號」的股(④ 實證固化的組合)。
# 取 backtest 的 detect 訊號矩陣最新一列做 AND。資減券增 + 大戶逆勢增持 是實證裡
# 唯一「樣本夠(78 筆)+ 平均正(+3.95%)+ 夏普>1」的籌碼底部組合(優於加 KD背離的稀有 6 筆組合)。
# 要調整這組合(④ 重新驗證後)只改這一行即可;UI 文字會引用此清單長度。
DIP_COMBO_SIGNALS = ["margin_squeeze", "chip_accumulation"]


def render_verdict_pill(verdict: str, reason: str = "") -> None:
    """渲染進場節奏 pill(用 markdown HTML)。
    reason:選填,顯示在標籤後(例「乖離季線 +22%」),讓使用者知道為何黃燈/綠燈。"""
    if verdict not in VERDICT_STYLE:
        return
    s = VERDICT_STYLE[verdict]
    import html as _html
    _reason_html = f" · {_html.escape(reason)}" if reason else ""
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
        ">{s['icon']} 進場節奏 · {s['label']}{_reason_html}</div>""",
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
    # 防呆:callback 可能在 session_state 尚未初始化 trading_notes 時就觸發 → 補載入,避免 AttributeError
    if 'trading_notes' not in st.session_state or not isinstance(st.session_state.trading_notes, dict):
        st.session_state.trading_notes = load_notes()
    st.session_state.trading_notes[sid] = val
    save_note(sid, val)

# ── 資料快取讀取 ───────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def get_stock_institutional(stock_id: str) -> pd.DataFrame:
    """讀取最新的三大法人快取並過濾指定標的。

    改用 pyarrow filters 在讀取層就過濾(只載入該股的 ~30 筆資料),
    不再讀整張 ~50,000 筆表。約 300 倍加速,K 線圖切換股票體感極快。
    """
    files = sorted(CACHE_DIR.glob('institutional_*.parquet'))
    if not files:
        return pd.DataFrame()
    try:
        df_stock = pd.read_parquet(
            files[-1],
            filters=[('stock_id', '==', str(stock_id))]   # ← pyarrow 層過濾
        )
        if df_stock.empty:
            return pd.DataFrame()
        df_stock['stock_id'] = df_stock['stock_id'].astype(str)
        df_stock['date']     = pd.to_datetime(df_stock['date'])
        return df_stock
    except Exception:
        return pd.DataFrame()

# 1 hr → 5 min:yfinance 偶爾失敗會被快取空結果擋住,縮 TTL 讓恢復更快
@st.cache_data(ttl=300, show_spinner=False)
def _load_twii_cached() -> pd.DataFrame:
    """讀取 ^TWII 2 年歷史 K 線,優先用本地 parquet (~ 50ms),沒有才打 yfinance (~ 10s)。

    fetch_cache.py 排程會把 ^TWII 寫進 cache/twii_*.parquet,
    冷啟動時這條就會走 parquet 路徑,從 10 秒降到 < 100ms。
    """
    # ── 路徑 A:本地 parquet (最快) ──
    try:
        twii_files = sorted(CACHE_DIR.glob("twii_*.parquet"))
        if twii_files:
            df_local = pd.read_parquet(twii_files[-1])
            if not df_local.empty and "date" in df_local.columns:
                df_local["date"] = pd.to_datetime(df_local["date"])
                df_local = df_local.set_index("date").sort_index()
                if df_local.index.tz is not None:
                    df_local.index = df_local.index.tz_localize(None)
                return df_local
    except Exception as e:
        print(f"⚠ 讀本地 twii parquet 失敗,改打 yfinance: {e}")

    # ── 路徑 B:fallback 到 yfinance (慢) ──
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


# ── 大盤情緒(快取 5 分鐘,提前定義供「狀態總覽」 + 大盤情緒 tab 共用) ──
@st.cache_data(ttl=300, show_spinner=False)
def _load_sentiment_cached():
    """純讀取:回傳 compute_sentiment 結果,被 Streamlit 快取 5 分鐘。

    ⚠ 純函式:不含副作用。寫歷史檔由 _get_sentiment_and_persist 在快取外負責。
    """
    return compute_sentiment(CACHE_DIR)


@st.cache_data(ttl=600, show_spinner=False)
def _load_sentiment_backtest_cached():
    """溫度計有效性回測(溫度 vs ^TWII 後續報酬),快取 10 分鐘。"""
    return backtest_sentiment(CACHE_DIR, n_days_list=(5, 10))


def _get_sentiment_and_persist():
    """拿快取結果 + 在快取外做副作用(寫歷史檔)。

    為什麼這樣設計:把寫檔放進 _load_sentiment_cached 會被 @st.cache_data 擋住
    (第二次呼叫直接回快取值,副作用不會跑)。改成讀寫分離,所有呼叫端都能
    保證歷史檔每次重整都會更新。
    """
    s = _load_sentiment_cached()
    if s and s.get("temperature") is not None:
        try:
            persist_sentiment_history(CACHE_DIR, s["temperature"], s.get("label", ""))
        except Exception as e:
            print(f"⚠ persist_sentiment_history 失敗: {e}")
        # 同步寫外資期貨歷史(支援百分位制累積)
        try:
            _fi = s.get("indicators", {}).get("fi_futures", {})
            if _fi.get("value") is not None:
                persist_fi_history(CACHE_DIR, _fi["value"])
        except Exception as e:
            print(f"⚠ persist_fi_history 失敗: {e}")
        # 同步寫散戶估算歷史(支援百分位制累積)
        try:
            _rt = s.get("indicators", {}).get("retail_futures", {})
            if _rt.get("value") is not None and _rt.get("source"):
                persist_retail_history(CACHE_DIR, _rt["value"], _rt["source"])
        except Exception as e:
            print(f"⚠ persist_retail_history 失敗: {e}")
    return s


# ── 從 ^TWII parquet 算「多/空 + 今日漲跌% + 季線乖離」 ──
# 不依賴 meta(meta 要等使用者按過「開始選股」才有),讓狀態列冷啟動就有大盤資訊。
# 不自己加 @st.cache_data,直接用 _load_twii_cached 的 cache 即可,
# 避免「上層也快取 None」造成雙重失敗卡住。
def _market_summary_from_twii():
    df_twii = _load_twii_cached()
    # 若上次抓失敗(空 df 被快取),session 內提供一次自動重試機會
    if (df_twii is None or df_twii.empty) and not st.session_state.get("_twii_retry_used"):
        st.session_state["_twii_retry_used"] = True
        _load_twii_cached.clear()
        df_twii = _load_twii_cached()
    if df_twii is None or df_twii.empty or len(df_twii) < 60:
        return None
    closes = df_twii['Close'].dropna()
    if len(closes) < 60:
        return None
    today_close = float(closes.iloc[-1])
    prev_close  = float(closes.iloc[-2])
    ma60        = float(closes.tail(60).mean())
    ma20        = float(closes.tail(20).mean())
    # 與 screening0515 的盤整判斷同口徑:站上季線但「跌破月線」或「近 20 日下跌」
    chg20       = (today_close / float(closes.iloc[-21]) - 1) * 100 if len(closes) > 20 else 0.0
    bullish     = today_close >= ma60
    return {
        "close":    today_close,
        "ma60":     ma60,
        "pct":      (today_close - prev_close) / prev_close * 100,
        "bias":     (today_close - ma60) / ma60 * 100,
        "bullish":  bullish,
        "consolidating": bullish and (today_close < ma20 or chg20 < 0),
        "latest_date": closes.index[-1],   # 最新一筆指數的日期(供延遲偵測)
    }


# ── 頁面設定與快取狀態橫幅 ──────────────────────────────────────────────────
st.set_page_config(page_title="台股選股", page_icon="📊", layout="wide")
# 縮小頂部大留白 + 收緊標題/區塊間距,讓首頁不那麼空、內容往上移
st.markdown("""
<style>
.block-container, [data-testid="stMainBlockContainer"], [data-testid="stAppViewBlockContainer"] {
    padding-top: 2.8rem !important;   /* 比預設(~6rem)緊,但仍清開頂部工具列、不蓋標題 */
}
/* 標題字體縮小,首頁更精簡、不佔高 */
h1, [data-testid="stMainBlockContainer"] h1 {
    font-size: 2rem !important;
    margin-top: 0 !important;
    padding-top: 0 !important;
    padding-bottom: 0.2rem !important;
}
hr { margin: 0.6rem 0 !important; }
/* FAQ/說明 expander 內的標題縮小、間距收緊,讓說明更清爽不佔高(不影響主畫面 subheader) */
[data-testid="stExpander"] h2 { font-size: 1.2rem !important; margin: 0.5rem 0 0.3rem !important; }
[data-testid="stExpander"] h3 { font-size: 1.05rem !important; margin: 0.5rem 0 0.25rem !important; }
[data-testid="stExpander"] h4 { font-size: 0.98rem !important; margin: 0.4rem 0 0.2rem !important; }
[data-testid="stExpander"] p,
[data-testid="stExpander"] li { font-size: 0.9rem !important; line-height: 1.5 !important; }
[data-testid="stExpander"] hr { margin: 0.4rem 0 !important; }
</style>
""", unsafe_allow_html=True)
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

# ── 📊 狀態總覽(資料 / 大盤 / 情緒 三欄合一,取代 3 個獨立 banner) ────────
# 顏色用 freshness level 對應 emoji 顯示,避免重度 styling
_LEVEL_EMOJI = {"missing": "❌", "ok": "✅", "info": "ℹ️", "warn": "⚠️", "error": "❌"}
_status_cols = st.columns(3)

# 第 1 欄:資料新鮮度
with _status_cols[0]:
    _emoji = _LEVEL_EMOJI.get(_freshness["level"], "ℹ️")
    # 用精簡版 short(維持單行不擠);詳細說明留在「資料更新」expander
    _data_lines = [f"**{_emoji} {_freshness.get('short', _freshness['msg'])}**"]
    # 嘗試讀抓取時戳
    try:
        _ts_file = CACHE_DIR / "last_fetch_daily.txt"
        if _ts_file.exists():
            _fetch_raw = _ts_file.read_text(encoding="utf-8").strip()
            _is_fallback = "FALLBACK" in _fetch_raw
            _fetch_ts = _fetch_raw.replace("FALLBACK", "").strip()
            try:
                # 盤中 = 平日 09:00~13:30(台股交易時段);其餘時間(含凌晨/盤前/週末)一律盤後
                _ft = datetime.strptime(_fetch_ts, "%Y-%m-%d %H:%M:%S")
                _is_intraday = _ft.weekday() < 5 and (9, 0) <= (_ft.hour, _ft.minute) <= (13, 30)
                _suffix = "⏰ 盤中" if _is_intraday else "🌙 盤後"
            except Exception:
                _suffix = ""
            if _is_fallback:
                _suffix += " ⚠️ 抓取失敗用舊檔"
            _data_lines.append(f"🕐 {_fetch_ts} {_suffix}")
    except Exception:
        pass
    st.markdown("<br>".join(_data_lines), unsafe_allow_html=True)

# 第 2 欄:大盤(冷啟動就有,不等使用者按開始選股)
with _status_cols[1]:
    _mkt = _market_summary_from_twii()
    if _mkt is None:
        st.markdown("**📈 大盤** N/A<br>_(yfinance 暫無資料)_", unsafe_allow_html=True)
    else:
        if not _mkt["bullish"]:
            _bull_icon, _bull_txt = "📉", "空頭"
        elif _mkt.get("consolidating"):
            _bull_icon, _bull_txt = "🌀", "盤整修正"
        else:
            _bull_icon, _bull_txt = "📈", "多頭"
        _pct_color = "#dc2626" if _mkt["pct"] > 0 else ("#16a34a" if _mkt["pct"] < 0 else "#6b7280")
        # 延遲偵測(B):指數最新日期落後個股 cache 最新日 → 標示「資料延遲」,避免拿落後值誤判
        _stale_note = ""
        _twii_date = _mkt.get("latest_date")
        if _twii_date is not None and _cache_date is not None:
            try:
                if pd.Timestamp(_twii_date).normalize() < pd.Timestamp(_cache_date).normalize():
                    _stale_note = (
                        f"<br><span style='color:#d97706;font-size:12px'>"
                        f"⚠️ 指數資料延遲至 {pd.Timestamp(_twii_date).strftime('%m/%d')}(來源更新中)</span>"
                    )
            except Exception:
                pass
        st.markdown(
            f"**{_bull_icon} {_bull_txt}** "
            f"<span style='color:{_pct_color};font-weight:600'>{_mkt['pct']:+.2f}%</span><br>"
            f"加權指數 {_mkt['close']:,.0f} / 乖離 {_mkt['bias']:+.1f}%"
            f"{_stale_note}",
            unsafe_allow_html=True,
        )

# 第 3 欄:市場情緒溫度
with _status_cols[2]:
    try:
        _sent_overview = _get_sentiment_and_persist()
    except Exception:
        _sent_overview = None
    if _sent_overview and _sent_overview.get("temperature") is not None:
        _t  = _sent_overview["temperature"]
        _tl = _sent_overview["label"]
        _ti = _sent_overview["icon"]
        _tcolor = (
            "#16a34a" if _t >= 70 else
            "#65a30d" if _t >= 55 else
            "#ca8a04" if _t >= 45 else
            "#ea580c" if _t >= 30 else
            "#dc2626"
        )
        st.markdown(
            f"**🌡️ 市場溫度** "
            f"<span style='color:{_tcolor};font-weight:700;font-size:18px'>{_t}/100</span><br>"
            f"{_ti} {_tl} <span style='color:#6b7280;font-size:12px'>(詳情看「大盤情緒」tab)</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("**🌡️ 市場溫度** N/A<br>_(情緒指標暫時無法取得)_", unsafe_allow_html=True)

# 「資料更新」expander 的位置佔位:讓它顯示在「狀態列正下方」
# (實際內容仍由下方 if 區塊填入,用 _update_bar_slot.expander 渲染到這裡,避免大段搬移)
_update_bar_slot = st.container()

# ── 🔍 快速搜尋(永遠顯示,不依賴 cache 狀態) ─────────────────────────────
# 輸入代號 → 設定 target_sid → rerun,自動跳到下方個股分析區。
# 不在 picks 名單內的也能查(會在下方提示「未進入今日達標」)。
_search_cols = st.columns([5, 1, 1])
with _search_cols[0]:
    _search_query = st.text_input(
        "🔍 代號 / 名稱搜尋",
        value="",
        placeholder="輸入代號 (例 2330) 或名稱關鍵字 (例 台積),按 Enter 或點旁邊按鈕",
        label_visibility="collapsed",
        key="_quick_search_input",
    )
with _search_cols[1]:
    _search_clicked = st.button(
        "🔍 跳轉", type="primary", use_container_width=True,
        help="跳到個股分析區。已存在於今日達標清單會自動高亮對應 row。"
    )
with _search_cols[2]:
    _search_clear = st.button(
        "↺ 清除", use_container_width=True,
        help="清除目前選定的個股,回到「未指定」狀態。"
    )

if _search_clear:
    st.session_state.target_sid = None
    st.rerun()

# 處理搜尋:支援「純代號」 或「中文名稱關鍵字」
if (_search_query.strip()) and (_search_clicked or _search_query.strip().isdigit()):
    _q = _search_query.strip()
    _hit_sid = None
    # 純 4 位數字 → 直接當代號
    if _q.isdigit() and 3 <= len(_q) <= 6:
        _hit_sid = _q
    else:
        # 名稱關鍵字 → 從 ui_name_map 模糊比對(第一筆命中)
        for _sid, _name in ui_name_map.items():
            if _q in str(_name):
                _hit_sid = _sid
                break
    if _hit_sid:
        st.session_state.target_sid = _hit_sid
        # 不直接 rerun,讓畫面提示後再讓使用者繼續滾動
        _name_hit = ui_name_map.get(_hit_sid, "")
        st.success(f"✅ 已跳轉至 {_hit_sid} {_name_hit}(請滾動到下方個股分析區)")
    else:
        st.warning(f"⚠️ 找不到「{_q}」對應的股票(代號需 4 位數字,名稱關鍵字需精確)")

if _freshness["level"] != "missing":

    # ── 雲端更新按鈕區塊(預設摺疊,需要時點開) ─────────────────────────
    # 改成 expander 是為了把「平常不會用到的兩顆按鈕」收起來,讓頂部更乾淨。
    # 預設只在 freshness 為 warn / error 時自動展開(代表使用者該抓資料)。
    _update_default_expanded = _freshness["level"] in ("warn", "error")
    with _update_bar_slot.expander("🔄 資料更新 / 上次抓取時戳", expanded=_update_default_expanded):
        _caption_parts = ["雲端環境專用:點擊後從網路抓取最新台股資料。"]
        try:
            daily_files = sorted(CACHE_DIR.glob('daily_*.parquet'))
            if daily_files:
                _stem = daily_files[-1].stem
                _fetch_day = _stem.replace('daily_', '')
                _data_max = _cache_date.strftime('%Y-%m-%d') if _cache_date is not None else "?"
                if _fetch_day == _data_max:
                    _caption_parts.append(f"📅 資料日期 **{_fetch_day}**")
                else:
                    _caption_parts.append(f"📅 檔名 **{_fetch_day}** / 最新交易日 **{_data_max}**")

            # ── 抓取時戳:讀 last_fetch_daily.txt 的「內容」,不靠 mtime(git pull 會重置) ──
            # 含時間才能判斷盤中(平日 09:00~13:30)或盤後;若帶 FALLBACK 標記 = 本次抓失敗用舊檔
            _ts_file = CACHE_DIR / "last_fetch_daily.txt"
            if _ts_file.exists():
                _fetch_raw = _ts_file.read_text(encoding="utf-8").strip()
                _is_fallback = "FALLBACK" in _fetch_raw
                _fetch_ts = _fetch_raw.replace("FALLBACK", "").strip()
                _suffix = ""
                try:
                    # 盤中 = 平日 09:00~13:30;其餘(含凌晨/盤前/週末)一律盤後
                    _ft = datetime.strptime(_fetch_ts, "%Y-%m-%d %H:%M:%S")
                    _is_intraday = _ft.weekday() < 5 and (9, 0) <= (_ft.hour, _ft.minute) <= (13, 30)
                    _suffix = " ⏰ 盤中" if _is_intraday else " 🌙 盤後"
                except Exception:
                    pass
                if _is_fallback:
                    _suffix += " ⚠️ <span style='color:#d97706'>**本次抓取失敗,仍為前次資料**</span>"
                _caption_parts.append(f"🕐 抓取於 **{_fetch_ts}**{_suffix}")
        except Exception:
            pass
        st.caption(" · ".join(_caption_parts))

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
                        "無視當日 cache,重新抓取**個股日 K + 加權指數**(兩者都是台股,盤後重抓才是真收盤)。\n\n"
                        "**用途**:盤中已經跑過、盤後要拿到真正收盤價時(14:00 後再按)。\n"
                        "VIX 是美股、當天抓過不會變,故**不重抓**;法人/融資券/營收/大戶 也沿用今日 cache,不浪費 API。\n"
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
                    # 抓完自動重跑選股 (避免使用者忘記再按一次「開始選股」)
                    st.session_state["_auto_rerun_screening"] = True
                    st.rerun()
                except subprocess.CalledProcessError as e:
                    st.error("❌ 雲端腳本執行失敗!")
                    st.code(e.stderr)
                except Exception as e:
                    st.error(f"❌ 發生未知的錯誤:{e}")

        # ── 按鈕 2:強制更新日 K(無視 cache,重抓 daily + 加權指數 twii;VIX 不重抓=美股當天不變) ──
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
                        # 抓完 daily 自動重跑選股 (避免使用者忘記再按一次「開始選股」)
                        st.session_state["_auto_rerun_screening"] = True
                        _status.update(label="✅ daily K 線已更新,自動重跑選股中...", state="complete")
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

    with st.expander("💡 策略邏輯導覽 (FAQ)", expanded=False):
        st.markdown("""
### 🎯 第一區:快速套用組合(不會調?先點這個)
不知道參數怎麼設?這裡有四套現成設定,**點一下就套用**:
* **預設**:最通用,平常就用這個。
* **多頭寬鬆**:大盤明顯在漲時用 → 條件放寬,才不會漏掉剛起漲的股。
* **空頭嚴格**:大盤在跌/亂時用 → 條件變嚴,只挑最強、最抗跌的股。
* **低檔發動**:**抄底向**,跟上面三個「追強勢」相反。專找「KD 從**很深的超賣**剛翻揚、而且**今日還在低檔**(還沒漲上去)」的股,門檻 6 分、日均量 ≥ 500 張(濾掉冷門股)。適合抓**跌深翻揚的轉折**。
    > 💡 還是覺得選太多?把上方「過關門檻」滑桿再往上調(7、8)就會更精;或改用「🔬 訊號回測」裡更精準的反轉組合(大戶增持 + 資減券增)。
    > ⚠️ 抄底天生勝率較低、容易接刀子 → **進場要等帶量紅K確認、嚴設停損**;先用訊號回測驗證有沒有 edge 再實戰。

> 💡 小白就先用「預設」,熟了再依大盤多空切換;想找轉折再用「低檔發動」。

---

### 📌 第二區:核心 / KD(打分數 + 看技術線)
這區決定「一檔股票**及不及格**」。

* **過關門檻(滿分 10 分)**:系統幫每檔股票**打分數**,**達到門檻才會被選出來**。
    * 分數怎麼來:法人有沒有買、大戶有沒有增加(2 分)、散戶有沒有在跑(2 分)、營收有沒有成長、技術面強不強…等加總。大戶/散戶配重最高,因為歷史數據證明它們最會挑到贏家。(「券相關」歷史數據反而扣分,已暫停計分、僅記錄觀察)
    * 門檻**調高=更嚴**(選出來少而精)、**調低=更寬鬆**(選出來多)。
    * 🛡️ 大盤轉弱(跌破季線)時,系統會**自動把門檻 +1 分**,幫你變謹慎、避開容易跟跌的弱股。
* **KD 是什麼?** 一個常見技術指標,簡單看股價是「便宜被低估」還是「太貴被高估」。
* **那三個 KD 設定**(觀察區間 / 低檔啟動 / 今日上限),白話就是一句話:
    > **找「最近剛從低點翻揚、但還沒漲過頭」的股**,避免你買在山頂套牢。

---

### 📌 第三區:預篩 / 法人(先擋地雷 + 跟著大戶走)
* **均量 / ATR% 上限 = 第一道濾網**:
    * 均量太低 = **冷門股**(買得到、賣不掉)→ 擋掉。
    * ATR% 太高 = **暴漲暴跌的妖股**(今天漲停明天跌停)→ 擋掉。
* **法人買超天數(重點:算「累計」不是「連續」)**:
    * 例:看 7 天、要求至少 5 天有買 → 只要這 7 天裡**任意 5 天**法人站買方就算數。
    * 為什麼?法人吃貨常「買三天、休一天」,用累計才不會因中間休一天就把好股漏掉。
* **「★籌碼共振」是什麼?**
    * 同期間「**大戶增加持股**(主力在吃貨)」+「**散戶減少持股**(散戶被洗出場)」同時發生 = 共振。
    * 它不加分,但**分數一樣時,有共振的排最前面**(代表籌碼最乾淨、最有主力照顧)。

---

> 📌 大盤情緒、資料更新等說明,請看**首頁上方對應的分頁**(🌡️ 大盤情緒、🔄 資料更新)。
        """)
    st.divider()
    st.subheader("🎯 快速套用組合")
    st.button("預設",     on_click=apply_preset, args=('default',),   use_container_width=True)
    st.button("多頭寬鬆", on_click=apply_preset, args=('bull',),      use_container_width=True)
    st.button("空頭嚴格", on_click=apply_preset, args=('bear',),      use_container_width=True)
    st.button("低檔發動", on_click=apply_preset, args=('low_launch',), use_container_width=True,
              help="抄底向:找 KD 從深超賣剛翻揚、今日仍在低檔(還沒漲上去)的股;門檻 6 分、日均量 ≥ 500 張。"
                   "選太多可再把門檻往上調;與其他偏追強勢的組合互補,先用訊號回測驗證再用。")

    st.divider()
    st.subheader("📌 核心 / KD")
    st.slider(labeled("過關門檻 PASS_SCORE", 'pass_score'), 1, 10, key='pass_score',
              help="滿分 10 分，達標才算入榜。\n\n計分包含：法人買超、大戶增散戶減（各 2 分，訊號歸因唯二正 edge）、KD金叉、營收成長與大盤相對強弱。券相關 2026-06 起、技術面三合一 2026-06-19 起停用觀察（只記錄不計分；技術面三合一歸因 edge -4.99% 為反指標）。\n\n💡 現行門檻 8：技術面停用後滿分由 11 降為 10，門檻維持 8 → 實質收緊（凡含技術面的股少 1 分）。9 分=大戶+散戶雙籌碼共振；沒有籌碼訊號的股票進不來。大盤破季線會再自動 +1（空頭從嚴）。")
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
    bullish       = meta.get('market_bullish', True)
    consolidating = meta.get('market_consolidating', False)
    twii_now_raw = meta.get('twii_now')
    twii_ma_raw  = meta.get('twii_ma')
    change_raw   = meta.get('twii_lookback_change')
    # Bug B 修正：用 pd.notna 同時擋 None 與 NaN
    change       = float(change_raw) if pd.notna(change_raw) else 0.0
    if not bullish:
        state_txt = "📉 空頭(跌破季線)"
    elif consolidating:
        state_txt = "🌀 盤整修正(站上季線但跌破月線/近20日下跌)"
    else:
        state_txt = "📈 多頭(站上季線)"
    twii_now_s   = f"{twii_now_raw:,.0f}" if pd.notna(twii_now_raw) else "N/A"
    twii_ma_s    = f"{twii_ma_raw:,.0f}"  if pd.notna(twii_ma_raw)  else "N/A"
    if base is None or eff is None:
        thr_txt = "過關門檻未知"
    elif base == eff:
        thr_txt = f"過關門檻 {base}"
    else:
        thr_txt = f"過關門檻 {base} → **{eff}**"
    msg = f"{state_txt}  |  {thr_txt}  |  加權指數 {twii_now_s} / MA60 {twii_ma_s} | 近 20 日 {change:+.2f}%"
    if consolidating:
        msg += "  |  ⚠️ RS 不計分,且需「大戶↑或散戶↓」至少其一才入選"
    if bullish and not consolidating:
        st.success(msg)
    else:
        st.warning(msg)

# 自動重跑旗標:強制更新日 K / 標準抓取完成後設置,讓選股不需手動再按一次
_auto_rerun = st.session_state.pop("_auto_rerun_screening", False)

# ── 選股結果 cache(同參數 + 同資料日期 → 秒切) ────────────────────────
# 為什麼這樣設計:
#   - run_screening 跑一次要 ~ 15 秒(讀 5 parquet + 1800 檔逐檔計分)
#   - 使用者常按完選股後又改其他參數重按、或頁面 rerun 重觸發
#   - 用 (params_tuple, cache_date) 當 cache key,同參數+同日資料完全省下
#   - 改任何參數或資料日期變動 → cache miss,自動重算
@st.cache_data(ttl=3600, show_spinner="選股中(讀 parquet + 1800 檔逐檔計分,約 15 秒)…")
def _run_screening_cached(params_tuple: tuple, cache_date_str: str):
    """params_tuple = (pass_score, lookback_days, it_min, fi_min, kd_lookback,
                        kd_low_from, kd_high_cap_now, min_avg_vol_lots, atr_max_pct)
    cache_date_str: 用最新資料日期當 cache key,daily parquet 更新會自然觸發重算。

    回傳 (df, files_bytes, meta):
        - files_bytes:{ftype: (filename, bytes)},因為 tmpdir 會在函式結束時刪除,
          必須把檔案內容讀進 bytes 再回傳,不能傳 path。
    """
    (_p, _lb, _it, _fi, _kdlb, _kdlo, _kdhi, _vol, _atr) = params_tuple
    with tempfile.TemporaryDirectory() as tmpdir:
        df, file_paths, meta = run_screening(
            pass_score=_p, lookback_days=_lb,
            it_min_buy_days=_it, fi_min_buy_days=_fi,
            kd_lookback=_kdlb, kd_low_from=_kdlo, kd_high_cap_now=_kdhi,
            min_avg_vol_lots=_vol, atr_max_pct=_atr,
            output_dir=Path(tmpdir),
        )
        files_bytes: dict = {}
        for ftype, fpath in file_paths.items():
            if Path(fpath).exists():
                with open(fpath, 'rb') as fh:
                    files_bytes[ftype] = (Path(fpath).name, fh.read())
    return df, files_bytes, meta


if run_clicked or _auto_rerun:
    _spinner_msg = ("資料已更新,自動重跑選股中..." if _auto_rerun else "選股中...")
    _params_tuple = (
        st.session_state.pass_score, st.session_state.lookback_days,
        st.session_state.it_min_buy_days, st.session_state.fi_min_buy_days,
        st.session_state.kd_lookback, st.session_state.kd_low_from,
        st.session_state.kd_high_cap_now, st.session_state.min_avg_vol_lots,
        st.session_state.atr_max_pct,
    )
    _cache_key_str = _cache_date.strftime('%Y-%m-%d') if _cache_date is not None else "no_data"
    df, files_bytes, meta = _run_screening_cached(_params_tuple, _cache_key_str)
    st.session_state.result_df = df
    st.session_state.result_files = files_bytes
    st.session_state.result_meta = meta
    if _auto_rerun:
        st.success("✅ 資料已更新,選股結果自動重跑完成")
    else:
        st.success("✅ 完成，看下方結果")

# ── 結果顯示區 ─────────────────────────────────────────────────────────────
df = st.session_state.result_df
files_bytes = st.session_state.result_files or {}
meta = st.session_state.result_meta

if meta: show_market_banner(meta)

# ── 7 日入選熱度榜(資料來自 cache/previous_picks.json,由 Telegram 每日推播寫入) ──
HOT_WINDOW = 20  # 熱度榜只看最近 N 個交易日(歷史保留可達一年,但「熱度」只反映近期)

@st.cache_data(ttl=300, show_spinner=False)
def _load_hot_picks_cached(top_n: int = 10):
    """快取 5 分鐘,避免每次 rerun 都重讀 JSON。

    熱度榜套用 HOT_WINDOW 近期視窗;_hist_days 仍回傳「總累積天數」(供績效門檻判斷)。
    """
    hist = load_history()
    return compute_hot_picks(hist, top_n=top_n, window=HOT_WINDOW), len(hist)

_hot, _hist_days = _load_hot_picks_cached(top_n=10)

# 績效計算函式 提前定義(供首頁速覽卡片 + 後續 tab_perf 共用)
@st.cache_data(ttl=600, show_spinner="計算策略績效中…")
def _load_performance_cached():
    hist = load_history()
    return compute_performance(hist, CACHE_DIR, n_days_list=(3, 5, 10, 20))


@st.cache_data(ttl=600, show_spinner=False)
def _load_market_filter_cached(hold_days=5):
    """大盤濾網實證(只在多頭/空頭進場的績效對照),快取 10 分鐘。"""
    return backtest_market_filter(load_history(), CACHE_DIR, hold_days=hold_days)


@st.cache_data(ttl=600, show_spinner=False)
def _load_signal_attribution_cached(hold_days=5):
    """訊號歸因(10 個計分細項各自對後續報酬的貢獻),快取 10 分鐘。"""
    return attribute_signals(load_history(), CACHE_DIR, hold_days=hold_days)


@st.cache_data(ttl=600, show_spinner=False)
def _load_exit_rules_cached(max_hold=10):
    """出場規則回測(固定持有 vs 停損/停利/移動停損),快取 10 分鐘。"""
    return backtest_exit_rules(load_history(), CACHE_DIR, max_hold=max_hold)


@st.cache_data(ttl=600, show_spinner=False)
def _load_per_stock_perf_cached(hold_days=5):
    """個股層級績效(系統選的哪些股真的賺/賠),快取 10 分鐘。"""
    return compute_per_stock_performance(load_history(), CACHE_DIR, hold_days=hold_days)


@st.cache_data(ttl=600, show_spinner=False)
def _load_system_health_cached(hold_days=5, recent_window=20):
    """系統失效監控(近期 edge 是否還在),快取 10 分鐘。"""
    return check_system_health(load_history(), CACHE_DIR,
                               hold_days=hold_days, recent_window=recent_window)

# ── 🎯 首頁速覽卡片(3 秒看完今日重點) ──
# 4 張 metric:今日達標 / 大盤狀態 / 最強產業 / 系統 5 日勝率
if meta is not None and df is not None:
    _qc1, _qc2, _qc3, _qc4 = st.columns(4)

    # 1. 今日達標檔數
    _n_hit = len(df)
    _qc1.metric("📋 今日達標", f"{_n_hit} 檔",
                "🔥 高張力" if _n_hit >= 20 else ("✅ 正常" if _n_hit >= 5 else "⚠ 稀少"))

    # 2. 市場溫度 + 7 日趨勢(原「大盤」訊息已在頂部狀態總覽,改放溫度)
    # 雙重保險:
    #   ① 優先用 sentiment_history.json 算「近 N 日趨勢」
    #   ② 若歷史檔還沒累積,降級用 _get_sentiment_and_persist() 拿今日值
    # 此邏輯仍保留 fallback,即便 wrapper 已負責讀寫分離,首次部署時 history 仍為空
    _temp_metric_value, _temp_metric_delta = "—", ""
    try:
        _hist_sent = load_sentiment_history(CACHE_DIR)
        if _hist_sent:
            # 模式 ①:歷史檔有資料
            _today_temp  = _hist_sent[-1].get("temp")
            _today_label = _hist_sent[-1].get("label", "")
            _temp_metric_value = f"{_today_temp}/100" if _today_temp is not None else "—"
            if len(_hist_sent) >= 2:
                _ref = _hist_sent[-min(7, len(_hist_sent))]
                _delta = _today_temp - _ref.get("temp", _today_temp)
                _arrow = "↗" if _delta > 2 else ("↘" if _delta < -2 else "→")
                _temp_metric_delta = f"{_arrow} 近 {len(_hist_sent)} 日 {_delta:+d} · {_today_label}"
            else:
                _temp_metric_delta = f"{_today_label} · 累積中(目前 1 日)"
        else:
            # 模式 ②:歷史檔空 → 用快取的當日 sentiment 直接顯示
            _s_fallback = _get_sentiment_and_persist()
            if _s_fallback and _s_fallback.get("temperature") is not None:
                _today_temp = _s_fallback["temperature"]
                _temp_metric_value = f"{_today_temp}/100"
                _temp_metric_delta = f"{_s_fallback.get('label','')} · 累積中(明日起有趨勢)"
    except Exception:
        pass
    _qc2.metric("🌡️ 市場溫度", _temp_metric_value, _temp_metric_delta, delta_color="off")

    # 3. 最強產業
    _top_industry = "—"
    if not df.empty and '產業' in df.columns:
        _v = df['產業'].dropna()
        _v = _v[_v != ""]
        if not _v.empty:
            _top_industry = f"{_v.value_counts().idxmax()} ×{int(_v.value_counts().max())}"
    _qc3.metric("🏭 最強產業", _top_industry)

    # 4. 系統 5 日勝率(需要 history ≥ 5 天才算)
    _wr_text, _wr_delta = "—", "資料累積中"
    try:
        if _hist_days >= 5:
            _o = _load_performance_cached().get("overall", {})
            if "win_rate_5d" in _o:
                _wr_text = f"{_o['win_rate_5d']*100:.0f}%"
                _wr_delta = f"樣本 {_o['n_5d']} 筆"
    except Exception:
        pass
    _qc4.metric("🎯 5 日勝率", _wr_text, _wr_delta)

    st.divider()

# ── 產業輪動追蹤(對比最近 7 日 vs 前 7 日各產業上榜次數) ──────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _load_industry_rotation_cached():
    hist = load_history()
    return compute_industry_rotation(hist, recent_days=7, prev_days=7)

_rotation = _load_industry_rotation_cached()

# ── 4 個分析區塊改用 Tabs 並排,省直向空間 ──
# 「熱度榜 + 產業輪動」同屬「近期趨勢觀察」(個股層級 vs 產業層級),合併同頁。
_tab_trend, _tab_perf, _tab_bt, _tab_sent, _tab_bottom, _tab_margin, _tab_alloc = st.tabs([
    f"🔥 熱度 & 輪動(近{HOT_WINDOW}日)",
    "📊 策略績效",
    "🔬 訊號回測",
    "🌡️ 大盤情緒",
    "🛑 止跌判讀",
    "🧮 期貨保證金",
    "💰 資金配置",
])

# ── 🛑 止跌判讀分頁(21 項訊號 × 四級分級,VIXTWN 為閘門) ─────────────────
with _tab_bottom:
    try:
        from bottom_signal_ui import render_bottom_tab
        render_bottom_tab(CACHE_DIR)
    except Exception as _bs_err:
        st.error(f"止跌判讀載入失敗:{_bs_err}")

# ── 🧮 期貨保證金分頁(個股期貨) ──────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _load_stock_futures_map():
    """讀 TAIFEX 個股期貨標的清單 → {stock_id: {"name", "mults":[乘數...], "init_rate", "maint_rate", "tier"}}。
    同一檔可能同時有標準型(2000)與小型(100)契約 → 收集所有乘數(升冪)。
    另 merge 官方逐檔保證金適用比例(stock_margin 快取),供 UI 自動帶入正確比例;
    無該快取時 init_rate/maint_rate/tier 為 None,UI 退回固定預設。"""
    try:
        files = sorted(CACHE_DIR.glob("stock_futures_*.parquet"))
        if not files:
            return {}
        df_sf = pd.read_parquet(files[-1])
        out = {}
        for _, r in df_sf.iterrows():
            sid = str(r["stock_id"])
            m = int(r.get("multiplier", 2000) or 2000)
            if sid not in out:
                out[sid] = {"name": str(r.get("name", "")), "mults": [], "eng": {},
                            "init_rate": None, "maint_rate": None, "tier": ""}
            if m not in out[sid]["mults"]:
                out[sid]["mults"].append(m)
            _ec = str(r.get("eng_code", "") or "").strip()
            if _ec:
                out[sid]["eng"][m] = _ec        # 英文期貨代碼,per 契約乘數(標準/小型不同碼)
        for v in out.values():
            v["mults"].sort()
        # merge 官方保證金比例(逐檔原始/維持/級距)
        try:
            mfiles = sorted(CACHE_DIR.glob("stock_margin_*.parquet"))
            if mfiles:
                df_m = pd.read_parquet(mfiles[-1])
                for _, r in df_m.iterrows():
                    sid = str(r["stock_id"])
                    if sid in out:
                        ir = r.get("init_rate")
                        mr = r.get("maint_rate")
                        out[sid]["init_rate"]  = float(ir) if pd.notna(ir) else None
                        out[sid]["maint_rate"] = float(mr) if pd.notna(mr) else None
                        out[sid]["tier"]       = str(r.get("tier", "") or "")
        except Exception:
            pass
        return out
    except Exception:
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def _load_etf_futures_map():
    """讀 ETF 期貨快取 → {code: {"name", "specs": {mult: {"init_amt","maint_amt"}}}}。
    ETF 期貨保證金為**固定金額(元/口)**(非比例),標準乘數 10,000、小型 1,000。
    無快取回 {}(UI 優雅降級)。"""
    try:
        files = sorted(CACHE_DIR.glob("etf_futures_*.parquet"))
        if not files:
            return {}
        df = pd.read_parquet(files[-1])
        out = {}
        for _, r in df.iterrows():
            code = str(r["stock_id"])
            m = int(r.get("multiplier", 10000) or 10000)
            if code not in out:
                # 去掉名稱裡的「小型」前綴讓顯示乾淨;保留一個代表名
                out[code] = {"name": str(r.get("name", "")).replace("小型", "").strip(), "specs": {}}
            out[code]["specs"][m] = {
                "init_amt":  int(r["init_amt"]) if pd.notna(r.get("init_amt")) else None,
                "maint_amt": int(r["maint_amt"]) if pd.notna(r.get("maint_amt")) else None,
                "eng":       str(r.get("eng_code", "") or "").strip(),
            }
        return out
    except Exception:
        return {}

@st.cache_data(ttl=300, show_spinner=False)
def _load_latest_close_map():
    """從 daily 快取建 {stock_id: 最新收盤},給保證金頁自動帶價。"""
    try:
        files = sorted(CACHE_DIR.glob("daily_*.parquet"))
        if not files:
            return {}
        d = pd.read_parquet(files[-1], columns=["stock_id", "date", "close"])
        d = d.sort_values("date").drop_duplicates("stock_id", keep="last")
        return {str(r.stock_id): float(r.close) for r in d.itertuples() if pd.notna(r.close)}
    except Exception:
        return {}

@st.cache_data(ttl=600, show_spinner=False)
def _load_bias_map(cache_key: str):
    """從 daily 快取算每檔的乖離率 → {stock_id: {"bias20","bias60","hot"}}。
    bias20 = (最新收盤 − MA20)/MA20×100;bias60 同理用季線 MA60(資料不足 60 筆則 None)。
    hot = 乖離過熱(MA20>BIAS_MA20_HOT 或 季線>BIAS_MA60_HOT)→ 選股表標 🟡 避免追高。
    cache_key 傳 daily 日期字串,日期一變自動重算。"""
    try:
        files = sorted(CACHE_DIR.glob("daily_*.parquet"))
        if not files:
            return {}
        d = pd.read_parquet(files[-1], columns=["stock_id", "date", "close"])
        if d.empty:
            return {}
        d["stock_id"] = d["stock_id"].astype(str)
        d = d.drop_duplicates(subset=["date", "stock_id"], keep="last")
        close = d.pivot(index="date", columns="stock_id", values="close").sort_index()
        ma20 = close.tail(20).mean()                      # 各股近 20 日均價(Series,index=stock_id)
        ma60 = close.tail(60).mean() if len(close) >= 60 else None
        latest = close.ffill().iloc[-1]                   # 各股最新收盤(補洞避免最後一日缺值)
        out = {}
        for sid in close.columns:
            lt, m20 = latest.get(sid), ma20.get(sid)
            if pd.isna(lt) or pd.isna(m20) or m20 <= 0:
                continue
            b20 = (lt - m20) / m20 * 100
            b60 = None
            if ma60 is not None:
                m60 = ma60.get(sid)
                if pd.notna(m60) and m60 > 0:
                    b60 = (lt - m60) / m60 * 100
            hot = (b20 > BIAS_MA20_HOT) or (b60 is not None and b60 > BIAS_MA60_HOT)
            out[str(sid)] = {"bias20": round(b20, 1),
                             "bias60": (round(b60, 1) if b60 is not None else None),
                             "hot": bool(hot)}
        return out
    except Exception:
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def _load_idx_margin_map():
    """讀 TAIFEX 台指期保證金 → {key: {product, pv, init, maint}}。"""
    try:
        files = sorted(CACHE_DIR.glob("idx_margin_*.parquet"))
        if not files:
            return {}
        d = pd.read_parquet(files[-1])
        out = {}
        for _, r in d.iterrows():
            out[str(r["key"])] = {
                "product": str(r.get("product", "")),
                "pv":   int(r.get("point_value", 0) or 0),
                "init": int(r["init_margin"]) if pd.notna(r.get("init_margin")) else None,
                "maint": int(r["maint_margin"]) if pd.notna(r.get("maint_margin")) else None,
            }
        return out
    except Exception:
        return {}

with _tab_margin:
    st.caption(
        "算期貨的**保證金、槓桿、對應現股/指數本金**,評估「用期貨槓桿操作」要準備多少錢、槓桿多大。"
        "資料來自期交所(TAIFEX)。"
    )
    _fmode = st.radio("類型", ["個股期貨", "ETF 期貨", "台指期(大/小/微台)"],
                      horizontal=True, key="_margin_mode")

    if _fmode == "個股期貨":
        _sf_map = _load_stock_futures_map()
        if not _sf_map:
            st.info("尚無個股期貨標的快取——需 `fetch_cache.py` 抓過 TAIFEX 一次後才有(部署後跑一次即可)。")
        else:
            _close_map = _load_latest_close_map()
            # 保證金比例快取日期(供使用者判斷新舊;週更,故附官網連結方便核對最新)
            _mgn_files = sorted(CACHE_DIR.glob("stock_margin_*.parquet"))
            _mgn_date = (_mgn_files[-1].stem.replace("stock_margin_", "") if _mgn_files else None)
            _date_note = (f"保證金比例資料日期 **{_mgn_date}**" if _mgn_date
                          else "⚠️ 尚無官方保證金比例快取(暫用預設 13.5%)")
            # 契約數 = 各股的契約規格(標準/小型)加總;與官網「以契約計」的列數對齊
            _n_contracts = sum(len(v.get("mults") or [1]) for v in _sf_map.values())
            st.caption(
                f"目前共 **{len(_sf_map)} 檔股票**有個股期貨可交易"
                f"(含標準/小型契約共 **{_n_contracts}** 個;官網以**契約**計,故列數會比檔數多——"
                f"同一檔的標準型與小型契約在官網分開列,在這裡是進去後的「契約規格」選項)。{_date_note}。"
                f"　🔗 [上 TAIFEX 官網核對最新保證金](https://www.taifex.com.tw/cht/5/stockMarginingDetail)"
                f"(比例每週更新一次,連假前/大波動後若有調整,以官網為準)。"
            )
            _mc1, _mc2 = st.columns([2, 1])
            _q_in = _mc1.text_input("股票代號 / 名稱", value="", placeholder="例 2330 或 台積",
                                    key="_margin_sid",
                                    help="可輸入代號(例 2330)或名稱關鍵字(例 台積);名稱比對僅限有個股期貨的標的").strip()
            _lots = _mc2.number_input("口數", min_value=1, max_value=1000, value=1, step=1,
                                      key="_margin_lots")
            # 解析輸入:純代號直接用;否則當名稱關鍵字,在個股期貨清單裡模糊比對
            _sid_in = ""
            if _q_in:
                if _q_in in _sf_map:                      # 完全命中代號
                    _sid_in = _q_in
                elif _q_in.isdigit():                     # 純數字但不在清單 → 視為「無個股期貨的代號」
                    _sid_in = _q_in
                else:                                     # 名稱關鍵字 → 比對 _sf_map 的 name
                    _matches = [(s, v["name"]) for s, v in _sf_map.items() if _q_in in str(v["name"])]
                    if len(_matches) == 1:
                        _sid_in = _matches[0][0]
                    elif len(_matches) > 1:
                        _opts = [f"{s} {n}" for s, n in _matches[:30]]
                        _pick = _mc1.selectbox(f"找到 {len(_matches)} 檔含「{_q_in}」,請選擇",
                                               _opts, key="_margin_name_pick")
                        _sid_in = _pick.split()[0] if _pick else ""
                    else:
                        st.warning(f"⚠️ 個股期貨標的中找不到名稱含「{_q_in}」的股票。"
                                   f"請改輸入代號,或確認該股是否有個股期貨。")
            if _sid_in:
                if _sid_in not in _sf_map:
                    st.warning(f"⚠️ **{_sid_in} 沒有個股期貨**(不在 TAIFEX 標的清單,無法用期貨交易)。"
                               f"目前只有約 {len(_sf_map)} 檔較活絡的股票有個股期貨。")
                else:
                    _info = _sf_map[_sid_in]
                    _nm = _info["name"]
                    _mults = _info["mults"] or [2000]
                    _has_small = 100 in _mults
                    _auto_price = _close_map.get(_sid_in)
                    _p1, _p2, _p3 = st.columns(3)
                    _price_in = _p1.number_input(
                        "期貨價格(≈股價)", min_value=0.0,
                        value=round(_auto_price, 2) if _auto_price else 0.0, step=0.05,
                        key=f"_margin_price_{_sid_in}",
                        help="預設帶入最新收盤,可自行改成你的進場價")
                    # 有小型契約就讓使用者選標準/小型;否則直接用唯一乘數
                    if len(_mults) > 1:
                        _mc_lbl = {m: (f"小型 {m} 股" if m == 100 else f"標準 {m} 股") for m in _mults}
                        _mult_pick = _p2.radio("契約規格", _mults,
                                               format_func=lambda m: _mc_lbl[m],
                                               key=f"_margin_spec_{_sid_in}",
                                               help="此標的同時有標準型與小型契約,選一個試算")
                        _mult_in = int(_mult_pick)
                    else:
                        _mult_in = int(_mults[0])
                        _p2.metric("契約乘數(股/口)", f"{_mult_in:,}")
                    # 官方逐檔原始保證金比例(有 stock_margin 快取時自動帶入,否則退回 13.5)
                    _off_init  = _info.get("init_rate")
                    _off_maint = _info.get("maint_rate")
                    _tier      = _info.get("tier") or ""
                    _rate_default = float(_off_init) if _off_init else 13.5
                    _rate_help = (
                        f"已自動帶入 TAIFEX 官方比例({_tier}:{_off_init:.2f}%);可自行覆寫"
                        if _off_init else
                        "期交所依個股風險分級,常見 13.5% / 16.2% / 20.25%;"
                        "(本檔暫無官方比例快取,使用預設 13.5%,請依該檔實際適用比例調整)"
                    )
                    # key 帶上官方比例:當該檔官方比例(_rate_default)變動時 key 跟著變,
                    # 強制 number_input 重新以新 value 建立——否則 Streamlit 會沿用 session_state
                    # 舊值、忽略 value=(曾發生:parquet 未同步前先搜過 → 卡在 13.5 不跳官方比例)。
                    _rate_in = _p3.number_input(
                        "原始保證金比例 %", min_value=1.0, max_value=100.0,
                        value=_rate_default, step=0.05,
                        key=f"_margin_rate_{_sid_in}_{_rate_default:g}", help=_rate_help)
                    if _price_in > 0:
                        _contract_val = _price_in * _mult_in * _lots
                        _init_margin  = _contract_val * _rate_in / 100
                        # 維持保證金:使用者沒改原始比例 → 直接套官方維持比例;改過 → 用官方維持/原始的比值,
                        # 都沒有官方資料才退回 ~0.767 估算
                        if _off_maint and _off_init:
                            if abs(_rate_in - _off_init) < 1e-6:
                                _maint_margin = _contract_val * _off_maint / 100
                            else:
                                _maint_margin = _init_margin * (_off_maint / _off_init)
                        else:
                            _maint_margin = _init_margin * 0.767
                        _leverage     = 100 / _rate_in
                        _eng = _info.get("eng", {}).get(_mult_in, "")
                        st.divider()
                        _eng_tag = f"（{_eng}）" if _eng else ""
                        st.markdown(f"#### {_sid_in} {_nm}{_eng_tag}　{_lots} 口 × {_mult_in:,} 股 @ {_price_in:,.2f}")
                        _r = st.columns(4)
                        _r[0].metric("契約總值", f"{_contract_val:,.0f}", help="= 買等量現股要花的錢(元)")
                        _r[1].metric("原始保證金", f"{_init_margin:,.0f}", help="進場需準備的保證金(元)")
                        _maint_help = (f"低於此值會被追繳;官方維持比例 {_off_maint:.2f}%"
                                       if _off_maint else "低於此值會被追繳;約為原始的 ~77%(各風險級接近)")
                        _r[2].metric("維持保證金", f"{_maint_margin:,.0f}", help=_maint_help)
                        _r[3].metric("槓桿倍數", f"{_leverage:.1f} 倍", help="= 100 ÷ 保證金比例")
                        st.caption(
                            f"💡 用期貨只需 **{_init_margin:,.0f} 元**保證金,就能操作 **{_contract_val:,.0f} 元**的部位"
                            f"(買現股要全額)。槓桿 **{_leverage:.1f} 倍**是兩面刃——同步放大獲利與虧損,"
                            f"且跌破維持保證金會被**追繳**,沒補就強制平倉。"
                        )
                        if _off_init:
                            st.caption(
                                f"✅ 保證金比例已自動帶入 **TAIFEX 官方**逐檔資料"
                                f"({_tier}:原始 {_off_init:.2f}% / 維持 {_off_maint:.2f}%)。"
                                f"期交所會依波動不定期調整,本頁每週更新;"
                                f"[查官網最新](https://www.taifex.com.tw/cht/5/stockMarginingDetail)。"
                            )
                        else:
                            st.caption(
                                "⚠️ 本檔暫無官方比例快取(跑過 `fetch_cache.py` 後即有),目前用預設 13.5%。"
                                "保證金比例依個股風險分級,期交所定期調整,"
                                "請以 [TAIFEX 官網](https://www.taifex.com.tw/cht/5/stockMarginingDetail)/你的券商公告為準。"
                            )

    elif _fmode == "ETF 期貨":
        # ── ETF 期貨:保證金為期交所公告「固定金額(元/口)」,非比例;契約值用 ETF 價格 × 乘數 ──
        _etf_map = _load_etf_futures_map()
        if not _etf_map:
            st.info("尚無 ETF 期貨快取——需跑過新版 `fetch_cache.py` 一次才有(部署後跑一次即可)。")
        else:
            _close_map = _load_latest_close_map()
            _efiles = sorted(CACHE_DIR.glob("etf_futures_*.parquet"))
            _edate = _efiles[-1].stem.replace("etf_futures_", "") if _efiles else None
            st.caption(
                f"目前共 **{len(_etf_map)} 檔 ETF** 有期貨可交易"
                f"(ETF 期貨保證金是 **TAIFEX 公告固定金額(元/口)**,不是比例%)。"
                + (f"資料日期 **{_edate}**。" if _edate else "")
                + "　🔗 [上 TAIFEX 官網核對最新保證金](https://www.taifex.com.tw/cht/5/stockMarginingDetail)"
                  "(每週更新一次,以官網為準)。"
            )
            _ec1, _ec2 = st.columns([2, 1])
            _eq = _ec1.text_input("ETF 代號 / 名稱", value="", placeholder="例 0050 或 台灣50",
                                  key="_etf_q",
                                  help="可輸入代號(例 0050)或名稱關鍵字(例 台灣50);名稱比對僅限有 ETF 期貨的標的").strip()
            _elots = _ec2.number_input("口數", min_value=1, max_value=1000, value=1, step=1, key="_etf_lots")
            # 解析:純代號(00 開頭)直接用;否則名稱關鍵字模糊比對
            _ecode = ""
            if _eq:
                if _eq in _etf_map:
                    _ecode = _eq
                elif _eq.startswith("00"):
                    _ecode = _eq
                else:
                    _ms = [(c, v["name"]) for c, v in _etf_map.items() if _eq in str(v["name"])]
                    if len(_ms) == 1:
                        _ecode = _ms[0][0]
                    elif len(_ms) > 1:
                        _opts = [f"{c} {n}" for c, n in _ms[:30]]
                        _pk = _ec1.selectbox(f"找到 {len(_ms)} 檔含「{_eq}」,請選擇", _opts, key="_etf_pick")
                        _ecode = _pk.split()[0] if _pk else ""
                    else:
                        st.warning(f"⚠️ ETF 期貨中找不到名稱含「{_eq}」的標的。請改輸入代號。")
            if _ecode:
                if _ecode not in _etf_map:
                    st.warning(f"⚠️ **{_ecode} 沒有 ETF 期貨**(不在 TAIFEX 清單)。")
                else:
                    _einfo  = _etf_map[_ecode]
                    _enm    = _einfo["name"]
                    _especs = _einfo["specs"]              # {mult: {init_amt, maint_amt}}
                    _emults = sorted(_especs.keys())
                    _auto   = _close_map.get(_ecode)
                    _p1, _p2, _p3 = st.columns(3)
                    _eprice = _p1.number_input(
                        "ETF 價格(≈市價)", min_value=0.0,
                        value=round(_auto, 2) if _auto else 0.0, step=0.01,
                        key=f"_etf_price_{_ecode}",
                        help="預設帶入最新收盤(若快取有),可自行改成你的進場價")
                    if len(_emults) > 1:
                        _lbl = {m: (f"小型 {m:,} 股" if m == 1000 else f"標準 {m:,} 股") for m in _emults}
                        _emult = int(_p2.radio("契約規格", _emults, format_func=lambda m: _lbl[m],
                                               key=f"_etf_spec_{_ecode}",
                                               help="此 ETF 同時有標準型與小型契約,選一個試算"))
                    else:
                        _emult = int(_emults[0])
                        _p2.metric("契約乘數(股/口)", f"{_emult:,}")
                    _einit_amt  = _especs[_emult]["init_amt"]
                    _emaint_amt = _especs[_emult]["maint_amt"]
                    _p3.metric("原始保證金(元/口)", f"{_einit_amt:,}" if _einit_amt else "—",
                               help="TAIFEX 公告固定金額;ETF 期貨用金額計、非比例")
                    if _eprice > 0 and _einit_amt:
                        _ecv    = _eprice * _emult * _elots
                        _einit  = _einit_amt * _elots
                        _emaint = (_emaint_amt * _elots) if _emaint_amt else None
                        _elev   = _ecv / _einit if _einit > 0 else 0
                        _eeng = _especs[_emult].get("eng", "")
                        st.divider()
                        _eeng_tag = f"（{_eeng}）" if _eeng else ""
                        st.markdown(f"#### {_ecode} {_enm}{_eeng_tag}　{_elots} 口 × {_emult:,} 股 @ {_eprice:,.2f}")
                        _r = st.columns(4)
                        _r[0].metric("契約總值", f"{_ecv:,.0f}", help="= ETF 價格 × 乘數 × 口數(元)")
                        _r[1].metric("原始保證金", f"{_einit:,.0f}", help="進場需準備(公告固定金額 × 口數)")
                        _r[2].metric("維持保證金", f"{_emaint:,.0f}" if _emaint else "—",
                                     help="低於此值會被追繳")
                        _r[3].metric("槓桿倍數", f"{_elev:.1f} 倍", help="= 契約總值 ÷ 原始保證金")
                        st.caption(
                            f"💡 用 ETF 期貨只需 **{_einit:,.0f} 元**保證金,就能操作 **{_ecv:,.0f} 元**的部位。"
                            f"槓桿 **{_elev:.1f} 倍**會同步放大損益,跌破維持保證金會被**追繳**。"
                        )
                        st.caption(
                            "✅ ETF 期貨保證金為 **TAIFEX 公告固定金額(元/口)**(非比例),每週更新;"
                            "[查官網最新](https://www.taifex.com.tw/cht/5/stockMarginingDetail)。"
                        )

    else:
        # ── 台指期(大/小/微台):保證金為期交所公告固定金額,契約值用加權指數 ──
        _idx = _load_idx_margin_map()
        if not _idx:
            st.info("尚無台指期保證金快取——需 `fetch_cache.py` 抓過 TAIFEX 一次後才有(部署後跑一次即可)。")
        else:
            _taiex = None
            try:
                _tw = _load_twii_cached()
                if _tw is not None and not _tw.empty and "Close" in _tw.columns:
                    _taiex = float(_tw["Close"].dropna().iloc[-1])
            except Exception:
                pass
            _ic1, _ic2 = st.columns([2, 1])
            _prod = _ic1.radio("商品", ["大台", "小台", "微台"], horizontal=True, key="_idx_prod")
            _ilots = _ic2.number_input("口數", min_value=1, max_value=1000, value=1, step=1, key="_idx_lots")
            _meta = _idx.get(_prod, {})
            _pv = _meta.get("pv") or {"大台": 200, "小台": 50, "微台": 10}[_prod]
            _ip1, _ip2 = st.columns(2)
            _taiex_in = _ip1.number_input(
                "加權指數", min_value=0.0, value=round(_taiex, 2) if _taiex else 0.0, step=1.0,
                key="_idx_taiex", help="預設帶入最新加權指數(證交所官方)")
            _init_def = _meta.get("init")
            _init_in = _ip2.number_input(
                "原始保證金(元/口)", min_value=0, value=int(_init_def) if _init_def else 0, step=1000,
                key=f"_idx_init_{_prod}", help="取自期交所最新公告;會依波動定期調整")
            if _taiex_in > 0 and _init_in > 0:
                _cv = _taiex_in * _pv * _ilots
                _tot_init = _init_in * _ilots
                _maint = _meta.get("maint")
                _tot_maint = (_maint * _ilots) if _maint else None
                _lev = _cv / _tot_init
                st.divider()
                st.markdown(f"#### {_prod}({_meta.get('product','')})　{_ilots} 口 × {_pv} 元/點 @ 指數 {_taiex_in:,.0f}")
                _r = st.columns(4)
                _r[0].metric("契約總值", f"{_cv:,.0f}", help="= 加權指數 × 每點價值 × 口數(元)")
                _r[1].metric("原始保證金", f"{_tot_init:,.0f}", help="進場需準備(期交所公告金額 × 口數)")
                _r[2].metric("維持保證金", f"{_tot_maint:,.0f}" if _tot_maint else "—",
                             help="低於此值會被追繳")
                _r[3].metric("槓桿倍數", f"{_lev:.1f} 倍", help="= 契約總值 ÷ 原始保證金")
                st.caption(
                    f"💡 {_prod} 1 口保證金 **{_init_in:,.0f} 元**就能操作每口 **{_taiex_in*_pv:,.0f} 元**的指數部位,"
                    f"槓桿約 **{_lev:.1f} 倍**——比個股期貨高很多,指數動 1% 對本金影響放大約 {_lev:.0f}%。務必嚴設停損。"
                )
                st.caption("⚠️ 保證金金額由期交所依波動定期調整,本頁取自 TAIFEX 最新快取;實際以期交所公告為準。")

# ── 💰 資金配置(把今日達標股 → 各買多少) ────────────────────────────────
with _tab_alloc:
    st.caption(
        "回答「**各買多少**」:依「每筆固定風險 %」反推建議張數(停損用 ATR),"
        "並控管**單檔 / 單一產業集中度**。把選股清單變成可執行的部位配置。"
    )

    with st.expander("📖 使用說明 / FAQ", expanded=False):
        st.markdown("""
### 🎯 這頁在做什麼?
把「今日選出來的股」變成**實際能下單的張數/口數**——回答「**每檔各買多少、會不會壓太多在同產業、最壞會虧多少**」。

核心一句話:**先想好「每筆最多能虧多少錢」,再回推該買幾張**,而不是憑感覺亂買。

---

### 🧮 它怎麼算的(風險反推)
1. **每筆風險預算** = 總資金 × 單筆風險%（例:30 萬 × 1.5% = 4,500 元)
2. **停損幅度** = ATR% × 停損 ATR 倍數（個股波動越大、停損抓越寬)
3. **可投入金額** = 風險預算 ÷ 停損幅度 → 觸及停損時,虧損 ≈ 你設定的風險%
4. 再受 **單檔上限 / 產業上限** 收斂,最後換成整數張(或零股 / 口數)

> 精神:**先想「能虧多少」,再決定「買多少」**。每檔停損都只虧總資金的固定 %,單筆爆掉也不傷本金。

---

### ⚙️ 參數說明
| 參數 | 意義 | 建議 |
|---|---|---|
| 單筆風險 % | 每檔觸停損最多虧總資金的 % | 保守 1%、一般 1.5~2% |
| 停損 ATR 倍數 | 停損 = ATR% × 此倍數 | 2 倍常見;放大=停損更寬、張數更少 |
| 單檔上限 % | 單一個股投入上限 | 避免單押,常 15~25% |
| 單一產業上限 % | 同產業合計上限 | 避免全押同類股,常 30~40% |
| ATR 缺值預設停損 % | 抓不到 ATR 時的後備停損 | 預設 8% |

---

### 🆚 現股 vs 個股期貨
- **現股**:買股票,佔用**全額**。可選**零股**(任意股數,小資金也配得進高價股)或**整張**(1000 股倍數)。
- **個股期貨**:用**保證金**(約 13.5%)槓桿操作,僅限**有個股期貨的標的**;以**小型契約(100 股/口)**為主,1 口曝險小、適合小資金。

### ❓ 為什麼有些股「建議 0 口 / 未列入」?
- **現股(整張)**:1 張高價股(如台積電 1 張 = 100 萬)超過單檔上限 → 改用**零股**即可。
- **個股期貨**:1 口的曝險就讓「停損虧損」超過你的單筆風險%(高價股 1 口很大)→ **提高單筆風險%、增加總資金,或改用現股零股**。

---

### 🔗 跟其他頁的分工
- **這頁(資金配置)**= 組合層級:今日整批各買多少、別過度集中。
- **個股「💰 資金管理」**= 單筆精算:用你自己的停損價算單一檔。可從本頁**點表格列**跳過去。
- **「🧮 期貨保證金」**= 單一商品的保證金/槓桿試算。

---

### ⚠️ 重要提醒
- 這是**機械式風險試算**,非投資建議;ATR 停損僅供參考,實際請用你的紀律。
- 「總風險暴露」= 若**所有部位同時觸停損**會虧多少(壓力情境),正常不會全中,但用來確認最壞狀況可承受。
        """)

    if df is None or df.empty:
        st.info("先按左側「開始選股」產生今日達標清單,才能做資金配置。")
    else:
        _ac = st.columns(5)
        _cap   = _ac[0].number_input("總資金(元)", min_value=10000, value=300_000,
                                     step=50_000, key="_alloc_cap")
        _risk  = _ac[1].number_input("單筆風險 %", min_value=0.1, max_value=10.0, value=1.5,
                                     step=0.1, key="_alloc_risk",
                                     help="每檔若觸及停損,最多虧總資金的這個 %(專業常用 1~2%)")
        _atrm  = _ac[2].number_input("停損 ATR 倍數", min_value=0.5, max_value=6.0, value=2.0,
                                     step=0.5, key="_alloc_atr",
                                     help="停損幅度 = ATR% × 倍數;停損價 = 現價 ×(1 − 停損幅度)")
        _pcap  = _ac[3].number_input("單檔上限 %", min_value=1, max_value=100, value=20,
                                     step=5, key="_alloc_pcap", help="單一個股投入上限(佔總資金)")
        _icap  = _ac[4].number_input("單一產業上限 %", min_value=5, max_value=100, value=40,
                                     step=5, key="_alloc_icap", help="同產業合計投入上限;超標自動等比例縮減")

        _mode = st.radio("商品類型", ["現股", "個股期貨"], horizontal=True, key="_alloc_mode",
                         help="現股=買股票(全額/零股);個股期貨=用保證金槓桿操作(僅限有個股期貨的標的)")
        _smin, _smax = int(df['總分'].min()), int(df['總分'].max())
        _minscore = st.slider("只配置總分 ≥", _smin, _smax, _smin) if _smin != _smax else _smin
        _risk_budget = _cap * _risk / 100.0
        _single_cap  = _cap * _pcap / 100.0
        _ind_cap     = _cap * _icap / 100.0
        _work = df[df['總分'] >= _minscore].copy()

        def _stop_of(r):
            _atr = r.get('ATR%')
            v = (_atrm * _atr) if (_atr is not None and pd.notna(_atr) and _atr > 0) else _fb_stop
            return max(float(v), 0.5)

        if _mode == "現股":
            _sc = st.columns(2)
            _unit = _sc[0].radio(
                "交易單位", ["零股(可買零股,小資金友善)", "整張(1 張 = 1000 股)"],
                horizontal=True, key="_alloc_unit",
                help="零股=盤中零股,任意股數,小資金也配得進高價股;整張=1000 股倍數,高價股需大資金")
            _fb_stop = _sc[1].number_input("ATR 缺值時的預設停損 %", min_value=1.0, max_value=30.0,
                                           value=8.0, step=0.5, key="_alloc_fbstop_s")
            _odd = _unit.startswith("零股")
            _rows = []
            for _, r in _work.iterrows():
                _price = r.get('現價')
                if _price is None or pd.isna(_price) or _price <= 0:
                    continue
                _stop_pct = _stop_of(r)
                _invest = min(_risk_budget / (_stop_pct / 100.0), _single_cap)
                _rows.append({"代號": str(r['代號']), "名稱": r.get('名稱', ''),
                              "產業": (r.get('產業') or '其他'), "price": float(_price),
                              "stop_pct": _stop_pct, "invest": _invest})
            if not _rows:
                st.warning("沒有可配置的標的(可能現價缺失,或都低於設定總分)。")
            else:
                _ind_tot = {}
                for x in _rows:
                    _ind_tot[x["產業"]] = _ind_tot.get(x["產業"], 0) + x["invest"]
                _over = {k: v for k, v in _ind_tot.items() if v > _ind_cap}
                for x in _rows:
                    if x["產業"] in _over:
                        x["invest"] *= _ind_cap / _ind_tot[x["產業"]]
                _out = []
                for x in _rows:
                    _shares = int(x["invest"] // x["price"]) if _odd else int(x["invest"] // (x["price"]*1000))*1000
                    _actual = _shares * x["price"]
                    _out.append({**x, "shares": _shares, "actual": _actual,
                                 "risk_amt": _actual * x["stop_pct"]/100.0,
                                 "stop_price": x["price"]*(1-x["stop_pct"]/100.0)})
                _placed = [o for o in _out if o["shares"] >= 1]
                _skipped = [o for o in _out if o["shares"] < 1]
                _tot_invest = sum(o["actual"] for o in _placed)
                _tot_risk   = sum(o["risk_amt"] for o in _placed)
                _m = st.columns(4)
                _m[0].metric("建議檔數", f"{len(_placed)} 檔")
                _m[1].metric("總投入", f"{_tot_invest:,.0f}", f"{_tot_invest/_cap*100:.0f}% 資金")
                _m[2].metric("保留現金", f"{_cap-_tot_invest:,.0f}", f"{(_cap-_tot_invest)/_cap*100:.0f}%")
                _m[3].metric("總風險暴露", f"{_tot_risk:,.0f}", f"全停損虧 {_tot_risk/_cap*100:.1f}%", delta_color="off")
                if _placed:
                    _qty_label = "建議股數" if _odd else "建議張數"
                    _disp = pd.DataFrame([{
                        "代號": o["代號"], "名稱": o["名稱"], "產業": o["產業"],
                        "現價": f"{o['price']:,.2f}", "停損價": f"{o['stop_price']:,.2f}",
                        _qty_label: (o["shares"] if _odd else o["shares"]//1000),
                        "投入金額": f"{o['actual']:,.0f}", "佔資金%": f"{o['actual']/_cap*100:.1f}%",
                        "該筆風險": f"{o['risk_amt']:,.0f}",
                    } for o in sorted(_placed, key=lambda z: z["actual"], reverse=True)])
                    _alloc_event = st.dataframe(_disp, use_container_width=True, hide_index=True,
                                                on_select="rerun", selection_mode="single-row", key="_alloc_table")
                    st.caption("💡 **點表格任一列** → 下方個股分析區跳到該檔(現價自動帶入),可再進「💰 資金管理」精算。")
                    _asel = _alloc_event.selection.rows
                    if _asel != st.session_state.get("_alloc_last_sel", []):
                        st.session_state["_alloc_last_sel"] = _asel
                        if _asel:
                            st.session_state.target_sid = str(_disp.iloc[_asel[0]]["代號"])
                _ind_final = {}
                for o in _placed:
                    _ind_final[o["產業"]] = _ind_final.get(o["產業"], 0) + o["actual"]
                if _ind_final:
                    _ind_line = " ｜ ".join(f"{k} {v/_cap*100:.0f}%" for k, v in
                                            sorted(_ind_final.items(), key=lambda z: z[1], reverse=True))
                    st.caption(f"🏭 產業佔比:{_ind_line}")
                    if _over:
                        st.caption(f"⚠️ 已自動把超過 {_icap}% 上限的產業({', '.join(_over)})等比例縮減。")
                if _skipped:
                    _u = "1 股" if _odd else "1 張"
                    st.caption(f"ℹ️ {len(_skipped)} 檔因「資金買不到 {_u}」未列入:"
                               f"{', '.join(o['代號'] for o in _skipped[:8])}" + ("…" if len(_skipped) > 8 else "")
                               + ("。試調高總資金/單檔上限,或改用個股期貨。" if not _odd
                                  else "。(零股下仍買不到 = 單檔上限/風險預算太小或股價極高)"))
                st.caption("⚠️ 風險基礎部位試算(每檔虧到停損 ≈ 固定風險%),非投資建議。")

        else:
            # ── 個股期貨模式:列出所有有期貨的標的 + 每口經濟;口數依風險%反推 ──
            _fc = st.columns(2)
            _fb_stop = _fc[0].number_input("ATR 缺值時的預設停損 %", min_value=1.0, max_value=30.0,
                                           value=8.0, step=0.5, key="_alloc_fbstop_f")
            _mrate = _fc[1].number_input("原始保證金比例 %(後備值)", min_value=1.0, max_value=100.0,
                                         value=13.5, step=0.05, key="_alloc_fut_rate",
                                         help="有官方比例的標的會自動套各自的官方比例(13.5/16.2/20.25%…);"
                                              "此值僅用於『無官方比例快取』的標的當後備。")
            _sf_map = _load_stock_futures_map()
            _only_small = st.checkbox("只列有小型個股期貨的標的(100 股/口,小資金友善)",
                                      value=True, key="_alloc_only_small",
                                      help="小型契約 1 口僅 100 股,曝險小、適合小資金;取消勾選則一併列出只有標準型(2000 股)的標的")
            if not _sf_map:
                st.info("尚無個股期貨清單快取(需 `fetch_cache.py` 抓過 TAIFEX 一次),無法用期貨配置。")
            else:
                _fr = []
                for _, r in _work.iterrows():
                    _sid = str(r['代號'])
                    _price = r.get('現價')
                    if _sid not in _sf_map or _price is None or pd.isna(_price) or _price <= 0:
                        continue
                    _mults = _sf_map[_sid].get("mults") or [2000]
                    if _only_small:
                        if 100 not in _mults:      # 「以小型為主」:沒有小型契約的標的不列
                            continue
                        _mult = 100
                    else:
                        _mult = min(_mults)        # 優先用最小契約(小型 100)
                    _spec = "小型" if _mult == 100 else ("標準" if _mult == 2000 else f"{_mult}股")
                    _stop_pct = _stop_of(r)
                    # 逐檔官方原始保證金比例;無官方資料才用使用者後備值 _mrate
                    _row_rate = _sf_map[_sid].get("init_rate") or _mrate
                    _one_notional = _mult * float(_price)
                    _one_margin   = _one_notional * _row_rate / 100.0
                    _one_risk     = _one_notional * _stop_pct / 100.0
                    _lots = int(_risk_budget // _one_risk) if _one_risk > 0 else 0
                    _fr.append({"代號": _sid, "名稱": r.get('名稱', ''), "產業": (r.get('產業') or '其他'),
                                "price": float(_price), "stop_pct": _stop_pct, "spec": _spec, "mult": _mult,
                                "rate": _row_rate,
                                "one_notional": _one_notional, "one_margin": _one_margin,
                                "one_risk": _one_risk, "lots": _lots})
                if not _fr:
                    if _only_small:
                        st.warning("今日達標股中,沒有任何一檔有**小型個股期貨**(100 股/口)。"
                                   "可取消上方勾選,改列含標準型(2000 股)的標的。")
                    else:
                        st.warning("今日達標股中,沒有任何一檔有個股期貨可交易。")
                else:
                    _placed = [x for x in _fr if x["lots"] >= 1]
                    _tot_margin   = sum(x["lots"] * x["one_margin"] for x in _placed)
                    _tot_notional = sum(x["lots"] * x["one_notional"] for x in _placed)
                    _tot_risk     = sum(x["lots"] * x["one_risk"] for x in _placed)
                    _lev = (_tot_notional / _cap) if (_cap > 0 and _tot_notional > 0) else 0
                    _m = st.columns(4)
                    _m[0].metric("總佔用保證金", f"{_tot_margin:,.0f}", f"{_tot_margin/_cap*100:.0f}% 資金")
                    _m[1].metric("總曝險(契約值)", f"{_tot_notional:,.0f}")
                    _m[2].metric("整體槓桿", f"{_lev:.1f} 倍", help="總曝險 ÷ 總資金", delta_color="off")
                    _m[3].metric("總風險暴露", f"{_tot_risk:,.0f}",
                                 f"全停損虧 {_tot_risk/_cap*100:.1f}%", delta_color="off")
                    if _tot_margin > _cap:
                        st.error(f"⚠️ 總保證金 {_tot_margin:,.0f} 已超過總資金 {_cap:,.0f},需減少口數/檔數。")
                    # 列出「所有有期貨的標的」——即使建議 0 口,也看得到每口經濟(避免畫面全空)
                    _fdisp = pd.DataFrame([{
                        "代號": x["代號"], "名稱": x["名稱"], "產業": x["產業"],
                        "契約": f"{x['spec']}({x['mult']:,})", "現價": f"{x['price']:,.2f}",
                        "保證金%": f"{x['rate']:.2f}%",
                        "1口保證金": f"{x['one_margin']:,.0f}",
                        "1口曝險": f"{x['one_notional']:,.0f}",
                        "1口風險%": f"{x['one_risk']/_cap*100:.1f}%",
                        "建議口數": x["lots"],
                    } for x in sorted(_fr, key=lambda z: z["one_notional"])])
                    st.dataframe(_fdisp, use_container_width=True, hide_index=True)
                    _n_off = sum(1 for x in _fr if _sf_map[x["代號"]].get("init_rate"))
                    st.caption(f"📊「保證金%」逐檔取自 TAIFEX 官方({_n_off}/{len(_fr)} 檔有官方資料,"
                               f"其餘用後備值 {_mrate:.2f}%)。")
                    _n0 = sum(1 for x in _fr if x["lots"] == 0)
                    if _only_small:
                        st.caption(
                            f"💡 列出 **{len(_fr)} 檔有小型個股期貨**(100 股/口)的標的。"
                            f"「建議口數」依**每筆風險%**反推;「1口風險%」超過你的單筆風險% → 建議口數=0。")
                    else:
                        _nsmall = sum(1 for x in _fr if x["mult"] == 100)
                        st.caption(
                            f"💡 今日達標中 **{len(_fr)} 檔有個股期貨**(其中 {_nsmall} 檔用小型 100 股/口)。"
                            f"「建議口數」依**每筆風險%**反推;「1口風險%」超過你的單筆風險% → 建議口數=0。")
                    if _n0:
                        st.caption(
                            f"ℹ️ 其中 {_n0} 檔建議 0 口:**1 口的曝險太大**(高價股),風險超過你的單筆風險%設定。"
                            f"可**提高單筆風險%**、**增加總資金**,或這些檔改用「現股(零股)」。")
                    st.caption("⚠️ 風險基礎試算,非投資建議;保證金比例依期交所分級調整,以公告為準。")

with _tab_trend:
    # ── 🔥 入選熱度榜(個股層級) ──
    st.markdown("#### 🔥 入選熱度榜")
    if _hot:
        st.caption(
            f"追蹤**近 {HOT_WINDOW} 個交易日**持續上榜的強勢股 — 連續出現次數越多,趨勢延續性越強。"
            f"(系統歷史保留一年供績效回測,但熱度榜只看近期)"
        )
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
    else:
        st.info("尚未累積足夠歷史紀錄(需 Telegram 每日推播寫入後累積)。")

    # ── 🔄 產業輪動(產業層級) ──
    st.divider()
    st.markdown("#### 🔄 產業輪動")
    if _rotation:
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
    else:
        st.info("歷史資料不足以分析產業輪動。")


# ── 策略績效追蹤(對歷史 picks 算入選後 N 日報酬) ───────────────────────
# (_load_performance_cached 已在首頁速覽卡片前定義,此處直接使用)

with _tab_perf:
    if _hist_days >= 5:  # 至少有 5 天歷史才有意義
        st.caption(
            "回答「我這套系統真的有用嗎?」— 對每筆歷史選股,從 daily 快取算出後續 N 日報酬。"
            "  \n⚙️ **進場假設:訊號日隔日開盤 + 0.1% 滑價**(去除前視偏誤);出場:持有 N 日後收盤;交易成本 0.5%。"
        )

        # ── 🚨 系統失效監控(近期 edge 還在嗎?)──────────────────────────
        # 比較近期 vs 全期的每入選日淨期望值 + 回檔,策略退化時主動示警(符合「何時該停止相信它」)。
        _health = _load_system_health_cached(hold_days=5, recent_window=20)
        _hstatus = _health.get("status")
        _hmsg = f"**{_health.get('label','')}　系統失效監控(近 {_health.get('n_recent',0)} 個入選日)**：{_health.get('reason','')}"
        if _hstatus == "fail":
            st.error(_hmsg)
        elif _hstatus == "warn":
            st.warning(_hmsg)
        elif _hstatus == "ok":
            st.success(_hmsg)
        else:
            st.info(_hmsg)   # insufficient → 累積中

        # ── 📖 指標判讀小抄(預設收合,點開對照) ──
        with st.expander("📖 指標怎麼看?(判讀門檻小抄)", expanded=False):
            st.markdown(
                "**看數據的順序:淨期望值 > 損益比 > 其他**\n\n"
                "| 指標 | 白話 | 判讀 |\n"
                "|------|------|------|\n"
                "| 勝率 | 賺錢的比例 | 高 ≠ 賺錢,要配損益比看 |\n"
                "| 平均報酬 | 每筆平均賺多少(含賺含賠) | 扣成本前的毛數字 |\n"
                "| 中位數 | 正中間那筆 | 比平均低很多 = 少數大賺拉高平均 |\n"
                "| 勝局平均獲利 | 只看賺的平均賺多少 | — |\n"
                "| 敗局平均虧損 | 只看賠的平均賠多少 | **核心:輸時輸多少** |\n"
                "| 損益比 | 總賺 ÷ 總賠 | 賺賠幅度的差距 |\n"
                "| 淨期望值 | 平均報酬 − 0.5% 成本 | **最終極:每筆實際落袋** |\n"
                "| 日均報酬 | 淨期望值 ÷ 持有天數 | **比效率:抱不同天數公平比** |\n"
                "| 最大回檔 | 資金曲線最大跌幅 | 風險:最壞情況會虧多少,越近 0 越穩 |\n"
                "| 夏普值 | 風險調整後報酬(年化) | 賺得穩不穩,>1 不錯 >2 很好 |\n"
            )
            st.markdown(
                "**🎯 淨期望值門檻**　< 0% ❌虧 ｜ 0~1% 🟡勉強 ｜ 1~2.5% ✅好 ｜ > 4% 🔥漂亮但多半是多頭蜜月期,別外推\n\n"
                "**📊 損益比門檻**　< 1 ❌虧 ｜ 1.5 ✅有效 ｜ 2~3 👍不錯 ｜ > 3 ⚠️罕見 ｜ > 5 🚩可能有泡沫\n\n"
                "**📉 夏普值門檻**　< 1 普通 ｜ 1~2 不錯 ｜ 2~3 很好 ｜ > 3 極佳(但短樣本/多頭易虛高,同樣打折看)\n\n"
                "**🔢 樣本可信度**　< 30 看方向就好 ｜ 30~100 有參考價值但偏樂觀 ｜ > 100 且經歷過回檔才算真本事\n\n"
                "**📅 持有天數**　淨期望值看「單筆賺最多」,日均報酬看「效率最高」;兩者不同時優先看日均(同期間短打可多做幾趟)。抱越久樣本越少、越不穩。"
            )
            st.caption(
                "⚠️ 三大陷阱:①勝率高就放心(錯,要配損益比) "
                "②多頭數字當常態(實戰至少打對折估) "
                "③中位數遠低於平均(獲利集中少數幾筆,別過度自信)。"
                "💡 把多頭期漂亮數字當「天花板」不是常態,資金規劃用保守值。"
            )
            st.caption(
                "📐 **回檔/夏普/波動的計算基礎**:用「每入選日平均報酬」序列(每天收斂成一點),"
                "不是逐筆——逐筆會把同日多檔當連續交易、重疊窗口重複複利,把波動灌水。"
                "因此這幾項的「天數」會少於上方明細的「筆數」,口徑與「vs 大盤」圖一致。"
            )

        perf = _load_performance_cached()
        overall = perf.get("overall", {})
        if not overall:
            st.info("資料尚不足以計算績效,建議累積更多天數的選股紀錄後再來看。")
        else:
            # ── 樣本狀態一次性橫幅:「樣本不足」提醒集中在此,下方各表不再逐一重複 ──
            if _hist_days < 30:
                st.warning(
                    f"⚠️ **目前僅累積 {_hist_days} 天選股歷史**(未滿 30 天、尚未經歷完整多空循環),"
                    f"以下所有統計都可能被短期噪音主導——先看方向、別當結論。"
                )

            # ── 二層分頁:9 個區塊分成 3 組,免長捲動 ──
            _sub_overview, _sub_validate, _sub_detail = st.tabs([
                "📅 總覽", "🧪 深入驗證", "📋 個股與樣本明細"
            ])

            with _sub_overview:
                # ── 📅 持有天數比較(找最佳出場時機) ──
                # 同樣本比較:鎖定「已滿最長持有期」的同一批 pick 統計所有持有期,
                # 避免「3 日含最新的單、10 日只剩早期的單」不同批互比(會跟出場回測打架)。
                # 全部持有期都顯示;尚無資料(pick 還沒活夠天數)者標「累積中」。
                _all_periods = (3, 5, 10, 20)
                _cmp_src = perf.get("overall_common") or overall   # 同樣本優先,舊快取退回全樣本
                _cmp_base = _cmp_src.get("base_n")
                if True:
                    st.markdown("**📅 不同持有天數比較(同一批選股,抱幾天最划算?)**")
                    _cmp_rows = []
                    _best_n, _best_exp = None, None       # 淨期望值最高
                    _best_dn, _best_daily = None, None     # 日均報酬最高(效率)
                    _has_pending = False
                    for n in _all_periods:
                        # 該持有期還沒有任何 pick 活夠天數 → 累積中
                        if f"n_{n}d" not in _cmp_src:
                            _has_pending = True
                            _need = max(1, n - _hist_days)   # 約還需幾天(以歷史天數粗估)
                            _cmp_rows.append({
                                "持有天數": f"{n} 日",
                                "樣本數":  0,
                                "勝率":    "⏳ 累積中",
                                "平均報酬": f"約還需 {_need} 天",
                                "淨期望值": "—",
                                "日均報酬": "—",
                                "損益比":  "—",
                            })
                            continue
                        _exp = _cmp_src.get(f"net_expectancy_{n}d")
                        _pf  = _cmp_src.get(f"profit_factor_{n}d")
                        _pf_str = "∞" if _pf == float("inf") else (f"{_pf:.2f}" if _pf is not None else "—")
                        if _exp is not None and (_best_exp is None or _exp > _best_exp):
                            _best_exp, _best_n = _exp, n
                        # 日均報酬 = 淨期望值 ÷ 持有天數(讓不同持有期能公平比較效率)
                        _daily = _exp / n if _exp is not None else None
                        if _daily is not None and (_best_daily is None or _daily > _best_daily):
                            _best_daily, _best_dn = _daily, n
                        _cmp_rows.append({
                            "持有天數": f"{n} 日",
                            "樣本數":  _cmp_src.get(f"n_{n}d", 0),
                            "勝率":    f"{_cmp_src.get(f'win_rate_{n}d', 0)*100:.0f}%",
                            "平均報酬": f"{_cmp_src.get(f'avg_return_{n}d', 0):+.2f}%",
                            "淨期望值": f"{_exp:+.2f}%" if _exp is not None else "—",
                            "日均報酬": f"{_daily:+.2f}%" if _daily is not None else "—",
                            "損益比":  _pf_str,
                        })
                    st.dataframe(pd.DataFrame(_cmp_rows), use_container_width=True, hide_index=True)
                    if _cmp_base:
                        st.caption(
                            f"📌 **同樣本比較**:只統計「已滿 {_cmp_base} 個交易日」的同一批 pick"
                            f"({_cmp_src.get('n_picks', 0)} 筆),所有持有期比的是同一批股票、同一段時期。"
                            f"最新還沒活滿 {_cmp_base} 日的 pick 不納入(它們的短天期報酬會出現在"
                            f"下方「各持有期詳細指標」的全樣本統計裡)。"
                        )
                    if _has_pending:
                        st.caption("⏳ 標「累積中」的持有期,是因為最早入選的 pick 還沒滿該天數;歷史累積足夠後會自動補上。")
                    if _best_n is not None:
                        _eff_note = ""
                        if _best_dn is not None and _best_dn != _best_n:
                            _eff_note = (
                                f" 但若看**日均報酬**(效率),**持有 {_best_dn} 日最高({_best_daily:+.2f}%/日)**——"
                                f"同一段時間內短打可以做更多趟,整體可能更賺。"
                            )
                        st.caption(
                            f"💡 以**淨期望值**(扣成本後每筆落袋)來看,**持有 {_best_n} 日最高({_best_exp:+.2f}%)**。"
                            f"{_eff_note}"
                            f" 提醒:抱越久樣本越少(最近的 pick 還沒滿天數),數字穩定度較低。"
                        )
                    st.divider()

                # 各持有期詳細指標(收進 expander,預設收起;標題即帶關鍵數字)
                # 風險指標(回檔/夏普/波動)用「每入選日平均報酬」序列算,點數 = risk_n。
                # 點數太少時,這幾項會出現「回檔 0.00% / 夏普飆到兩位數」這種小樣本假象
                # (全多頭、重疊窗口、年化係數放大),故 < RISK_MIN_N 時不顯示精確值,改標「樣本不足」。
                RISK_MIN_N = 20
                st.markdown("**📂 各持有期詳細指標**(點開看賺賠結構與風險面)")
                for n_days in (3, 5, 10, 20):
                    key_n = f"n_{n_days}d"
                    if key_n not in overall:
                        continue
                    # 標題摘要:收合狀態也能一眼看到重點
                    _wr  = overall.get(f"win_rate_{n_days}d", 0) * 100
                    _exp = overall.get(f"net_expectancy_{n_days}d", 0)
                    _shp = overall.get(f"sharpe_{n_days}d", 0)
                    _risk_n_hdr = overall.get(f"risk_n_{n_days}d", 0)
                    _shp_str = f"{_shp:.2f}" if _risk_n_hdr >= RISK_MIN_N else "—"
                    _label = (f"入選後 {n_days} 日　勝率 {_wr:.0f}% ｜ 淨期望值 {_exp:+.2f}% "
                              f"｜ 夏普 {_shp_str} ｜ 樣本 {overall[key_n]} 筆")
                    with st.expander(_label, expanded=False):
                        # 第一排:報酬
                        m_cols = st.columns(4)
                        m_cols[0].metric("樣本", f"{overall[key_n]} 筆")
                        m_cols[1].metric("勝率", f"{overall[f'win_rate_{n_days}d']*100:.0f}%")
                        avg_ret = overall[f"avg_return_{n_days}d"]
                        m_cols[2].metric("平均報酬", f"{avg_ret:+.2f}%",
                                         delta_color="normal" if avg_ret >= 0 else "inverse")
                        med_ret = overall.get(f"median_return_{n_days}d", 0)
                        m_cols[3].metric("中位數", f"{med_ret:+.2f}%")

                        # 第二排:損益面(輸的時候輸多少、期望值正不正)
                        if f"profit_factor_{n_days}d" in overall:
                            m2 = st.columns(4)
                            avg_gain = overall.get(f"avg_gain_{n_days}d", 0)
                            avg_loss = overall.get(f"avg_loss_{n_days}d", 0)
                            pf       = overall[f"profit_factor_{n_days}d"]
                            m2[0].metric("勝局平均獲利", f"{avg_gain:+.2f}%", delta_color="off")
                            m2[1].metric("敗局平均虧損", f"{avg_loss:+.2f}%", delta_color="off")
                            pf_str = "∞" if pf == float("inf") else f"{pf:.2f}"
                            m2[2].metric("損益比", pf_str,
                                         help="總獲利 ÷ 總虧損。>1.5 算有效,>2 不錯")
                            m2[3].metric("淨期望值", f"{_exp:+.2f}%",
                                         delta_color="normal" if _exp >= 0 else "inverse",
                                         help="平均報酬扣掉來回交易成本(約 0.5%)後,每筆實際落袋。需 >0 才真正划算")

                        # 第三排:風險面(最大回檔、夏普值、波動、最差單筆)
                        # 風險指標以「每入選日平均報酬」序列計算(非逐筆),避免重疊窗口灌水。
                        # 點數 < RISK_MIN_N 時整排隱藏(整排「樣本不足」只是佔版面),
                        # 僅留最差單筆(逐筆口徑,小樣本也有效)+ 一行說明;樣本夠了自動出現。
                        if f"sharpe_{n_days}d" in overall:
                            mdd = overall.get(f"mdd_{n_days}d", 0)
                            std = overall.get(f"std_{n_days}d", 0)
                            risk_n = overall.get(f"risk_n_{n_days}d", 0)
                            min_ret = overall.get(f"min_return_{n_days}d")
                            if risk_n >= RISK_MIN_N:
                                m3 = st.columns(4)
                                m3[0].metric("最大回檔", f"{mdd:.2f}%",
                                             delta_color="off",
                                             help="以每入選日平均報酬串成的資金曲線,從高點到低點最大跌幅。越接近 0 越穩")
                                m3[1].metric("夏普值", f"{_shp:.2f}",
                                             delta_color="off",
                                             help="風險調整後報酬(年化)。>1 不錯,>2 很好,>3 極佳")
                                m3[2].metric("報酬波動(標準差)", f"{std:.2f}%",
                                             delta_color="off",
                                             help="每日平均報酬的起伏程度,越大代表越不穩定")
                                if min_ret is not None:
                                    m3[3].metric("最差單筆", f"{min_ret:+.2f}%", delta_color="off",
                                                 help="這個持有期裡,賠最多的那一筆(逐筆,非每日平均)")
                                st.caption(
                                    f"⚖️ 回檔/夏普/波動以**每入選日平均報酬**計算(共 {risk_n} 個交易日),"
                                    f"已避開「逐筆+重疊窗口」的灌水,口徑與「vs 大盤」走勢圖一致。"
                                )
                            else:
                                if min_ret is not None:
                                    st.columns(4)[0].metric(
                                        "最差單筆", f"{min_ret:+.2f}%", delta_color="off",
                                        help="這個持有期裡,賠最多的那一筆(逐筆口徑,小樣本也有效)")
                                st.caption(
                                    f"📐 風險指標(回檔/夏普/波動)需 ≥ {RISK_MIN_N} 個入選日才有意義;"
                                    f"目前 {risk_n} 天,樣本足夠後會自動顯示。"
                                )

                # ── 📊 系統 picks vs 大盤 後續報酬對照 ─────────────────────────
                # 「跟著系統走 vs 直接買大盤」誰贏?這是現有勝率/平均報酬看不出來的關鍵問題。
                # 每個點 = 該入選日所有 picks 的「5 日後平均報酬」vs 同期 ^TWII。
                # 刻意不做複利(重疊的 5 日窗口複利累加會把報酬灌水),純比點報酬。
                st.divider()
                st.markdown("**📊 跟著系統走 vs 直接買大盤(每點 = 當日入選股的 5 日後平均報酬)**")
                _eq = None
                _eq_error = None
                try:
                    _eq = compute_equity_curve(load_history(), CACHE_DIR, hold_days=5)
                except Exception as _e:
                    _eq_error = str(_e)
                    print(f"⚠ equity curve 計算失敗: {_e}")

                # ── 三種狀態的明確提示 ──
                if _eq_error:
                    st.error(f"❌ 累積曲線計算錯誤: {_eq_error}")
                elif _eq is None:
                    st.info(
                        "📊 累積績效曲線**資料不足**: 需要至少 8 個交易日的歷史,"
                        "且 daily 快取要能對齊每個 pick 的日期。"
                        "(每筆 pick 都要算進場後 5 個交易日,所以最近 5 天的 pick 都還沒算完)"
                    )
                elif _eq["n_days"] < 3:
                    st.info(
                        f"📊 績效對照**累積中**: 目前只有 **{_eq['n_days']} 個有效資料點**,"
                        f"需要 ≥ 3 個才畫得出有意義的對照。"
                        f"持續累積中,**再過 {max(3 - _eq['n_days'], 1)} 個交易日**會自動出現。"
                    )
                else:
                    # 4 卡統計
                    _avg_pick = _eq["avg_pick"]
                    _avg_twii = _eq["avg_twii"]
                    _alpha = _eq["alpha"]
                    _win_days = _eq["win_days"]
                    _hd = _eq["hold_days"]
                    _ec_cols = st.columns(4)
                    _ec_cols[0].metric(f"系統平均 {_hd} 日報酬", f"{_avg_pick:+.2f}%",
                                        delta_color="off")
                    _ec_cols[1].metric(f"大盤平均 {_hd} 日報酬", f"{_avg_twii:+.2f}%",
                                        delta_color="off")
                    _ec_cols[2].metric("Alpha (系統-大盤)",
                                        f"{_alpha:+.2f} %",
                                        delta_color="off")
                    _ec_cols[3].metric(f"系統贏大盤天數",
                                        f"{_win_days} / {_eq['n_days']} 日",
                                        delta_color="off")

                    # plotly 雙線圖(每點代表「該入選日的 N 日後報酬」)
                    _fig_eq = go.Figure()
                    _fig_eq.add_trace(go.Scatter(
                        x=_eq["dates"], y=_eq["pick_returns"],
                        mode="lines+markers",
                        line=dict(color="#dc2626", width=2.5),
                        marker=dict(size=6),
                        name=f"系統 picks(平均 {_avg_pick:+.2f}%)",
                        hovertemplate=f"<b>%{{x}}</b><br>系統 {_hd} 日報酬:%{{y:+.2f}}%<extra></extra>",
                    ))
                    _fig_eq.add_trace(go.Scatter(
                        x=_eq["dates"], y=_eq["twii_returns"],
                        mode="lines+markers",
                        line=dict(color="#6b7280", width=1.8, dash="dot"),
                        marker=dict(size=5),
                        name=f"大盤 加權指數(平均 {_avg_twii:+.2f}%)",
                        hovertemplate=f"<b>%{{x}}</b><br>大盤 {_hd} 日報酬:%{{y:+.2f}}%<extra></extra>",
                    ))
                    # 0% 基準線
                    _fig_eq.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5,
                                      annotation_text="0%", annotation_position="left")
                    _fig_eq.update_layout(
                        height=320, margin=dict(l=10, r=10, t=20, b=20),
                        yaxis=dict(title=f"{_hd} 日後報酬 (%)", zeroline=True),
                        xaxis=dict(title=""),
                        plot_bgcolor="rgba(0,0,0,0.03)",
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="top", y=-0.1),
                    )
                    st.plotly_chart(_fig_eq, use_container_width=True)
                    st.caption(
                        f"每個點代表「該日入選的 picks 在 {_hd} 個交易日後的平均報酬」。"
                        f"線在 0% 上方表示賺、下方虧。"
                    )

                    # 文字結論(樣本不足的總提醒已集中在頁首橫幅,這裡只標狀態)
                    if _eq["n_days"] < 30:
                        st.caption(
                            f"📊 樣本 {_eq['n_days']} 天(未滿 30 天,見頁首提醒)——"
                            f"alpha {_alpha:+.2f}% 先看方向,別當結論。"
                        )
                    elif _alpha > 0.5:
                        st.success(
                            f"✅ **系統平均贏大盤 {_alpha:+.2f}% / {_hd} 日**"
                            f"(系統 {_avg_pick:+.2f}% vs 大盤 {_avg_twii:+.2f}%),"
                            f"勝日 {_win_days}/{_eq['n_days']}。樣本 {_eq['n_days']} 天,可信度逐步提升中。"
                        )
                    elif _alpha < -0.5:
                        st.warning(
                            f"⚠️ **系統平均輸大盤 {abs(_alpha):.2f}% / {_hd} 日**"
                            f"(系統 {_avg_pick:+.2f}% vs 大盤 {_avg_twii:+.2f}%),"
                            f"勝日 {_win_days}/{_eq['n_days']}。考慮調整訊號組合。"
                        )
                    else:
                        st.info(
                            f"系統表現與大盤接近(Alpha {_alpha:+.2f}%),勝日 {_win_days}/{_eq['n_days']}。"
                            f"沒有明顯超額,可能就跟大盤連動。"
                        )


            with _sub_validate:
                # ── 🧪 大盤濾網實證(只在多頭/空頭進場,淨期望值會更好嗎?) ──
                st.markdown("**🧪 大盤濾網實證(只在特定大盤狀態進場,績效會更好嗎?)**")
                _mf = _load_market_filter_cached(hold_days=5)
                if _mf.get("error"):
                    st.info(f"📊 {_mf['error']}")
                else:
                    _mf_rows = []
                    for _sc in _mf["scenarios"]:
                        _st = _sc["stat"]
                        if _st:
                            _mf_rows.append({
                                "進場條件": _sc["name"],
                                "樣本": _st["n"],
                                "勝率": f"{_st['win_rate']*100:.0f}%",
                                "平均報酬": f"{_st['avg']:+.2f}%",
                                "淨期望值": f"{_st['net_exp']:+.2f}%",
                            })
                        else:
                            _mf_rows.append({"進場條件": _sc["name"], "樣本": 0,
                                             "勝率": "—", "平均報酬": "—", "淨期望值": "—"})
                    st.dataframe(pd.DataFrame(_mf_rows), use_container_width=True, hide_index=True)

                    # 自動解讀:先處理「單一 regime、無對照」的退化情況,再比較淨期望值差
                    _base = _mf["scenarios"][0]["stat"]
                    _bull = _mf["scenarios"][1]["stat"]
                    _bear = _mf["scenarios"][2]["stat"]
                    _bull_n = _bull["n"] if _bull else 0
                    _bear_n = _bear["n"] if _bear else 0
                    if _base and (_bear_n == 0 or _bull_n == 0):
                        # 整段期間只有單一大盤狀態 → 沒有對照組,無法評估濾網
                        _which = "多頭(站上季線)" if _bear_n == 0 else "空頭(跌破季線)"
                        _other = "空頭" if _bear_n == 0 else "多頭"
                        st.caption(
                            f"💡 這段期間大盤**幾乎全程處於{_which}**,{_other}樣本為 0——"
                            f"所以「只在多頭」那列才會跟「全部進場」一模一樣(同一批股票)。"
                            f"**目前沒有對照組,無法評估大盤濾網是否有效**;"
                            f"要等大盤經歷一次狀態切換(跌破/站回季線)、雙邊都有樣本後才看得出來。"
                        )
                    elif _base and _bull and _bull_n >= 5:
                        _gain = _bull["net_exp"] - _base["net_exp"]
                        if _gain > 0.3:
                            st.caption(
                                f"💡 **只在多頭進場淨期望值高出基準 {_gain:+.2f}%**"
                                f"(多頭 {_bull['net_exp']:+.2f}% vs 全部 {_base['net_exp']:+.2f}%)——"
                                f"大盤濾網看起來有幫助,空頭時可考慮減碼或暫停。"
                            )
                        elif _gain < -0.3:
                            st.caption(
                                f"💡 多頭濾網反而較差({_gain:+.2f}%),代表這套選股在空頭也能打——"
                                f"不必特別用大盤狀態過濾。"
                            )
                        else:
                            st.caption(
                                f"💡 多頭/全部進場差異不大({_gain:+.2f}%),大盤濾網目前看不出明顯效果。"
                            )

                    # 溫度濾網(若情緒歷史足夠)
                    _tb = _mf.get("temp_block")
                    if _tb and _tb.get("warm") and _tb.get("cool"):
                        st.caption(
                            f"🌡️ 溫度濾網(限有溫度紀錄的 {_tb['n_temp']} 筆):"
                            f"溫度≥50 淨期望 {_tb['warm']['net_exp']:+.2f}%(n={_tb['warm']['n']}) "
                            f"vs 溫度<50 {_tb['cool']['net_exp']:+.2f}%(n={_tb['cool']['n']})。"
                        )

                    if (_mf.get("n_total") or 0) < 50:
                        st.caption(
                            f"📊 樣本 {_mf.get('n_total', 0)} 筆、分組後更少——需累積跨多空資料後才可信。"
                        )

                # 分數區間表
                by_score = perf.get("by_score", {})
                if by_score:
                    st.divider()
                    st.markdown("**各分數區間 5 日勝率**")
                    st.caption(
                        "🔥 9 分以上 = 頂級(滿分 10 制:大戶↑/散戶↓各 2 分、券與技術面三合一停用;9 分必含雙籌碼共振)"
                        "｜ ✅ 8 分 = 合格(現行門檻,2026-06-13 起)｜ ⚠️ 7 分以下 = 舊門檻/降標紀錄。"
                        "　註:歷史紀錄跨越多次改制(舊 10 制→11 制→2026-06-19 起現行 10 制[技術面三合一停用]),跨期比較請留意。"
                        "　每次入選算一筆(同檔不同天分開計),故筆數比明細表的檔數多是正常的。"
                    )
                    # 分數 → 標籤對映 (現行滿分 10 制、門檻 8,2026-06-19 技術面三合一停用後;
                    #                  更早為 11 制或舊 10 制的紀錄跨期僅供參考)
                    def _score_label(score: int) -> str:
                        if score >= 9:
                            return f"🔥 頂級({score} 分)"
                        elif score == 8:
                            return f"✅ 合格({score} 分)"
                        else:
                            return f"⚠️ {score} 分(低於現行門檻)"

                    rows = []
                    for score in sorted(by_score.keys(), reverse=True):
                        s = by_score[score]
                        _pf = s.get("profit_factor_5d")
                        if _pf is None:
                            _pf_str = "—"
                        elif _pf == float("inf"):
                            _pf_str = "∞"
                        else:
                            _pf_str = f"{_pf:.2f}"
                        # 該分數有入選(n_picks>0)但無任何已到期 5 日報酬 → 入選日還沒滿
                        # 5 個交易日,報酬未到期。顯示「進行中」而非冷冰冰的 0,避免誤會沒記到。
                        # (常見於剛改制後新出現的等級,如 9 分頂級才剛開始選到)
                        _pending = "win_rate_5d" not in s and s.get("n_picks", 0) > 0
                        rows.append({
                            "分級": _score_label(score),
                            # 整欄統一字串:未到期放提示字串、已到期放 str(n_5d)。
                            # 不可混 str 與 int,否則 st.dataframe 的 pyarrow 序列化會 ArrowTypeError。
                            "樣本數": f"⏳ {s['n_picks']} 筆未到期" if _pending else str(s.get("n_5d", 0)),
                            "勝率":   f"{s.get('win_rate_5d', 0)*100:.0f}%" if "win_rate_5d" in s else ("待滿 5 日" if _pending else "—"),
                            "平均報酬": f"{s.get('avg_return_5d', 0):+.2f}%" if "avg_return_5d" in s else "—",
                            "敗局均虧": f"{s.get('avg_loss_5d', 0):+.2f}%" if "avg_loss_5d" in s else "—",
                            "損益比": _pf_str,
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # ── 🔬 訊號歸因分析(10 個計分細項各自帶來多少報酬?) ──
                st.divider()
                st.markdown("**🔬 訊號歸因分析(哪個計分條件真的有效?)**")
                _SIG_DISPLAY = {
                    "投信": "投信買超", "外資": "外資買超", "雙買": "投信+外資雙買",
                    "券": "券相關(資減券增/軋空)", "大戶": "400張大戶上升", "散戶": "散戶下降",
                    "技術": "技術面三合一", "KD": "KD低檔金叉", "營收": "月營收達標",
                    "RS": "RS優於大盤",
                }
                _attr = _load_signal_attribution_cached(hold_days=5)
                if _attr.get("error"):
                    st.info(f"📊 {_attr['error']}")
                elif _attr.get("n_eval", 0) == 0:
                    st.info(
                        f"⏳ **訊號歸因累積中**:目前 **0 筆**含訊號細項的樣本。\n\n"
                        f"訊號細項(`sig`)是近期才開始記錄的——舊歷史沒存,無法回溯。"
                        f"從現在起每日推播會逐筆記錄 10 個計分條件,"
                        f"**累積到約 30+ 筆(滿 5 交易日)後這裡會自動跑出分析**:"
                        f"屆時可看出法人/大戶/RS/營收等條件,哪個真的帶來報酬、哪個其實沒差。"
                    )
                else:
                    st.caption(
                        "每個計分條件「有觸發 vs 沒觸發」時,後續 5 日報酬的差異(edge)。"
                        "edge 為正 = 該訊號確實加值;接近 0 或負 = 可能沒幫助甚至拖累,值得檢討權重。"
                    )
                    _ar_rows = []
                    for _s in sorted(_attr["per_signal"],
                                     key=lambda x: (x["edge"] is not None, x["edge"] or 0),
                                     reverse=True):
                        _on, _off, _edge = _s["on"], _s["off"], _s["edge"]
                        _ar_rows.append({
                            "計分條件": _SIG_DISPLAY.get(_s["key"], _s["key"]),
                            "觸發樣本": _on["n"] if _on else 0,
                            "觸發平均報酬": f"{_on['avg']:+.2f}%" if _on else "—",
                            "未觸發平均報酬": f"{_off['avg']:+.2f}%" if _off else "—",
                            "edge(差)": f"{_edge:+.2f}%" if _edge is not None else "—",
                        })
                    st.dataframe(pd.DataFrame(_ar_rows), use_container_width=True, hide_index=True)
                    if _attr["n_eval"] < 30:
                        st.caption(
                            f"📊 樣本 {_attr['n_eval']} 筆,且訊號彼此相關(多數 pick 同時觸發多個)——"
                            f"edge 僅供方向參考,別據此改權重。"
                        )

                # ── 🚪 出場規則回測(何時賣最賺?) ──
                st.divider()
                st.markdown("**🚪 出場規則回測(固定持有 vs 停損/停利/移動停損,哪種最賺?)**")
                _ex = _load_exit_rules_cached(max_hold=10)
                if _ex.get("error"):
                    st.info(f"📊 {_ex['error']}")
                else:
                    st.caption(
                        f"對每筆已滿 {_ex['max_hold']} 個交易日的 pick(共 {_ex['n']} 筆),"
                        f"模擬不同出場方式的實現報酬。"
                        f"「日均報酬」= 淨期望值 ÷ 平均持有天數,比效率用(早出場可多做幾趟)。"
                        f"樣本基準與上方「持有天數比較」一致(都只算活滿天數的 pick),數字可互相對照。"
                    )
                    _best_net = max((s["net_exp"] for s in _ex["strategies"]), default=None)
                    _best_daily = max((s["daily"] for s in _ex["strategies"]), default=None)
                    _ex_rows = []
                    for _s in _ex["strategies"]:
                        _flag_net   = " 🏆" if _s["net_exp"] == _best_net else ""
                        _flag_daily = " ⚡" if _s["daily"] == _best_daily else ""
                        _ex_rows.append({
                            "出場策略": _s["name"],
                            "勝率": f"{_s['win_rate']*100:.0f}%",
                            "平均報酬": f"{_s['avg']:+.2f}%",
                            "淨期望值": f"{_s['net_exp']:+.2f}%{_flag_net}",
                            "平均持有": f"{_s['avg_days']:.1f} 日",
                            "最差單筆": f"{_s['worst']:+.2f}%",
                            "日均報酬": f"{_s['daily']:+.3f}%{_flag_daily}",
                        })
                    st.dataframe(pd.DataFrame(_ex_rows), use_container_width=True, hide_index=True)
                    st.caption("🏆 = 淨期望值最高(單筆賺最多)　⚡ = 日均報酬最高(效率最高)")
                    if _ex["n"] < 30:
                        st.caption(
                            f"📊 樣本 {_ex['n']} 筆——且停損的價值要到空頭才會凸顯,累積跨多空後再採用。"
                        )


            with _sub_detail:
                # ── 📌 個股層級績效(系統選的『哪些股票』真的賺/賠?) ──
                st.markdown("**📌 個股層級績效(系統選的『哪些股票』真的賺/賠?)**")
                _psc1, _psc2 = st.columns([1, 1])
                _ps_hold = _psc1.selectbox("持有天數", [3, 5, 10, 20], index=1, key="_perstock_hold")
                _ps_min = _psc2.number_input("只看入選 ≥ N 次", min_value=1, max_value=50, value=1,
                                             step=1, key="_perstock_min",
                                             help="過濾只入選 1~2 次的雜訊;次數多又穩賺才是系統的常勝股")
                _ps = _load_per_stock_perf_cached(hold_days=_ps_hold)
                _ps = [r for r in _ps if r["n"] >= _ps_min]
                if not _ps:
                    st.info("📊 尚無足夠樣本(需有 pick 已滿持有天數,或放寬「入選 ≥ N 次」)。")
                else:
                    _ps_rows = [{
                        "代號":      r["sid"],
                        "名稱":      ui_name_map.get(r["sid"], ""),
                        "入選次數":  r["n"],
                        "勝率%":     round(r["win_rate"] * 100, 0),
                        f"平均{_ps_hold}日報酬%": round(r["avg"], 2),
                        "最佳%":     round(r["best"], 2),
                        "最差%":     round(r["worst"], 2),
                        "平均分":    round(r["avg_score"], 1) if r["avg_score"] is not None else None,
                        "最近入選":  r["last_date"],
                    } for r in _ps]
                    st.dataframe(pd.DataFrame(_ps_rows), use_container_width=True, hide_index=True)
                    st.caption(
                        f"依「平均 {_ps_hold} 日報酬」高→低排序,共 {len(_ps)} 檔(已滿天數)。"
                        "點欄位標題可改排序;**入選次數多又穩賺**=系統的常勝股,**常賠**的可加進選股表的「排除過熱」或自行避開。"
                        "進場口徑同其他回測(訊號日隔日開盤 + 0.1% 滑價);樣本少時僅看方向。"
                    )

                # ── 進行中觀察樣本(剛入選還沒滿 5 個交易日,目前浮動報酬) ──
                samples = perf.get("samples", [])

                def _dedup_keep_latest(rows: list) -> list:
                    """同 sid 只留 date 最大那筆(分流去重 = 模式 B)。
                    rows 內每筆需含 'sid' 與 'date' 欄。
                    """
                    latest = {}
                    for r in rows:
                        sid_k = str(r.get("sid", ""))
                        r_date = r.get("date", "")     # 防呆:缺欄位給空字串(排序時會被後來的取代)
                        if not sid_k:
                            continue
                        cur = latest.get(sid_k)
                        if cur is None or r_date > cur.get("date", ""):
                            latest[sid_k] = r
                    return list(latest.values())

                pending_raw = [s for s in samples if s.get("return_5d") is None]
                pending = _dedup_keep_latest(pending_raw)   # 同檔股票只留最新一筆進行中
                if pending:
                    # 從 daily parquet 抓最新 close 算「目前浮動報酬」
                    @st.cache_data(ttl=300, show_spinner=False)
                    def _load_latest_close_for_pending():
                        """回傳 {sid: (latest_date_str, latest_close)} 給進行中樣本算浮動報酬。"""
                        try:
                            files = sorted(CACHE_DIR.glob('daily_*.parquet'))
                            if not files:
                                return {}, None
                            df = pd.read_parquet(files[-1], columns=['stock_id', 'date', 'close'])
                            df['date'] = pd.to_datetime(df['date'])
                            df['stock_id'] = df['stock_id'].astype(str)
                            latest_date = df['date'].max()
                            latest_rows = (
                                df[df['date'] == latest_date]
                                  .drop_duplicates(subset='stock_id', keep='last')
                                  .set_index('stock_id')['close']
                            )
                            return latest_rows.to_dict(), latest_date.strftime('%Y-%m-%d')
                        except Exception as e:
                            print(f"進行中樣本 close 抓取失敗: {e}")
                            return {}, None

                    latest_close_map, latest_date_str = _load_latest_close_for_pending()

                    st.divider()
                    _today_naive = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None).normalize()
                    _dedup_note = ""
                    if len(pending_raw) > len(pending):
                        _dedup_note = f"(原 {len(pending_raw)} 筆,同檔已去重)"
                    st.markdown(f"**⏳ 進行中觀察樣本(共 {len(pending)} 檔{_dedup_note})**")
                    st.caption(
                        f"剛入選還沒成熟的樣本,**同檔股票只保留最新一次**,目前浮動報酬以最新收盤價計算"
                        f"(快取日期:{latest_date_str or 'N/A'})。**不列入勝率統計**,僅供觀察。"
                    )

                    pending_rows = []
                    pending_floats = []  # 蒐集浮動報酬給統計用
                    for s in sorted(pending, key=lambda x: x["date"], reverse=True):
                        sid_str = str(s["sid"])
                        entry_date_str = s["date"]
                        score = s.get("score")
                        entry_close = s.get("entry_close")  # v2 schema 有,v1 沒

                        # v1 entry 可能沒 entry_close → 從 daily parquet 找入選日的 close
                        if entry_close is None:
                            try:
                                ed = pd.Timestamp(entry_date_str)
                                files = sorted(CACHE_DIR.glob('daily_*.parquet'))
                                if files:
                                    # 用 ad-hoc 查詢,效能不重要因為 pending 通常 < 30 筆
                                    _d = pd.read_parquet(files[-1], filters=[('stock_id', '==', sid_str)])
                                    if not _d.empty:
                                        _d['date'] = pd.to_datetime(_d['date'])
                                        _d_match = _d[_d['date'] == ed]
                                        if not _d_match.empty:
                                            entry_close = float(_d_match['close'].iloc[0])
                            except Exception:
                                pass

                        latest_close = latest_close_map.get(sid_str)

                        if entry_close is not None and latest_close is not None and entry_close > 0:
                            float_ret = (latest_close - entry_close) / entry_close * 100
                            pending_floats.append(float_ret)
                            # 台股紅漲綠跌
                            if float_ret > 0.05:
                                ret_str = f"🔴 +{float_ret:.2f}%"
                            elif float_ret < -0.05:
                                ret_str = f"🟢 {float_ret:.2f}%"
                            else:
                                ret_str = f"⚪ {float_ret:+.2f}%"
                        else:
                            ret_str = "—"
                            latest_close = None

                        # 算「還剩幾個交易日成熟」(用實際自然日近似:5 交易日 ≈ 7 自然日)
                        try:
                            days_since = (_today_naive - pd.Timestamp(entry_date_str)).days
                            days_left = max(0, 7 - days_since)  # 粗估
                            if days_left == 0:
                                mature_status = "即將成熟"
                            else:
                                mature_status = f"剩 ~{days_left} 日"
                        except Exception:
                            mature_status = "—"

                        pending_rows.append({
                            "入選日":    entry_date_str,
                            "代號":      sid_str,
                            "名稱":      ui_name_map.get(sid_str, ""),
                            "分數":      f"{score} 分" if score is not None else "—",
                            "入選價":    f"{entry_close:.2f}" if entry_close else "—",
                            "目前價":    f"{latest_close:.2f}" if latest_close else "—",
                            "浮動報酬": ret_str,
                            "成熟":      mature_status,
                        })

                    st.dataframe(pd.DataFrame(pending_rows), use_container_width=True, hide_index=True)

                    # 進行中樣本的趨勢摘要
                    if pending_floats:
                        wins = sum(1 for r in pending_floats if r > 0)
                        avg_float = sum(pending_floats) / len(pending_floats)
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric("目前浮動勝率", f"{wins / len(pending_floats) * 100:.0f}%",
                                   f"{wins} / {len(pending_floats)}",
                                   help="這些股票還沒滿 5 個交易日,用最新收盤價算的「未實現」勝率,還沒定案。")
                        mc2.metric("平均浮動報酬", f"{avg_float:+.2f}%",
                                   delta_color="inverse",  # 台股紅漲綠跌 → metric 預設綠正紅負,要 inverse
                                   help=(
                                       "進行中(還沒滿 5 交易日)樣本的平均帳面報酬,尚未定案。\n\n"
                                       "右側狀態依此值判定:\n"
                                       "• > +3% → 📈 整體偏紅(訊號近期強勢)\n"
                                       "• −3% ~ +3% → ⏳ 接近平盤(漲跌不明顯,等滿 5 天再看)\n"
                                       "• < −3% → 📉 整體偏綠(留意訊號失效)"
                                   ))
                        # 警示
                        if avg_float > 3:
                            mc3.success("📈 進行中樣本目前**整體偏紅**,系統訊號近期表現強勢")
                        elif avg_float < -3:
                            mc3.warning("📉 進行中樣本目前**整體偏綠**,留意近期訊號失效")
                        else:
                            mc3.info("⏳ 進行中樣本目前**接近平盤**,等待成熟")

                # ── 樣本明細表(顯示哪幾檔股票、實際報酬多少) ──
                # 同檔股票只保留「日期最新的已完成樣本」 = 分流去重模式 B
                valid_samples_raw = [s for s in samples if s.get("return_5d") is not None]
                valid_samples = _dedup_keep_latest(valid_samples_raw)
                if valid_samples:
                    st.divider()
                    _comp_note = ""
                    if len(valid_samples_raw) > len(valid_samples):
                        _comp_note = f",原 {len(valid_samples_raw)} 筆同檔已去重"
                    st.markdown(f"**樣本明細(已完成,共 {len(valid_samples)} 檔{_comp_note},顯示前 50)**")
                    st.caption(
                        "同檔股票只保留**最新一次的已完成樣本**;勾選欄頭可排序;台股紅漲綠跌。"
                        "看到報酬巨大的個股可點進去研究是什麼條件讓它大漲/大跌。"
                    )

                    # 名稱對映用既有 ui_name_map(已從 daily parquet 載入)
                    rows_detail = []
                    for s in sorted(valid_samples, key=lambda x: x["date"], reverse=True)[:50]:
                        sid_str = str(s["sid"])
                        ret = s["return_5d"]
                        # 台股紅漲綠跌:emoji + 數值
                        if ret > 0.05:
                            ret_str = f"🔴 +{ret:.2f}%"
                        elif ret < -0.05:
                            ret_str = f"🟢 {ret:.2f}%"
                        else:
                            ret_str = f"⚪ {ret:+.2f}%"

                        rows_detail.append({
                            "入選日":    s["date"],
                            "代號":      sid_str,
                            "名稱":      ui_name_map.get(sid_str, ""),
                            "分數":      f"{s['score']} 分" if s.get("score") is not None else "—",
                            "5 日後報酬": ret_str,
                            "10 日後":   f"{s['return_10d']:+.2f}%" if s.get("return_10d") is not None else "—",
                            "20 日後":   f"{s['return_20d']:+.2f}%" if s.get("return_20d") is not None else "—",
                        })
                    st.dataframe(
                        pd.DataFrame(rows_detail),
                        use_container_width=True, hide_index=True,
                    )

                    # CSV 匯出:未去重的完整原始紀錄(含所有重複入選),方便 Excel 自行分析
                    full_df = pd.DataFrame(valid_samples_raw)
                    full_df["名稱"] = full_df["sid"].astype(str).map(ui_name_map).fillna("")
                    st.download_button(
                        f"📥 下載完整原始樣本 CSV(未去重,{len(valid_samples_raw)} 筆)",
                        full_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"performance_samples_raw_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=False,
                        help="UI 顯示已套用「同檔只留最新」去重,但這個 CSV 是原始完整紀錄(含所有重複入選),供 Excel 進階分析使用"
                    )
    else:
        st.info(f"目前累積 {_hist_days} 天歷史,需 ≥5 天才能算績效。每天執行 Telegram 推播自動累積。")


# ── 訊號回測:對 daily/法人 parquet 掃描全部技術訊號歷史報酬 ─────────────
# 設計:把「最重的全部訊號矩陣建構」拆出來獨立 cache,讓改變 hold_days /
# date_filter / combine_mode / stock_filter 等「輕」參數不必重算 5-7 秒。
@st.cache_data(ttl=1800, show_spinner="第一次建構訊號矩陣 (全市場掃描 180 天,~5 秒)…")
def _build_signal_matrices_cached(cache_date_str: str):
    """全部訊號矩陣 cache。

    cache key 用最新資料日期(每日 daily parquet 更新會自然觸發重算)。
    TTL 30 分鐘為保險,實務上 key 變動就會 invalidate。
    """
    return build_signal_matrices(CACHE_DIR)


@st.cache_data(ttl=1800, show_spinner=False)
def _load_dip_radar(cache_key: str, lookback: int = 6):
    """🎯 籌碼抄底雷達:回傳 (命中代號 set, 依據日期字串)。

    重用 _build_signal_matrices_cached(同一份 cache key,跟訊號回測分頁共用,已建過就秒切),
    取每個訊號矩陣「最近一個資料到齊的交易日」做 AND 交集。

    為何不直接用 iloc[-1]:融資融券(margin_squeeze)資料常比股價**慢一個交易日** → 最後一列會
    全 False(資料未到),直接取最後一列會讓雷達永遠空。故由最新往回找第一個「每個 combo 訊號該列
    都至少 1 檔觸發」的日期(= margin 資料到齊的最近日),在那天交集。lookback 限制最多回看幾天,
    避免資料真的長期缺漏時抓到太舊日期。缺資料/結構不符 → 回 (set(), '')。"""
    try:
        pre = _build_signal_matrices_cached(cache_key)
        if not pre or not isinstance(pre.get("sig_matrices"), dict):
            return set(), ""
        sig = pre["sig_matrices"]
        if not all(k in sig for k in DIP_COMBO_SIGNALS):
            return set(), ""
        mats = [sig[k] for k in DIP_COMBO_SIGNALS]
        if any(m is None or m.empty for m in mats):
            return set(), ""
        idx = mats[0].index
        for off in range(1, min(lookback, len(idx)) + 1):
            rows = [m.iloc[-off] for m in mats]
            if all(r.astype(bool).any() for r in rows):     # 該日每個訊號都有資料(≥1 觸發)
                hit = None
                for r in rows:
                    s = set(r[r.astype(bool)].index.astype(str))
                    hit = s if hit is None else (hit & s)
                return (hit or set()), str(idx[-off].date())
        return set(), ""
    except Exception:
        return set(), ""


@st.cache_data(ttl=3600, show_spinner=False)
def _load_industry_name_map() -> dict:
    """從 info parquet 建 {stock_id: 產業代號}(供雷達清單沿用既有產業欄口徑;原 df '產業' 也是代號)。"""
    try:
        files = sorted(CACHE_DIR.glob("info_*.parquet"))
        if not files:
            return {}
        d = pd.read_parquet(files[-1], columns=["stock_id", "industry_category"])
        d = d.drop_duplicates("stock_id", keep="last")
        return {str(r.stock_id): str(r.industry_category) for r in d.itertuples() if pd.notna(r.industry_category)}
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner="計算交易報酬中…")
def _run_backtest_cached(signals_tuple: tuple, hold_days: int, date_filter: str,
                         combine_mode: str, dedup: bool, stock_filter: str = ""):
    """signals_tuple: 用 tuple 才能被 cache_data hash。stock_filter='' 視同全市場。"""
    if date_filter == "all":
        date_range = None
    else:
        ndays = 90 if date_filter == "90d" else 30
        end = pd.Timestamp.now(tz="Asia/Taipei").tz_localize(None).normalize()
        start = end - pd.Timedelta(days=ndays)
        date_range = (start, end)
    # 取得已 cache 的訊號矩陣(第一次需 5-7s,之後秒切)
    # cache key 帶結構/訊號版本:每次「改 build_signal_matrices 結構」或「新增/移除訊號」都要 bump,
    # 否則殘留舊 cache(只 Rerun 未 full reboot 時)會找不到新訊號 → 靜默回 0 觸發。
    # v2-nextopen:改隔日開盤結構;v3-tierR:新增反轉訊號(washout/kd_divergence/chip_accumulation)。
    _cache_key = _cache_date.strftime('%Y-%m-%d') if _cache_date is not None else "no_data"
    _precomputed = _build_signal_matrices_cached(f"{_cache_key}|v3-tierR")
    return run_backtest(CACHE_DIR, signal=list(signals_tuple), hold_days=hold_days,
                        date_range=date_range, combine_mode=combine_mode,
                        dedup_within_hold=dedup,
                        stock_filter=stock_filter or None,
                        precomputed=_precomputed)


with _tab_bt:
    st.caption(
        "用**過去的歷史資料**驗證「某個買進條件,照著做到底會不會賺」。"
        "下面可勾一個或多個條件做組合測試。"
        "  \n⚙️ **進場假設:訊號日隔日開盤 + 0.1% 滑價**(去除前視偏誤);出場:持有 N 日後收盤。"
    )

    # ── 📖 使用說明 / FAQ(摺疊,整頁全寬;放在選擇訊號上方)──
    with st.expander("📖 使用說明 / FAQ", expanded=False):
        st.markdown("""
### 🔰 白話版(股市小白看這個就好)

**這頁在幹嘛?** 用**過去的歷史**幫你檢查:「**某個買進條件,以前照著做會不會賺?**」

**怎麼用(3 步):**
1. 下面**勾一個**想測的條件(例:量價齊揚突破)。
2. 設定**抱幾天**(預設 10 天)。
3. 看下方跑出來的 **勝率 / 平均報酬**。

**結果怎麼看(最重要):**
- **勝率** = 賺錢的比例;但 ⚠️ **勝率高 ≠ 賺錢!**
- **平均報酬** = 平均每次賺/賠幾 %。
- 👉 **兩個一起看**:勝率高、但平均報酬是負的 = 賺小賠大,還是不能用。
- 樣本太少(< 30 筆)會跳警告,別太當真。

**勾兩個以上時:**
- **AND** = 兩條件**同一天都符合**才算(嚴、機會少但更準)。
- **OR** = 任一個符合就算(寬、機會多)。只勾 1 個時兩者沒差。

**新手先試:** 只勾「量價齊揚突破」,看最經典的「帶量突破」過去賺不賺。

> 想深入了解全部訊號、進階組合、各指標好壞標準?往下看 👇(進階參考)

---

### 🎯 這頁在做什麼?
對歷史資料**掃描每個交易日**,標出符合訊號的點當作「進場」,算進場後 N 個交易日的報酬。
最終回答:**「我選的這些訊號條件,真的能賺錢嗎?」**

---

### 🔍 全部訊號(依「台股實戰強度」分檔;Tier S/A/B/C 偏追強勢,Tier R 偏抄底)

#### 🌟 Tier S — 台股神器(最強三檔)
| 訊號 | 觸發條件 | 為什麼台股特別強 |
|---|---|---|
| **籌碼共振(散戶↓)** | 本週散戶比例 < 上週 | TDCC 公開資料 + 散戶占比 >50%,主力吃貨訊號最銳利 |
| **外資連 5 日買超** | 近 5 日外資總淨額 > 0 且 ≥ 2 日買 | 外資占成交 30~40%,行為延續性高 |
| **月營收雙紅突破** | YoY > 10% 且 MoM > 0 且 當日突破新高 | 台灣**每月 10 日營收公告**是全球罕見的公開領先指標 |

#### ✅ Tier A — 有效(次強四檔)
| 訊號 | 觸發條件 | 評價 |
|---|---|---|
| **資減券增(軋空)** | 近 5 日 融資↓3% 且 融券↑5% | 台股獨有的融資融券公開資料,軋空行情常見 |
| **三大法人同步買超** | 同日 外資+投信+自營 全買 | 最強共識訊號,但樣本稀少 |
| **投信連 5 日買超** | 近 5 日投信總淨額 > 0 且 ≥ 2 日買 | 投信選股能力強,月底季底有作帳行情 |
| **量價齊揚突破** | close ≥ 60 日新高 × 0.995 且 量 ≥ MA20 × 1.5 | 經典動能,台股小型股動能更強 |

#### 🟡 Tier B — 效果一般(三檔)
| 訊號 | 觸發條件 | 評價 |
|---|---|---|
| **20 日動能 Top 10%** | 近 20 日報酬排名全市場前 10% | 學術實證的動能因子,但台股小型股易反轉 |
| **品質突破** | 突破 + 突破前 10 日有 ≥ 6 日量縮 | 邏輯合理但條件嚴,樣本少 |
| **MA 黃金交叉** | MA20 上穿 MA60(中長線轉折) | 用的人多 → 容易被搶先反映 |

#### ❌ Tier C — 效果不顯著(待驗證)
| 訊號 | 觸發條件 | 評價 |
|---|---|---|
| **KD 低檔金叉** | K 上穿 D 且 K < 設定門檻(低檔) | 散戶常用 → alpha 被吃光;**保留供使用者親自驗證** |

#### 🔄 Tier R — 反轉/抄底(實驗,與上面「追強勢」方向相反)
> 在價弱/跌深時找轉折,**勝率天生較低、訊號偏多**,單獨用篩選力不足 → 務必配「籌碼」AND 並嚴設停損。

| 訊號 | 觸發條件 | 評價 |
|---|---|---|
| **大戶逆勢增持(低檔吃貨)** | 大戶持股率(400 張以上)週對週↑ 且 今日 close < MA20 | **籌碼領先訊號,底部最可靠**;反轉組合裡 edge 最明顯 |
| **KD 低檔背離** | 價創 N 日新低、但 KD 沒破底(K < 35) | 動能先轉的「**預警**」(此時還在跌);訊號多,需配籌碼 |
| **跌深量縮後帶量翻揚** | close < MA60 + 前 5 日量縮 + 今日帶量紅K | 賣壓耗盡後第一根「**發動K**」(此時開始漲);少而明確 |

> 💡 **已移除**:`MA20 拉回守穩(葛蘭碧 2)` — 學術實證無 alpha;`RS 優於大盤` — 跟其他訊號高度相關。函式仍保留在程式碼中,未來想恢復可直接 register。

---

### ⚙️ 參數說明

**🔀 合併模式(AND vs OR)**
- **AND 交集** = 多個訊號**同日全部**觸發才算。訊號變少但更可信。範例:「外資 AND 突破」= 兩個都成立才進場
- **OR 聯集** = 任一訊號觸發就算。訊號變多、容錯高。範例:「KD 低檔金叉 OR 量價齊揚突破」= 兩個進場時機點都接受
- 💡 **只選 1 個訊號時兩者效果一樣**

**📅 持有天數**
- **5 日**:短線,適合動能型訊號(突破、爆量)
- **10 日**:中短線,通用默認值
- **20 日**:中線,適合籌碼型訊號(法人連買)

**📆 樣本期間**
- 「全部 ~180 日」最完整;近 30/90 日聚焦近期市況

**🔒 持倉鎖定期內不重複進場(預設打開)**
- 模擬真實「沒平倉前不再買」── 同股 N 日內第二次觸發會被忽略
- 關掉則是「逐日掃描」原始統計,**強勢股會被多次計入** → 灌水

**🎯 個股回測**
- 留空 = 全市場掃描
- 填代號(例如 `2454`)= 只跑這一檔的訊號績效 → 答「2454 過去這套策略賺不賺?」

---

### 📊 怎麼讀指標

| 指標 | 解讀 | 好壞標準 |
|---|---|---|
| **樣本數** | 觸發次數 | < 30 不顯著(會跳警示) |
| **勝率** | 賺錢交易 / 總交易 | > 55% 可參考、> 65% 不錯 |
| **平均報酬** | 每筆平均賺幾 % | 加上勝率一起看才有意義 |
| **MDD 最大回檔** | 累積資金曲線最大跌幅 | -10% 內低風險、-20%+ 高風險 |
| **Sharpe 夏普值** | 風險調整後報酬(年化) | > 1 良好、> 2 優秀、< 0 不及格 |

**⚠ 注意:勝率高 ≠ 賺錢**。例如勝率 80% 但平均 +1%、最差 -30% → 期望值是負的。看**平均報酬 + MDD + Sharpe 三項一起判斷**才正確。

---

### 💡 常見組合範例(直接複製貼上用)

> 📂 **Tier S/A/B/C = 追強勢**(多頭順勢買強股,由強到弱排);**Tier R = 抄底反轉**(跌深找轉折,方向相反,非「更弱」)。⭐ = 我會優先測的組合。多頭環境用前者、回檔後用 Tier R,兩者互補。

#### 🌟 Tier S 級組合(優先測試)

**1. ⭐ 只勾「月營收雙紅突破」**
- 在問:「公司營收年增 + 月增,同一天股價又突破新高,跟著進場勝率多高?」
- 適合:基本面派,要「真材實料 + 技術面同步啟動」的股票

**2. ⭐ 勾「月營收雙紅突破」+「外資連 5 日買超」→ 選 AND**
- 在問:「業績好的股票 + 外資也在掃貨,這種雙重確認的勝率多高?」
- 適合:**保守穩健派的最佳組合**,基本面+籌碼面雙重把關

**3. 勾「籌碼共振(散戶↓)」+「外資連 5 日買超」→ 選 AND**
- 在問:「散戶在拋售 + 外資在掃貨,主力吃貨最明顯的時刻」
- 適合:相信「籌碼面是領先指標」的人

**4. 勾「籌碼共振」+「量價齊揚突破」→ 選 AND**
- 在問:「主力先默默吃貨,然後股價才突破新高,這種能不能賺?」
- 適合:相信「籌碼面比技術面早一步」的人

#### ✅ Tier A 級組合

**5. ⭐ 只勾「資減券增」(台股經典軋空訊號)**
- 在問:「散戶融資認賠 + 空頭加碼放空的股票,被軋上去的機率多高?」
- 適合:想抓「籌碼結構翻轉、空頭潰敗」的反向機會

**6. 勾「資減券增」+「20 日動能 Top 10%」→ 選 AND**
- 在問:「漲勢強的股票同時又有軋空條件,進場後續報酬如何?」
- 適合:動能 + 軋空壓力 雙重確認的進階組合

**7. 只勾「三大法人同步買超」**
- 在問:「外資+投信+自營三家『同一天』都買超的股票,後續表現如何?」
- 適合:只想看最強共識訊號(樣本會少)

**8. 勾「外資連 5 日買超」+「投信連 5 日買超」→ 選 AND**
- 在問:「外資跟投信兩家法人都連續買的股票,跟著進場能不能賺?」
- 適合:跟單派(讓大資金當你的領頭羊)

**9. 只勾「量價齊揚突破」**
- 在問:「只看股價突破新高 + 帶量,經典技術派到底能不能賺?」
- 適合:想驗證最經典技術派策略

#### 🟡 Tier B 級組合(實驗用)

**10. 只勾「20 日動能 Top 10%」**
- 在問:「近 1 個月漲最多的前 10% 強勢股,未來繼續漲嗎?」
- 背景:Jegadeesh & Titman 1993 經典論文,**全球股市動能效應已被驗證 30 年**

**11. 勾「品質突破」單獨**
- 在問:「整理量縮一陣子後才爆量突破,跟『一路飆然後勉強突破』比起來真的比較會漲嗎?」
- 適合:想驗證「不是所有突破都一樣」

**12. 只勾「MA 黃金交叉」**
- 在問:「MA20 上穿 MA60 的中長線轉折日進場,後續報酬如何?」
- 適合:中長線投資人(訊號稀少,每個都是大轉折)

#### ❌ Tier C 級組合(對照組)

**13. 只勾「KD 低檔金叉」**
- 在問:「散戶最愛的 KD 金叉訊號,實際勝率到底有多少?」
- 適合:**驗證老師教的有沒有效**;預期勝率接近 50/50

**14. 全部訊號都勾起來 → 選 OR(寬鬆模式)**
- 在問:「不管哪個訊號觸發都當作買進機會,整體勝率長怎樣?」
- 適合:當作「基準對照組」,看你篩選後的策略有沒有比這個強

#### 🔄 Tier R 級組合(抄底/反轉,與上面「追強勢」相反)

> ⚠️ 抄底勝率天生較低、容易接刀子。**單一反轉訊號觸發很多、篩選力不足 → 一定要用 AND 組合**;進場再等帶量紅K確認、嚴設停損。
>
> 📉 **看「平均報酬」不是「勝率」**:抄底就是「對的次數少、但對的時候賺大」—— 勝率 45% 但平均 +2% 是好策略;勝率 60% 但平均 −0.5% 反而爛。別看到低勝率就以為訊號壞了。
>
> ⏳ **資料還沒輪到它表演**:Tier R 是為「**跌深後的反彈**」設計的,但目前歷史幾乎全在多頭、沒大回檔 → 反轉訊號結構性吃虧、數字偏弱是**正常**的。多頭就用 Tier S/A 追強勢,Tier R 先養樣本;**等真的跌一波、出現 washout,它才會發揮**。

**15. ⭐ 勾「大戶逆勢增持」+「資減券增」→ 選 AND**(推薦)
- 在問:「主力在低檔默默吃貨、同時散戶認賠+空方放空的股票,反轉機率多高?」
- 適合:**籌碼面雙確認的抄底** —— 底部最該信的是「聰明錢在進貨」。實測這組從幾千筆收斂到約 240 筆,是反轉組合裡 edge 最明顯的。

**16. 勾「大戶逆勢增持」+「KD 低檔背離」→ 選 AND**
- 在問:「主力在吃貨、技術面動能又開始背離轉強的股票,是不是落底訊號?」
- 適合:**籌碼 + 技術雙確認**的轉折

**17. 單勾「跌深量縮後帶量翻揚」**
- 在問:「跌深後賣壓耗盡、突然一根帶量紅K,是不是反轉起點?」
- 適合:純價量派抓「第一根發動K」(訊號少而明確)

> 🚫 **別把「KD 低檔背離」AND「跌深量縮後帶量翻揚」**:這兩個對「今天」的要求**相反**——背離要求今天**還在創新低**(跌),翻揚要求今天是**紅K上漲**;一天不可能又跌又漲,**AND 必定 0 觸發**(不是訊號壞了)。
> 它們是同一齣戲的**前後兩幕**:**背離=預警**(還在跌、動能先轉,拿來「盯哨」)→ **翻揚=發動**(今天開始漲,拿來「進場」)。
> 👉 要組合就**跟「籌碼」AND**(籌碼是週資料、跨天,不挑單日漲跌):背離/翻揚 任一 **+ 大戶逆勢增持 / 資減券增**,才是有篩選力又不互斥的抄底組合。

> 💡 這幾個反轉訊號**目前在全多頭資料裡偏弱**(抄底本來就吃虧於多頭),需累積更多、且**跨過一次大盤回檔**才看得出真價值——這正是它們存在的意義:等空頭來臨時派上用場。

---

### ⏱ 持有天數敏感度怎麼用
打開下面那個摺疊區,**同一組訊號**會跑 5/10/20/40 日對比 → 直接看到「**這策略最佳持有期是幾天?**」
        """)

    sig_options = list(SIGNAL_LABELS.keys())
    sig_choice_multi = st.multiselect(
        "選擇訊號(可複選)", sig_options,
        default=["breakout"],
        format_func=lambda k: SIGNAL_LABELS[k],
        key="bt_signals",
        help="多選時依「合併模式」交集 / 聯集 — 例:選『外資+投信』模式 AND = 兩家同日都買超才算",
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

    bcc1, bcc2 = st.columns([1, 2])
    dedup_choice = bcc1.checkbox(
        "持倉鎖定期內不重複進場", value=True, key="bt_dedup",
        help="勾選後:同股票進場後在持有天數內的第二次訊號會被忽略,模擬真實「沒平倉前不再買」。\n關掉則是「每天逐日掃描」的原始統計,強勢股會被多次計入。"
    )
    stock_filter_input = bcc2.text_input(
        "🎯 個股回測(留空 = 全市場)", value="", key="bt_stock_filter",
        placeholder="例:2330",
        help="輸入股票代號就只跑這一檔的歷史訊號績效,例如想看 2454 過去 KD 金叉進場是否能賺錢"
    ).strip()

    if not sig_choice_multi:
        st.info("👆 請至少勾選一個訊號才能跑回測")
        _bt = {"trades": pd.DataFrame(), "stats": {"n": 0}, "all_signals_stats": {}}
    else:
        _bt = _run_backtest_cached(tuple(sig_choice_multi), hold_choice, period_choice,
                                   combine_mode_choice, dedup_choice, stock_filter_input)
        # 個股回測且查無資料時的友善提示
        if stock_filter_input and _bt['stats'].get('n', 0) == 0:
            st.warning(f"⚠ 個股 {stock_filter_input} 在所選期間/訊號下無觸發,試試擴大期間或換訊號組合。")
    # 下方明細表/CSV 命名用的單一 label
    _suffix = f"_{stock_filter_input}" if stock_filter_input else ""
    sig_choice = ("_".join(sig_choice_multi) + ("_AND" if combine_mode_choice == "and" else "_OR") + _suffix
                  if sig_choice_multi else "none")

    if _bt.get("error"):
        st.error(f"⚠️ {_bt['error']}")
    elif _bt['stats'].get('n', 0) == 0:
        st.info("此訊號在所選期間內無觸發紀錄。試試擴大期間或換訊號。")
    else:
        stats = _bt['stats']

        # ── ⚠️ 樣本數不足警示:< 30 筆統計不顯著,易受極端值影響 ──
        if stats['n'] < 30:
            st.warning(
                f"⚠️ 樣本僅 **{stats['n']} 筆**,統計不顯著(< 30 筆易受極端值影響)。"
                "建議:擴大樣本期間、放寬訊號組合、或關掉持倉鎖定。"
            )

        # ── 6 張指標卡(第一列:基本統計 / 第二列:風險指標)──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("樣本數", f"{stats['n']:,}", "次觸發")
        m2.metric("勝率", f"{stats['win_rate']*100:.0f}%",
                  f"{int(stats['win_rate']*stats['n'])} 賺")
        avg = stats['avg_return']
        m3.metric("平均報酬", f"{avg:+.2f}%",
                  f"中位數 {stats['median_return']:+.2f}%",
                  delta_color="inverse")
        m4.metric("最差單筆", f"{stats['min_return']:+.2f}%",
                  f"最佳 {stats['max_return']:+.2f}%",
                  delta_color="inverse")

        # 第二列:風險指標(MDD / Sharpe / Std)
        r1, r2, r3 = st.columns(3)
        _mdd = stats.get('mdd', 0)
        _sharpe = stats.get('sharpe', 0)
        _std = stats.get('std', 0)
        # MDD 評級
        if _mdd >= -10:
            _mdd_eval = "✅ 低風險"
        elif _mdd >= -20:
            _mdd_eval = "🟡 中等"
        else:
            _mdd_eval = "🔴 高風險"
        # Sharpe 評級(年化):>1 不錯、>2 優秀、>3 罕見;< 0 不及格
        if _sharpe >= 2:
            _sharpe_eval = "🌟 優秀"
        elif _sharpe >= 1:
            _sharpe_eval = "✅ 良好"
        elif _sharpe >= 0:
            _sharpe_eval = "🟡 一般"
        else:
            _sharpe_eval = "🔴 不及格"
        r1.metric("📉 最大回檔(MDD)", f"{_mdd:+.2f}%", _mdd_eval,
                  delta_color="off",
                  help="累積資金曲線從高點到低點的最大跌幅。\n-10% 內算低風險;-20% 中等;-20%+ 高風險。")
        r2.metric("⚖️ 夏普值(年化)", f"{_sharpe:+.2f}", _sharpe_eval,
                  delta_color="off",
                  help="風險調整後報酬 = 平均/標準差 × √(252/持有天數)。\n>1 良好;>2 優秀;<0 不及格。")
        r3.metric("📊 標準差", f"{_std:.2f}%",
                  "波動大" if _std > 8 else "波動低",
                  delta_color="off",
                  help="單筆報酬的標準差。越小代表報酬越穩定。")

        # ── 全訊號對照圖 ──
        st.divider()
        st.markdown(f"**全部 {len(SIGNAL_LABELS)} 個訊號 {hold_choice} 日勝率 / 平均報酬對照**")
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

        # ── ⏱ 持有天數敏感度分析:同訊號組合下,5/10/20/40 日結果並排比較 ──
        # 答得「該策略最佳持有期是幾天?」(避免你硬選 10 天但其實 5 天勝率更高的盲區)
        if sig_choice_multi:
            st.divider()
            with st.expander("⏱ 持有天數敏感度(同訊號下 5/10/20/40 日對比)", expanded=False):
                st.caption(
                    "**這在回答什麼?** 同一組訊號,進場後該抱 5 天還是 40 天最好?"
                    "把相同訊號套到 4 種持有期,看勝率/報酬/風險怎麼隨持有天數變化,找出這套訊號的「最佳持有期」。\n\n"
                    "**怎麼看下面的圖:** 紅實線=勝率(左軸)、灰虛線=平均報酬(右軸)。"
                    "兩線都往上 → 抱越久越好;勝率升但報酬降(或相反)→ 看你要穩還是要大;都往下 → 這訊號適合短打、別久抱。"
                )
                _sens_rows = []
                for _hd in (5, 10, 20, 40):
                    try:
                        _bt_s = _run_backtest_cached(
                            tuple(sig_choice_multi), _hd, period_choice,
                            combine_mode_choice, dedup_choice, stock_filter_input
                        )
                        _s = _bt_s.get('stats', {"n": 0})
                        if _s.get('n', 0) > 0:
                            _sens_rows.append({
                                "持有天數": f"{_hd} 日",
                                "樣本": _s['n'],
                                "勝率(%)":   _s['win_rate'] * 100,
                                "平均報酬(%)": _s['avg_return'],
                                "MDD(%)":  _s.get('mdd', 0),
                                "Sharpe":  _s.get('sharpe', 0),
                            })
                    except Exception as _e:
                        print(f"⚠ 敏感度計算 hold={_hd} 失敗: {_e}")

                if _sens_rows:
                    _sens_df = pd.DataFrame(_sens_rows)
                    # 雙軸折線圖:勝率(左) + 平均報酬(右)
                    fig_sens = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_sens.add_trace(
                        go.Scatter(x=_sens_df['持有天數'], y=_sens_df['勝率(%)'],
                                   mode='lines+markers+text', name='勝率',
                                   line=dict(color='#A32D2D', width=2),
                                   marker=dict(size=10),
                                   text=[f"{v:.0f}%" for v in _sens_df['勝率(%)']],
                                   textposition='top center'),
                        secondary_y=False
                    )
                    fig_sens.add_trace(
                        go.Scatter(x=_sens_df['持有天數'], y=_sens_df['平均報酬(%)'],
                                   mode='lines+markers+text', name='平均報酬',
                                   line=dict(color='#888780', width=2, dash='dot'),
                                   marker=dict(size=10),
                                   text=[f"{v:+.1f}%" for v in _sens_df['平均報酬(%)']],
                                   textposition='bottom center'),
                        secondary_y=True
                    )
                    fig_sens.update_layout(
                        height=320,
                        margin=dict(l=10, r=10, t=20, b=10),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
                        template=chart_theme if 'chart_theme' in dir() else 'plotly',
                    )
                    fig_sens.update_yaxes(title_text="勝率 (%)", secondary_y=False)
                    fig_sens.update_yaxes(title_text="平均報酬 (%)", secondary_y=True)
                    st.plotly_chart(fig_sens, use_container_width=True)

                    # 詳細數字表
                    _sens_disp = _sens_df.copy()
                    _sens_disp['勝率(%)']     = _sens_disp['勝率(%)'].apply(lambda v: f"{v:.0f}%")
                    _sens_disp['平均報酬(%)'] = _sens_disp['平均報酬(%)'].apply(lambda v: f"{v:+.2f}%")
                    _sens_disp['MDD(%)']      = _sens_disp['MDD(%)'].apply(lambda v: f"{v:+.2f}%")
                    _sens_disp['Sharpe']      = _sens_disp['Sharpe'].apply(lambda v: f"{v:+.2f}")
                    st.dataframe(_sens_disp, use_container_width=True, hide_index=True)

                    # 提示:最佳持有期
                    _best_wr = max(_sens_rows, key=lambda r: r['勝率(%)'])
                    _best_ret = max(_sens_rows, key=lambda r: r['平均報酬(%)'])
                    _best_sharpe = max(_sens_rows, key=lambda r: r['Sharpe'])
                    _hints_sens = [
                        f"勝率最高 — **{_best_wr['持有天數']}** ({_best_wr['勝率(%)']:.0f}%)",
                        f"報酬最高 — **{_best_ret['持有天數']}** ({_best_ret['平均報酬(%)']:+.2f}%)",
                        f"風險調整最佳 — **{_best_sharpe['持有天數']}** (Sharpe {_best_sharpe['Sharpe']:+.2f})",
                    ]
                    st.info("💡 " + " / ".join(_hints_sens))
                    st.caption(
                        "⚠️ **怎麼挑、別踩雷:**\n\n"
                        "• 三個「最佳」可能指向不同天數——**Sharpe(風險調整後報酬)最值得參考**,"
                        "因為它同時考慮報酬和波動,不像單看勝率或報酬會偏頗。\n\n"
                        "• **久抱報酬高 ≠ 比較好**:40 日報酬通常比 5 日高,但那只是抱比較久、報酬沒年化。"
                        "同一段時間,抱 5 天可以做好幾趟。要比效率,心裡用「報酬 ÷ 持有天數」換算一下。\n\n"
                        "• **持有期越長、樣本越少**(訊號要夠老才算得出 40 日結果),"
                        "所以 40 日那點的勝率/報酬穩定度通常比 5 日差,別只因為它數字漂亮就採用。"
                    )
                else:
                    st.info("4 個持有天數下皆無觸發紀錄,試試擴大期間或換訊號組合。")

        # ── 觸發明細表 ──
        trades = _bt['trades']
        if not trades.empty:
            st.divider()
            top_n_bt = min(50, len(trades))
            st.markdown(f"**觸發明細(顯示前 {top_n_bt} 筆 / 共 {len(trades):,} 筆)**")

            display_df = trades.head(top_n_bt).copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df['名稱'] = display_df['stock_id'].map(ui_name_map).fillna('')
            display_df['進場'] = display_df['entry_price'].apply(lambda x: f"{x:,.1f}")
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


with _tab_sent:
    st.caption(
        "整合 6 個市場訊號的綜合溫度計 — 分數高(偏熱/樂觀)代表多頭情緒強,"
        "分數低(偏冷/恐慌)代表市場恐慌或空頭佔優。"
        "任一指標抓取失敗自動降級計算,不影響其他指標。"
    )

    try:
        _s = _get_sentiment_and_persist()
    except Exception as _e:
        _s = None
        st.error(f"情緒指標計算失敗: {_e}")

    if _s and _s.get("temperature") is not None:
        # ── 溫度計主數字 ──
        _temp = _s["temperature"]
        _tlabel = _s["label"]
        _ticon  = _s["icon"]

        _tcolor = (
            "#16a34a" if _temp >= 70 else
            "#65a30d" if _temp >= 55 else
            "#ca8a04" if _temp >= 45 else
            "#ea580c" if _temp >= 30 else
            "#dc2626"
        )
        st.markdown(
            f"""<div style="
                text-align:center;
                padding:18px 12px 10px;
                background:rgba(0,0,0,0.04);
                border-radius:12px;
                margin-bottom:16px;
            ">
            <div style="font-size:52px;font-weight:800;color:{_tcolor};">{_temp}</div>
            <div style="font-size:20px;color:#666;">/ 100 &nbsp; {_ticon} &nbsp; {_tlabel}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        # ── 各指標卡片 ──
        _ind = _s["indicators"]
        _CARD_DEFS = [
            ("vix",            "🇺🇸 美股 VIX",       lambda v: f"{v['value']}" if v.get('value') is not None else "N/A",
             lambda v: v.get('label','N/A'), "美股恐慌指數。< 15 樂觀；> 30 恐慌。"),
            ("taiex_vix",      "🇹🇼 台指波動率",     lambda v: f"{v['value']}" if v.get('value') is not None else "N/A",
             lambda v: f"歷史 {int(v['pct_rank'])}%位 {v['label']}" if v.get('pct_rank') is not None else v.get('label','N/A'),
             "從 ^TWII 算 20 日年化實現波動率(std × √252)。高百分位 = 波動升溫、恐慌情緒高。"),
            ("taiex_pos",      "📐 大盤位階",        lambda v: f"{v['value']:+}%" if v.get('value') is not None else "N/A",
             lambda v: v.get('label','N/A'), "加權指數 vs MA60 乖離率。> +8% 過熱；< -8% 深跌逢低。"),
            ("margin_level",   "💰 融資水位",        lambda v: f"{int(v['pct_rank'])}%位" if v.get('pct_rank') is not None else "N/A",
             lambda v: v.get('label','N/A'), "全市場融資餘額近 90 日百分位。高水位代表散戶槓桿偏重。"),
            ("fi_futures",     "🏦 外資期貨",
             lambda v: f"{v['value']:+,}口" if v.get('value') is not None else "N/A",
             lambda v: (
                 f"{v.get('label','N/A')}"
                 + (f" · 歷史 {int(v['pct_rank'])}%位"
                    if v.get('mode') == 'percentile' and v.get('pct_rank') is not None
                    else (f" · 累積{v.get('n_days',0)}/20日"
                          if v.get('mode') == 'absolute' else ""))
             ),
             "外資大台期貨未平倉淨口數(TAIFEX)。"
             "累積 ≥ 20 日後改用 90 日歷史百分位(更穩,永久不用校準),"
             "之前用絕對門檻 ±15k/±40k(2025~2026 放寬版)。"),
            ("retail_futures", "👥 散戶估算",
             lambda v: (f"{int(v['pct'])}%" if v.get('pct') is not None else "N/A"),
             lambda v: (
                 f"{v.get('label','N/A')}"
                 + (
                     f" · 歷史百分位"
                     if v.get('mode') == 'percentile' else
                     (f" · 線性估算 (累積{v.get('n_days',0)}/20日)"
                      if v.get('mode') == 'linear' else "")
                 )
                 + (f" · {v['value']:+,}口" if v.get('value') is not None else "")
             ),
             "散戶部位 0~100% 指數(50%=中性、100%=極多、0%=極空,反指標)。"
             "微台優先、小台備援。累積 ≥ 20 日後改用 90 日歷史百分位(更穩),"
             "之前用 ±30k 為半幅的線性估算。"),
        ]

        _cols = st.columns(4)
        for _i, (_key, _title, _val_fn, _lbl_fn, _help) in enumerate(_CARD_DEFS):
            _v = _ind.get(_key, {})
            _score = _v.get("score")
            _card_icon = _v.get("icon", "⚪")
            _val_str = _val_fn(_v)
            _lbl_str = _lbl_fn(_v)
            with _cols[_i % 4]:
                st.metric(
                    label=f"{_card_icon} {_title}",
                    value=_val_str,
                    delta=_lbl_str if _lbl_str != "N/A" else None,
                    delta_color="off",
                    help=_help,
                )

        # ── 分數條形圖 ──
        st.divider()
        _bar_rows = []
        _LABELS = {
            "vix": "美股 VIX", "taiex_vix": "台指波動率", "taiex_pos": "大盤位階",
            "margin_level": "融資水位",
            "fi_futures": "外資期貨", "retail_futures": "散戶估算",
        }
        for _k, _lbl in _LABELS.items():
            _v = _ind.get(_k, {})
            if _v.get("score") is not None:
                _bar_rows.append({"指標": _lbl, "分數": _v["score"], "標籤": _v.get("label","")})
        if _bar_rows:
            import plotly.express as px
            _bar_df = pd.DataFrame(_bar_rows)
            _bar_colors = [
                "#16a34a" if s >= 65 else "#84cc16" if s >= 50 else "#eab308" if s >= 40 else "#f97316" if s >= 25 else "#dc2626"
                for s in _bar_df["分數"]
            ]
            _fig_bar = go.Figure(go.Bar(
                x=_bar_df["指標"], y=_bar_df["分數"],
                text=_bar_df["標籤"], textposition="outside",
                marker_color=_bar_colors,
            ))
            _fig_bar.update_layout(
                height=280, margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(range=[0, 105], title="情緒分數(0=極冷, 100=極熱)"),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            _fig_bar.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5)
            st.plotly_chart(_fig_bar, use_container_width=True)

        # ── 📈 大盤情緒趨勢圖(讀 sentiment_history.json,看溫度怎麼演進) ──
        # 利用每次 _get_sentiment_and_persist 累積的歷史檔,畫出近 30/90 日溫度走勢。
        # 比單看當下溫度更有用:可以分辨「溫度下行還在繼續」vs「已從谷底回升」。
        _trend_hist = load_sentiment_history(CACHE_DIR)
        if _trend_hist and len(_trend_hist) >= 2:
            st.divider()
            _trend_col1, _trend_col2 = st.columns([3, 1])
            with _trend_col1:
                st.markdown("#### 📈 市場溫度走勢")
            with _trend_col2:
                _trend_range = st.radio(
                    "範圍", ["近 30 日", "近 90 日", "全部"],
                    horizontal=True, label_visibility="collapsed",
                    key="_sent_trend_range",
                )

            _n_keep = (30 if _trend_range == "近 30 日"
                       else 90 if _trend_range == "近 90 日"
                       else len(_trend_hist))
            _hist_slice = _trend_hist[-_n_keep:] if len(_trend_hist) > _n_keep else _trend_hist

            _dates = [pd.to_datetime(h["date"]) for h in _hist_slice]
            _temps = [h.get("temp") for h in _hist_slice]

            # 線色:依當前(最後一筆)溫度的區間給色,跟主數字一致
            _last_t = _temps[-1] if _temps else 50
            _line_color = (
                "#16a34a" if _last_t >= 70 else
                "#65a30d" if _last_t >= 55 else
                "#ca8a04" if _last_t >= 45 else
                "#ea580c" if _last_t >= 30 else
                "#dc2626"
            )

            _fig_trend = go.Figure()
            _fig_trend.add_trace(go.Scatter(
                x=_dates, y=_temps,
                mode="lines+markers",
                line=dict(color=_line_color, width=2.5),
                marker=dict(size=6),
                name="溫度",
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>溫度 %{y}/100<extra></extra>",
            ))
            # 4 條參考線標示區間
            for _y, _lbl, _col, _dash in [
                (70, "70 偏熱", "#16a34a", "dash"),
                (55, "55 略偏多", "#65a30d", "dot"),
                (45, "45 中性", "gray",    "dot"),
                (30, "30 偏冷", "#dc2626", "dash"),
            ]:
                _fig_trend.add_hline(
                    y=_y, line_dash=_dash, line_color=_col, opacity=0.4,
                    annotation_text=_lbl, annotation_position="right",
                    annotation_font=dict(size=10, color=_col),
                )

            _fig_trend.update_layout(
                height=260, margin=dict(l=10, r=60, t=10, b=20),
                yaxis=dict(range=[0, 100], title="溫度"),
                xaxis=dict(title=""),
                plot_bgcolor="rgba(0,0,0,0.03)",
                showlegend=False,
            )
            st.plotly_chart(_fig_trend, use_container_width=True)

            # 統計摘要
            _avg = sum(_temps) / len(_temps)
            _high = max(_temps)
            _low  = min(_temps)
            _delta = _temps[-1] - _temps[0]
            _arrow = "↗" if _delta > 2 else ("↘" if _delta < -2 else "→")
            _stat_cols = st.columns(4)
            _stat_cols[0].metric("區間平均", f"{_avg:.0f}")
            _stat_cols[1].metric("區間最高", f"{_high}")
            _stat_cols[2].metric("區間最低", f"{_low}")
            _stat_cols[3].metric(f"變化 {_arrow}", f"{_delta:+d}",
                                 help=f"從 {_hist_slice[0]['date']} 到 {_hist_slice[-1]['date']}")
        elif _trend_hist:
            st.info(f"📈 市場溫度走勢:已累積 {len(_trend_hist)} 日,**明天起會自動畫成趨勢圖**。")

        # ── 🎯 溫度計有效性回測(溫度 vs 大盤後續報酬) ──
        # 回答「這溫度計準不準?」:把歷史溫度按高低分組,看各組大盤後續 5/10 日實際走勢。
        st.divider()
        st.markdown("#### 🎯 溫度計準不準?(溫度 vs 大盤後續報酬)")
        _sbt = _load_sentiment_backtest_cached()
        if _sbt.get("error"):
            st.info(f"📊 {_sbt['error']}")
        else:
            _bands = _sbt["bands"]
            _corr  = _sbt.get("corr", {})
            # 分組表
            _bt_rows = []
            for _b in _bands:
                if _b["n_total"] == 0:
                    continue
                _row = {"溫度區間": _b["band"], "樣本": _b["n_total"]}
                for _n in _sbt["n_days_list"]:
                    _cell = _b.get(f"{_n}d")
                    if _cell:
                        _row[f"{_n}日勝率"]  = f"{_cell['win_rate']*100:.0f}%"
                        _row[f"{_n}日平均"]  = f"{_cell['avg']:+.2f}%"
                    else:
                        _row[f"{_n}日勝率"]  = "—"
                        _row[f"{_n}日平均"]  = "—"
                _bt_rows.append(_row)
            if _bt_rows:
                st.dataframe(pd.DataFrame(_bt_rows), use_container_width=True, hide_index=True)

            # 相關係數解讀(以 5 日為主)
            _r5 = _corr.get(5)
            _n5 = _sbt.get("n_eval", {}).get(5, 0)
            if _r5 is not None:
                if _r5 > 0.2:
                    _interp = (f"**正相關 (r={_r5:+.2f})**:溫度越高、大盤後續越偏漲 → "
                               f"溫度計偏「**順勢/趨勢確認**」,高溫可順勢、低溫保守。")
                elif _r5 < -0.2:
                    _interp = (f"**負相關 (r={_r5:+.2f})**:溫度越高、大盤後續反而越偏跌 → "
                               f"溫度計偏「**反指標**」,高溫該減碼、低溫(恐慌)反而是機會。")
                else:
                    _interp = (f"**幾乎不相關 (r={_r5:+.2f})**:目前溫度對大盤後續 5 日報酬"
                               f"沒有明顯預測力,當參考就好,別當訊號。")
                st.caption(f"📐 溫度 vs 後續 5 日報酬相關係數:{_interp}(樣本 {_n5} 筆)")

            if (_sbt.get("n_eval", {}).get(5, 0) or 0) < 30:
                st.warning(
                    f"⚠️ 樣本僅 {_sbt.get('n_eval', {}).get(5, 0)} 筆,統計極不穩定。"
                    f"溫度歷史要累積夠久(≥ 30 筆、跨多空)結論才可信,目前看趨勢就好。"
                )

        st.caption(
            "⚠️ 情緒指標為輔助參考，不構成買賣建議。"
            "外資期貨 / 散戶估算資料來源為 TAIFEX 公開揭露，每交易日盤後更新一次。"
        )

        # ── 🤖 AI 大盤操作建議 ─────────────────────────────────────
        # 把 6 個情緒指標 + 今日達標數 + 歷史 5 日勝率 + 大盤多空狀態
        # 餵給 AI,請它給今日具體部位建議 + 風險點。
        # Button-triggered (節省 API 配額) + session_state 快取
        st.divider()
        st.markdown("#### 🤖 AI 大盤操作建議")
        st.caption(
            "把 6 個情緒指標 + 今日達標數 + 歷史勝率餵給 AI,"
            "請它給今日具體部位建議與風險點。**仍非投資建議,自行判斷。**"
        )

        # 取得既有的績效 / 大盤 / 達標數 三項背景資料(都有就加進 prompt)
        _bg_perf = ""
        try:
            _perf_for_ai = _load_performance_cached()
            _o = _perf_for_ai.get("overall", {})
            if "win_rate_5d" in _o:
                _bg_perf = (
                    f"\n- 過去 5 日勝率: {_o['win_rate_5d']*100:.0f}% "
                    f"(avg {_o.get('avg_return_5d', 0):+.2f}%, 樣本 {_o.get('n_5d', 0)} 筆)"
                )
        except Exception:
            pass

        _bg_market = ""
        if meta is not None:
            _bull = meta.get('market_bullish', None)
            _twii_pct = meta.get('twii_pct')
            if _bull is not None:
                if not _bull:
                    _state_desc = "空頭(跌破季線)"
                elif meta.get('market_consolidating'):
                    _state_desc = "盤整修正(站上季線但跌破月線/近20日下跌,RS 不計分、籌碼門票啟用)"
                else:
                    _state_desc = "多頭(站上季線)"
                _bg_market = f"\n- 大盤狀態: {_state_desc}"
                if pd.notna(_twii_pct):
                    _bg_market += f", 今日 {_twii_pct:+.2f}%"

        _bg_hit = ""
        if df is not None:
            _bg_hit = f"\n- 今日達標個股: {len(df)} 檔"
            if len(df) > 0 and '產業' in df.columns:
                _ti_series = df['產業'].dropna()
                _ti_series = _ti_series[_ti_series != ""]
                if not _ti_series.empty:
                    _ti_vc = _ti_series.value_counts()
                    _bg_hit += f", 主流產業: {_ti_vc.idxmax()} ({_ti_vc.max()} 檔)"

        # 把每個指標的「原始值 + 標籤 + 分數」攤平給 AI
        _IND_NAMES_AI = {
            "vix":            "美股 VIX",
            "taiex_vix":      "台指波動率",
            "taiex_pos":      "大盤位階",
            "margin_level":   "融資水位",
            "fi_futures":     "外資期貨",
            "retail_futures": "散戶估算",
        }
        _ind_detail_lines = []
        for _k, _name in _IND_NAMES_AI.items():
            _v = _ind.get(_k, {})
            if _v.get("score") is None:
                continue
            _val = _v.get("value")
            _val_str = "?"
            if _k == "vix" and _val is not None:
                _val_str = f"{_val}"
            elif _k == "taiex_vix" and _val is not None:
                _val_str = (f"{_val}% (歷史 {int(_v['pct_rank'])}%位)"
                            if _v.get('pct_rank') is not None else f"{_val}%")
            elif _k == "taiex_pos" and _val is not None:
                _val_str = f"{_val:+}% (vs MA60 乖離率)"
            elif _k == "margin_level" and _v.get('pct_rank') is not None:
                _val_str = f"{int(_v['pct_rank'])}%位 (90 日)"
            elif _k == "fi_futures" and _val is not None:
                _val_str = f"{_val:+,}口"
            elif _k == "retail_futures" and _v.get('pct') is not None:
                _val_str = f"{int(_v['pct'])}% 部位指數"
            _ind_detail_lines.append(
                f"  - {_name}: {_val_str} → {_v.get('label', '?')} (分數 {_v.get('score')}/100)"
            )
        _ind_detail_str = "\n".join(_ind_detail_lines)

        _ai_market_prompt = (
            f"你是一位台灣股市的資深策略分析師。基於以下今日數據,給出具體操作建議。\n\n"
            f"【大盤情緒溫度計】\n"
            f"- 綜合溫度: {_temp}/100 ({_tlabel})\n"
            f"- 各指標詳情:\n{_ind_detail_str}\n\n"
            f"【市場現況】{_bg_market}{_bg_hit}{_bg_perf}\n\n"
            f"請用繁體中文(台灣用語)給出 120~180 字的具體建議,涵蓋三點:\n"
            f"1. 部位建議:滿倉/七成/五成/三成/減碼,並說明為什麼\n"
            f"2. 選股聚焦:是否只挑高分(≥9)、避開哪類產業或型態\n"
            f"3. 風險點 + 觸發條件:哪些訊號出現時要調整部位\n\n"
            f"直接輸出純文字,絕對不要使用 Markdown 語法(不要 *、#、反引號、項目符號)。"
            f"語氣專業客觀,避免「肯定會漲」這類絕對化用詞,多用「可考慮」「留意」等語氣。"
        )

        _btn_col1, _btn_col2 = st.columns([3, 1])
        with _btn_col2:
            _gen_advice_clicked = st.button(
                "🤖 產生 AI 建議",
                type="primary",
                use_container_width=True,
                help=("呼叫 OpenRouter 免費模型(10~20 秒)。同一 session 結果會快取,"
                      "可手動重新產生。"),
            )

        if _gen_advice_clicked:
            with st.spinner("AI 分析中(預計 10~20 秒)..."):
                _model_name_adv, _ai_advice_text = call_openrouter_ai(
                    _ai_market_prompt, max_tokens=500
                )
                if _ai_advice_text:
                    st.session_state["ai_market_advice"] = {
                        "model":       _model_name_adv,
                        "text":        _ai_advice_text,
                        "temp_when":   _temp,
                        "label_when":  _tlabel,
                        "ts":          pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M"),
                    }
                else:
                    st.error("❌ AI 暫時無法回應(可能當日 API 配額已滿,稍後再試)。")

        _a_cached = st.session_state.get("ai_market_advice")
        if _a_cached:
            # 若快取時的溫度跟當下差 > 5 度,提示資訊可能過時
            _stale = abs(_a_cached["temp_when"] - _temp) > 5
            _stale_note = (" · ⚠️ 溫度已變化,建議重新產生"
                           if _stale else "")
            st.markdown(
                f"<div style='background:#fff7ed;border-left:4px solid #f97316;"
                f"padding:14px 16px;border-radius:8px;margin-top:8px;'>"
                f"<div style='font-size:12px;color:#6b7280;margin-bottom:8px;'>"
                f"🤖 <b>{html.escape(str(_a_cached['model']))}</b> · "
                f"產生於 {_a_cached['ts']} · "
                f"當時溫度 {_a_cached['temp_when']}/100 ({html.escape(_a_cached['label_when'])})"
                f"{_stale_note}</div>"
                f"<div style='font-size:15px;line-height:1.75;color:#1f2937;"
                f"white-space:pre-wrap;'>{html.escape(_a_cached['text'])}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("👆 點上方按鈕產生今日操作建議(免費模型,10~20 秒)")

        # ── 📚 各指標完整說明 ──
        with st.expander("📚 各指標完整說明 / FAQ", expanded=False):
            st.markdown("""
### 🌡️ 溫度計怎麼算出來的?

把每個有效指標都換算成 **0~100 的情緒分數**(0 = 極冷恐慌、100 = 極熱樂觀),
然後**加權平均**得到總溫度:

| 指標 | 權重 | 為什麼這樣設 |
|---|---|---|
| 🇺🇸 美股 VIX | 22% | 全球恐慌錨,台股開盤先看它 |
| 🇹🇼 台指波動率 | 11% | 在地恐慌訊號 |
| 📐 大盤位階 | 22% | 過熱/超跌的關鍵 |
| 💰 融資水位 | 11% | 散戶槓桿水位(反指標) |
| 🏦 外資期貨 | 22% | 主力期貨佈局,聰明錢方向盤 |
| 👥 散戶估算 | 12% | 散戶期貨方向(反指標) |

> 📌 **權重設計原則**:主軸 3 個(VIX、位階、外資)各佔 ~ 22%,共 ~ 66%;
> 輔助 3 個(波動率、融資、散戶)各佔 ~ 11%,共 ~ 33%。
> 任一指標 N/A 時,剩餘指標權重會自動歸一化,**不會因為一個失敗整套停擺**。
>
> 已下架「融資週變化」── 與「融資水位」訊號重疊度高(同一資料源、僅
> level vs momentum 視角差異),保留水位即可代表散戶融資面情緒。

> **任一指標抓取失敗,該指標標 N/A,剩下指標自動歸一化權重繼續算**,不會因為單點失敗整套停擺。

最終分數 → 文字標籤對應:

| 溫度 | 標籤 | 解讀 |
|---|---|---|
| ≥ 70 | ☀️ 偏熱(樂觀) | 多頭情緒強,但留意是否「市場已過熱」 |
| 55~69 | 🌤️ 略偏多 | 健康多頭氛圍 |
| 45~54 | 🌥️ 中性 | 觀望盤、方向不明 |
| 30~44 | 🌦️ 略偏空 | 警戒,但未到恐慌 |
| < 30 | ❄️ 偏冷(恐慌) | 恐慌情緒濃,**反向常是逢低機會** |

---

### 1️⃣ 🇺🇸 美股 VIX

- **資料來源**:Yahoo Finance `^VIX`(芝加哥選擇權交易所恐慌指數)
- **意義**:美股 S&P 500 未來 30 天的隱含波動率
- **判讀**:

| VIX 值 | 標籤 | 分數 |
|---|---|---|
| < 15 | 樂觀 🟢 | 75 |
| 15~20 | 偏低 🟢 | 65 |
| 20~25 | 中性 🟡 | 50 |
| 25~30 | 警戒 🟠 | 30 |
| > 30 | 恐慌 🔴 | 15 |

> 📌 **為什麼台股要看美股 VIX?** 因為台股對美股連動性極高(尤其電子權值股),
> VIX 飆升通常隔日台股開低機率大。VIX 是「全球風險偏好」的最佳代理。

---

### 2️⃣ 🇹🇼 台指波動率(實現波動率)

- **資料來源**:Yahoo Finance `^TWII` 自行計算
- **算法**:`std(20 日日對數報酬) × √252 × 100`(年化標準差)
- **判讀**:看當前值在**過去 90 日**的百分位

| 百分位 | 標籤 | 分數 |
|---|---|---|
| < 25% | 偏低 🟢 | 70 |
| 25~50% | 中低 🟢 | 60 |
| 50~75% | 中高 🟡 | 40 |
| > 75% | 偏高 🔴 | 20 |

> 📌 原本用富邦 VIX ETF(00677U)當代理,該 ETF 2024 年下市後改為直接從加權指數算實現波動率,**資料源穩定無下市風險**。
> 台股年化實現波動率常態約 10~20%,恐慌時可飆 30%+。

---

### 3️⃣ 📐 大盤位階(乖離率)

- **資料來源**:Yahoo Finance `^TWII` 收盤 vs MA60
- **算法**:`(目前 - MA60) / MA60 × 100`
- **判讀**:

| 乖離 | 標籤 | 分數 | 解讀 |
|---|---|---|---|
| > +8% | 過熱 🔴 | 15 | 短線拉太遠,留意修正 |
| +3~+8% | 略高 🟡 | 40 | 健康多頭 |
| -3~+3% | 正常 🟢 | 60 | 多頭正常運行 |
| -8~-3% | 略低 🟢 | 70 | 接近年線,觀察反彈 |
| < -8% | 深跌 🟠 | 80 | **逢低買進機會**(反向加分) |

> 📌 為什麼 < -8% 反而分數高?因為長期均值回歸,深跌後反彈期望值較高。
> 這個指標跟「漲跌方向」無關,只看「相對位置」。

---

### 4️⃣ 💰 融資水位

- **資料來源**:cache/margin_*.parquet(每日 TWSE/TPEx 抓)
- **算法**:目前融資餘額在**過去 90 日**的百分位
- **判讀**(反指標):

| 百分位 | 標籤 | 分數 |
|---|---|---|
| < 25% | 偏低 🟢 | 70(健康) |
| 25~75% | 中等 🟡 | 50 |
| 75~90% | 偏高 🟠 | 35 |
| > 90% | 極高 🔴 | 20(過熱警訊) |

> 📌 「融資」 = 借錢買股票的槓桿部位,散戶占大宗。**散戶急著加碼往往在頭部、急著認賠往往在底部**,所以這是反指標。
>
> 📌 **為什麼只看水位、不看週變化?** 同一份 margin parquet 算出來的「絕對水位」和「短期動能」訊號重疊度高,在實務上**水位反映更穩**(週變化容易被連假/補班日噪音干擾)。原本的 `get_margin_change()` 函式仍保留供需要直接調用。

---

### 5️⃣ 🏦 外資期貨(大台 TX 未平倉淨口數)

- **資料來源**:TAIFEX 三大法人未平倉(每交易日盤後 ~15:00 更新)
- **意義**:外資在大台指期貨的多空淨部位
- **評分**(雙模式自動切換,跟散戶估算同一招):

**模式 A — 90 日歷史百分位(累積 ≥ 20 日後啟用,主要)**

當前淨口數放進「過去 90 日的分布」算百分位:

| 百分位 | 標籤 | 分數 | 解讀 |
|---|---|---|---|
| > 80% | 強多 🟢 | 75 | 比近 3 個月任何時候都更偏多 |
| 60~80% | 偏多 🟢 | 65 | |
| 40~60% | 中性 🟡 | 50 | |
| 20~40% | 偏空 🟠 | 35 | |
| < 20% | 強空 🔴 | 20 | 比近 3 個月任何時候都更偏空 |

**模式 B — 絕對門檻(歷史不足 20 日的過渡用,2025~2026 放寬版)**

| 淨口數 | 標籤 | 分數 |
|---|---|---|
| > +40,000 | 強多 🟢 | 75 |
| +15,000~+40,000 | 偏多 🟢 | 65 |
| -15,000~+15,000 | 中性 🟡 | 50 |
| -40,000~-15,000 | 偏空 🟠 | 35 |
| < -40,000 | 強空 🔴 | 20 |

> 📌 **為什麼從絕對門檻改成百分位?** 外資總部位會隨市值規模演進:
>
> | 時期 | 顯著方向門檻 |
> |---|---|
> | 2018~2020 | ±15k |
> | 2021~2023 | ±20k |
> | 2025~2026 上半 | ±30k |
> | 2025~2026 近期 | 動輒 ±40~50k |
>
> **每隔一段時間就要重新校準絕對門檻**,有點累。改用百分位制是**相對自己過去 3 個月**,
> 自動隨市場結構演進、永久不用調,跟散戶估算同一套邏輯。
>
> 📌 **歷史檔位置**:`cache/fi_futures_history.json`,每天 append、保留最近 90 日。
> 累積到 20 日後自動切換到百分位模式。
>
> 外資期貨被視為「聰明錢的方向盤」,持續強空往往伴隨現貨賣壓。但**單日翻轉就追**容易被洗,看趨勢比看單日重要。

---

### 6️⃣ 👥 散戶估算(0~100% 統一指數)

- **資料來源**:TAIFEX 三大法人未平倉
  - **主來源:微型臺指期貨**(散戶占比 ~ 90%,最純散戶代理)
  - **備援:小型臺指期貨**(MXF,散戶占比 ~ 50%,訊號較雜)
- **算法**:`散戶淨口數 ≈ -(自營商 + 投信 + 外資 合計淨口數)`
- **顯示方式**:**永遠顯示 0~100% 散戶部位指數**(反指標)
  - 50% = 中性
  - 100% = 極度散戶多頭(反向警訊)
  - 0% = 極度散戶空頭(反向利多)
- **評分標準**(對應 % 區間):

| 散戶部位 % | 標籤 | 分數 | 解讀 |
|---|---|---|---|
| > 80% | 散戶極多 🔴 | 25 | 散戶嚴重偏多 → **反向警訊** |
| 60~80% | 散戶偏多 🟠 | 40 | |
| 40~60% | 中性 🟡 | 55 | |
| 20~40% | 散戶偏空 🟢 | 65 | |
| < 20% | 散戶極空 🟢 | 75 | **反向利多訊號** |

**這個 % 怎麼算的?(雙模式自動切換,使用者無感)**

| 累積歷史 | 計算方式 | 卡片副標 |
|---|---|---|
| **< 20 日**(過渡) | **線性估算**:以 ±30k 為半幅線性映射,例:+15k → 75%、-30k → 0% | "線性估算 (累積 N/20 日)" |
| **≥ 20 日**(主要) | **歷史百分位**:當前值在過去 90 日的位階,例:位居前 25% → 75% | "歷史百分位" |

> 📌 **為什麼一開始是線性、後來是百分位?**
> - 系統剛上線時沒有歷史資料,先用 ±30k 為半幅做線性歸一化(快速可用)
> - 累積 20 個交易日(約一個月)後,改用真實的歷史百分位,**訊號穩定度大幅提升**
> - 切換是自動的,使用者看到的永遠是「0~100% 指數」,門檻語意一致
>
> 📌 **歷史檔位置**:`cache/retail_futures_history.json`,每天 append、保留最近 90 日。
> 若不小心刪掉,系統會從頭重新累積(20 日後自動恢復百分位制)。
>
> 📌 **為什麼用微台不用小台?**
> 小台 MXF 法人也常用(尤其自營商避險),散戶純度 ~ 50%;
> 微台合約值約 1/10 小台,**只有散戶會玩**,純度 ~ 90%。
>
> ⚠️ 兩種來源的歷史**分開計算百分位**(避免尺度不一致汙染分布)。
>
> **散戶長期勝率偏低,反向是傳統策略**。% 高(分數低) → 警戒;% 低(分數高) → 反而看多。

> 📌 **為什麼從絕對門檻改成百分位?**
> 微台口數會隨契約規模、散戶習慣演進而變大,**固定門檻長期會失準**。
> 百分位制是相對自身過去 3 個月,**永遠不需要手動調**,跟「融資水位」同一套邏輯。
>
> 📌 **歷史檔位置**:`cache/retail_futures_history.json`,每次抓到當日資料就 append、自動保留最近 90 日。
> 若不小心刪掉,系統會從頭重新累積(20 日後自動恢復百分位制)。
>
> 📌 **為什麼用微台不用小台?**
> 小台 MXF 法人也常用(尤其自營商避險),散戶純度只有 ~ 50%;
> 微台合約值約 1/10 小台,基本上**只有散戶會玩**(法人覺得太小),純度高達 ~ 90%。
>
> ⚠️ 散戶/小台 兩個來源的歷史是**分開計算百分位**的(避免尺度不一致汙染分布)。
>
> **散戶長期勝率偏低,反向是傳統策略**。散戶大舉做多(分數低) → 警戒;
> 散戶大舉做空(分數高) → 反而看多。

---

### 💡 怎麼用這個溫度計?

**情境 1:溫度 ≥ 70(偏熱)**
- 多頭情緒強,**選股訊號比較容易出現**,但要小心**過熱回檔**
- 建議搭配「大盤位階」看,若位階也 > +8% 就要謹慎、減碼

**情境 2:溫度 45~69(中性偏多)**
- 標準操作區間,正常依選股訊號進場
- 這是策略最舒服的環境

**情境 3:溫度 30~44(略偏空)**
- 警戒區,**降低部位、嚴格停損**
- 選股訊號變少是正常的,寧缺勿濫

**情境 4:溫度 < 30(極度恐慌)**
- 反而是**長線買點的徵兆**(歷史上低於 25 往往出現在 V 型反轉前夕)
- 但短線仍有下行壓力,**分批承接、不要梭哈**

---

### ❓ 常見問題

**Q1:為什麼有時候顯示 N/A?**
- 該指標的資料源暫時拉不到(yfinance 限速 / TAIFEX 維護 / margin cache 還沒更新)
- 系統會自動降級,**其他指標仍正常算溫度**

**Q2:多久更新一次?**
- 整個 sentiment tab 快取 5 分鐘
- VIX / 台指波動率 / 大盤位階 用 yfinance(near real-time,有 15 分鐘延遲)
- TAIFEX 期貨資料**每日盤後 15:00 後**才有當日數字
- 融資資料**每日盤後 15:00 後**由 fetch_cache.py 抓

**Q3:溫度跟「漲跌」是什麼關係?**
- **沒有直接關聯**。溫度高 ≠ 今天會漲、溫度低 ≠ 今天會跌
- 它衡量的是「**市場情緒**」而非「**短線方向**」
- 高溫常出現在多頭末段(風險升高)、低溫常出現在恐慌底部(機會浮現)

**Q4:能用來當買賣訊號嗎?**
- ❌ 不建議單獨當訊號
- ✅ 適合當「環境濾網」:溫度極端時調整部位、平時搭配個股選股訊號使用

**Q5:「🎯 溫度計準不準?」那張表怎麼看?**
- 它拿**歷史每日溫度**對照 **^TWII 後續 5/10 日實際報酬**,按溫度高低分組,驗證這套溫度計到底有沒有預測力
- **看相關係數 (r)** 是關鍵:
  - **r > +0.2(正相關)**:溫度越高、大盤後續越漲 → 溫度計偏「**順勢確認**」,高溫可順勢、低溫保守
  - **r < −0.2(負相關)**:溫度越高、大盤後續反而越跌 → 溫度計偏「**反指標**」,高溫該減碼、低溫(恐慌)反而是機會
  - **−0.2 ~ +0.2(幾乎不相關)**:目前沒明顯預測力,當參考就好、別當訊號
- ⚠️ **樣本 < 30 筆別當真**:相關係數在小樣本下極不穩,要溫度歷史累積夠久(≥ 30 筆、最好跨過一次多空)結論才可信
""")

    else:
        st.info("情緒指標資料暫時無法取得(網路限制或資料尚未更新),將在下次重整後自動重試。")


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
        # 籌碼信心分級概覽(欄位由 screening0515 產生;舊結果無此欄時跳過,避免 KeyError)
        if "籌碼信心" in df.columns:
            _hi  = int((df["籌碼信心"] == "🔥 高信心").sum())
            _mid = int((df["籌碼信心"] == "⭐ 中信心").sum())
            _cc1, _cc2, _cc3 = st.columns(3)
            _cc1.metric("🔥 高信心", f"{_hi} 檔",
                        help="大戶↑ 且 散戶↓ 共振 — 訊號歸因中 edge 最高的籌碼組合,優先看")
            _cc2.metric("⭐ 中信心", f"{_mid} 檔", help="大戶↑ 或 散戶↓ 其一成立")
            _cc3.metric("一般", f"{len(df) - _hi - _mid} 檔")
        score_min, score_max = int(df['總分'].min()), int(df['總分'].max())
        industries = sorted([i for i in df['產業'].dropna().unique() if i])
        if score_min != score_max:
            # 分數有高低 → 左滑桿 + 右產業篩選 並排
            fc1, fc2 = st.columns(2)
            with fc1:
                score_threshold = st.slider("總分 ≥", score_min, score_max, score_min)
            with fc2:
                selected_industries = st.multiselect("產業篩選", industries, default=industries)
        else:
            # 達標股全部同分 → 不需分數滑桿,產業篩選用整列寬(避免左側空一塊)
            score_threshold = score_min
            st.caption(f"📊 達標股皆為 **{score_min} 分**(同分,免用分數篩選)")
            selected_industries = st.multiselect("產業篩選", industries, default=industries)

        # 🔥 只看高信心:籌碼共振(大戶↑+散戶↓)。欄位存在才顯示;預設 False 避免下方 NameError。
        _only_hi = False
        if "籌碼信心" in df.columns:
            _only_hi = st.checkbox(
                "🔥 只看高信心(大戶↑+散戶↓ 共振)",
                value=False, key="_only_hi",
                help="只留籌碼共振的股 — 訊號歸因中 edge 最高的組合;當『優先觀察』,不代表自動買進。",
            )

        filtered = df[(df['總分'] >= score_threshold) & (df['產業'].isin(selected_industries))].copy()
        if _only_hi and "籌碼信心" in filtered.columns:
            filtered = filtered[filtered["籌碼信心"] == "🔥 高信心"]

        # 🚦 乖離煞車:標出離均線過遠(追高風險)的標的,可一鍵排除(#3 過熱/估值煞車的清單版)
        _daily_files = sorted(CACHE_DIR.glob("daily_*.parquet"))
        _bias_map = _load_bias_map(_daily_files[-1].stem if _daily_files else "none")
        if _bias_map and not filtered.empty:
            _codes = filtered['代號'].astype(str)
            filtered['乖離月線%'] = _codes.map(lambda c: _bias_map.get(c, {}).get('bias20'))
            filtered['乖離季線%'] = _codes.map(lambda c: _bias_map.get(c, {}).get('bias60'))
            filtered['過熱'] = _codes.map(lambda c: '🟡' if _bias_map.get(c, {}).get('hot') else '')
            _n_hot = int((filtered['過熱'] == '🟡').sum())
            if _n_hot:
                if st.checkbox(
                    f"🚦 排除過熱標的(乖離高、追高風險)— 目前 {_n_hot}/{len(filtered)} 檔",
                    value=False, key="_excl_hot",
                    help=f"乖離月線 > {BIAS_MA20_HOT:.0f}% 或 季線 > {BIAS_MA60_HOT:.0f}% 標 🟡;勾選後從清單與複製碼移除",
                ):
                    filtered = filtered[filtered['過熱'] != '🟡']

        # 🎯 籌碼抄底雷達(④ 實證固化:資減券增 + 大戶逆勢增持)。預設關閉 → 不啟用就不付建矩陣的 ~5 秒。
        # 勾選後「整個表替換為 13 檔雷達清單」——抄底雷達是『找弱股反轉』,與選股清單『追強勢』邏輯相反,
        # 命中股幾乎不會在選股清單裡,故獨立呈現才看得到。重用訊號回測分頁的同一份矩陣 cache。
        _dip_on = st.checkbox(
            "🎯 切換到籌碼抄底雷達(資減券增＋大戶逆勢增持)",
            value=False, key="_dip_on",
            help="勾選後表格切換為「全市場今天同時觸發兩個籌碼底部訊號」的股(與『追強勢』選股相反邏輯,故獨立顯示)。\n"
                 "④回測:78 筆、勝率 46%、平均 +3.95%、夏普 1.35(全多頭資料、中位數偏負)→ 當『觀察名單』,"
                 "別當閉眼買;進場等帶量確認、嚴設停損。第一次勾選需建訊號矩陣約 5 秒。",
        )
        if _dip_on:
            _radar_key = f"{_cache_date.strftime('%Y-%m-%d') if _cache_date is not None else 'no_data'}|v3-tierR"
            _dip_set, _dip_date = _load_dip_radar(_radar_key)
            _dstr = f"(依 {_dip_date} 融資券資料)" if _dip_date else "(暫無資料)"
            if not _dip_set:
                st.info(f"🎯 抄底雷達:今日全市場 **0** 檔命中 {_dstr}——此組合很挑,常整天 0~數檔,屬正常。")
                filtered = filtered.iloc[0:0]               # 清空表格,避免下方還顯示舊的追強勢清單
            else:
                _close_map = _load_latest_close_map()
                _ind_map   = _load_industry_name_map()
                _bias_for_radar = _bias_map if _bias_map else {}
                _rows = []
                for sid in sorted(_dip_set):
                    _rows.append({
                        "代號":      sid,
                        "名稱":      ui_name_map.get(sid, ""),
                        "最新價":    round(_close_map.get(sid), 2) if _close_map.get(sid) is not None else None,
                        "產業":      _ind_map.get(sid, ""),
                        "乖離月線%": _bias_for_radar.get(sid, {}).get('bias20'),
                        "乖離季線%": _bias_for_radar.get(sid, {}).get('bias60'),
                    })
                filtered = pd.DataFrame(_rows)
                st.success(f"🎯 抄底雷達命中 **{len(_dip_set)}** 檔 {_dstr}。"
                           f"與選股清單邏輯相反(找跌深+籌碼吃貨)→ 獨立呈現;當『觀察名單』,**別閉眼買**、進場等帶量、嚴設停損。")

        filtered = filtered.reset_index(drop=True)
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

        # ── 📌 進場節奏快速判斷(純量化規則,不打 AI,個股首頁自動顯示) ──
        # 規則:
        #   🟡 拉回再進 — 短線過熱:K>80 / 5 日漲幅 >10% / 乖離過大(追高煞車)
        #   🟢 可進場   — K<70 且 5 日漲幅 <8% 且站上 MA20 且站上 MA60 且乖離不過大
        #   🔵 觀察     — 其他(訊號模糊、跌破 MA20/MA60、新股資料不足)
        # 乖離煞車(#3 過熱/估值煞車):股價離均線太遠 → 追高風險高,即使 K/漲幅未過熱也黃燈。
        #   門檻(標準):MA20 乖離 > 12% 或 MA60(季線)乖離 > 20%。個股波動大,故比大盤的 ±8% 寬。
        BIAS20_HOT, BIAS60_HOT = BIAS_MA20_HOT, BIAS_MA60_HOT   # 與選股結果表共用模組常數
        _hist_quick = _cached_stock_history(sid)
        _verdict_quick = None
        _verdict_reason = ""
        if _hist_quick is not None and not _hist_quick.empty and len(_hist_quick) >= 20:
            try:
                _high_c = 'max' if 'max' in _hist_quick.columns else 'high'
                _low_c  = 'min' if 'min' in _hist_quick.columns else 'low'
                _close_s = _hist_quick['close']
                _latest  = float(_close_s.iloc[-1])
                _ma20    = float(_close_s.tail(20).mean())
                # MA60 需 60 筆,新股資料不足時設 None,後續判斷會跳過 MA60 條件
                _ma60    = float(_close_s.tail(60).mean()) if len(_close_s) >= 60 else None
                # 乖離率(%):股價相對均線的延伸程度
                _bias20 = (_latest - _ma20) / _ma20 * 100 if _ma20 > 0 else 0.0
                _bias60 = ((_latest - _ma60) / _ma60 * 100) if (_ma60 and _ma60 > 0) else None
                _hot_bias = (_bias20 > BIAS20_HOT) or (_bias60 is not None and _bias60 > BIAS60_HOT)
                if len(_close_s) < 6:
                    _verdict_quick = "觀察"
                else:
                    _gain5 = (_latest / float(_close_s.iloc[-6]) - 1) * 100
                    _k_list, _ = _calc_kd_series(_hist_quick[_high_c], _hist_quick[_low_c], _close_s)
                    _k_last = next((k for k in reversed(_k_list) if k is not None), None) if _k_list else None
                    # 「站上季線」條件:有 MA60 時要 latest>MA60;沒 MA60 則略過(避免新股直接被判觀察)
                    _above_ma60 = (_ma60 is None) or (_latest > _ma60)
                    _k_hot = _k_last is not None and _k_last > 80
                    if _k_hot or _gain5 > 10 or _hot_bias:
                        _verdict_quick = "拉回再進"
                        # 組過熱原因(讓使用者知道為何黃燈,不只一個顏色)
                        _rs = []
                        if _k_hot:           _rs.append(f"KD {_k_last:.0f} 過熱")
                        if _gain5 > 10:      _rs.append(f"5 日漲 {_gain5:+.0f}%")
                        if _bias20 > BIAS20_HOT:                    _rs.append(f"乖離月線 {_bias20:+.0f}%")
                        if _bias60 is not None and _bias60 > BIAS60_HOT: _rs.append(f"乖離季線 {_bias60:+.0f}%")
                        _verdict_reason = "追高風險:" + "、".join(_rs)
                    elif (_k_last is None or _k_last < 70) and _gain5 < 8 and _latest > _ma20 and _above_ma60:
                        _verdict_quick = "可進場"
                        _verdict_reason = f"乖離月線 {_bias20:+.0f}%" + (f"、季線 {_bias60:+.0f}%" if _bias60 is not None else "")
                    else:
                        _verdict_quick = "觀察"
            except Exception:
                pass

        if _verdict_quick:
            render_verdict_pill(_verdict_quick, _verdict_reason)

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

                # ── 🎨 MA 多頭/空頭排列狀態(實際 add_vrect 染色延後到 fig 建立後) ──
                _ma_state = pd.Series('flat', index=hist.index)
                _bull = (hist['MA5'] > hist['MA20']) & (hist['MA20'] > hist['MA60'])
                _bear = (hist['MA5'] < hist['MA20']) & (hist['MA20'] < hist['MA60'])
                _ma_state[_bull] = 'bull'
                _ma_state[_bear] = 'bear'
                _state_change = (_ma_state != _ma_state.shift(1)).cumsum()

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

                # ── 🎨 MA 多頭/空頭背景染色(台股慣例:紅=多頭、綠=空頭)──
                for _, _seg in hist.groupby(_state_change):
                    _s = _ma_state.loc[_seg.index[0]]
                    if _s == 'flat':
                        continue
                    _fill = "rgba(244,67,54,0.10)" if _s == 'bull' else "rgba(76,175,80,0.10)"
                    fig.add_vrect(x0=_seg['date'].iloc[0], x1=_seg['date'].iloc[-1],
                                  fillcolor=_fill, line_width=0, layer="below", row=1, col=1)
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
                defense_label = "🚨 防守(週MA20)" if timeframe == "週K" else "🚨 防守(MA20)"
                fig.add_hline(y=defense_y, line_dash="dash", line_color="red", annotation_text=defense_label, annotation_position="bottom right", row=1, col=1)

                # ── ⭐ 爆量警示星(紅 K):量 > 5 期均量 × 2 ──
                _vol5 = hist[vol_col].rolling(5).mean()
                _big_vol = (hist[vol_col] > _vol5 * 2) & _vol5.notna()
                _big_red = _big_vol & (hist['close'] >= hist['open'])
                if _big_red.any():
                    _bv = hist[_big_red]
                    fig.add_trace(go.Scatter(
                        x=_bv['date'], y=_bv['max'] * 1.015,
                        mode='markers+text', text='⭐', textposition='top center',
                        textfont=dict(size=14, color='gold'),
                        marker=dict(size=1, color='rgba(0,0,0,0)'),
                        name='爆量紅K',
                        hovertext=[f"爆量紅K (量 {int(v/1000)} 張, 約 5 期均量 {r:.1f}x)"
                                   for v, r in zip(_bv[vol_col], _bv[vol_col]/_vol5[_big_red])],
                        hoverinfo='text',
                    ), row=1, col=1)

                # ── 🟡 出貨警訊:爆量黑 K(量 > 5 期均量 × 2 且收盤 < 開盤 × 0.97) ──
                _big_black = _big_vol & (hist['close'] < hist['open'] * 0.97)
                if _big_black.any():
                    _bb = hist[_big_black]
                    fig.add_trace(go.Scatter(
                        x=_bb['date'], y=_bb['max'] * 1.015,
                        mode='markers+text', text='🟡', textposition='top center',
                        textfont=dict(size=14),
                        marker=dict(size=1, color='rgba(0,0,0,0)'),
                        name='出貨警訊',
                        hovertext='⚠️ 爆量黑 K(主力出貨警訊)',
                        hoverinfo='text',
                    ), row=1, col=1)

                # ── 📈 站上季線首日:今日 close > MA60 且 昨日 close ≤ MA60(中長線轉折) ──
                if 'MA60' in hist.columns and hist['MA60'].notna().any():
                    _above_ma60_now  = hist['close'] > hist['MA60']
                    _above_ma60_prev = hist['close'].shift(1) <= hist['MA60'].shift(1)
                    _cross_up_ma60 = _above_ma60_now & _above_ma60_prev & hist['MA60'].notna() & hist['MA60'].shift(1).notna()
                    if _cross_up_ma60.any():
                        _cu = hist[_cross_up_ma60]
                        fig.add_trace(go.Scatter(
                            x=_cu['date'], y=_cu['max'] * 1.025,
                            mode='text', text='📈', textfont=dict(size=14),
                            name='站上季線', hovertext='📈 站上季線首日(中長線轉強)',
                            hoverinfo='text',
                        ), row=1, col=1)

                    # ── 📉 跌破季線首日:今日 close < MA60 且 昨日 close ≥ MA60(中長線轉弱) ──
                    _below_ma60_now  = hist['close'] < hist['MA60']
                    _below_ma60_prev = hist['close'].shift(1) >= hist['MA60'].shift(1)
                    _cross_dn_ma60 = _below_ma60_now & _below_ma60_prev & hist['MA60'].notna() & hist['MA60'].shift(1).notna()
                    if _cross_dn_ma60.any():
                        _cd = hist[_cross_dn_ma60]
                        fig.add_trace(go.Scatter(
                            x=_cd['date'], y=_cd['max'] * 1.025,
                            mode='text', text='📉', textfont=dict(size=14),
                            name='跌破季線', hovertext='📉 跌破季線首日(中長線轉弱、必看出場)',
                            hoverinfo='text',
                        ), row=1, col=1)

                # ── 🔻 KD 高檔死叉:K 從上穿過 D 且 K > 60(出場警示) ──
                if 'K' in hist.columns and 'D' in hist.columns:
                    _kd_death = ((hist['K'] < hist['D']) & (hist['K'].shift(1) >= hist['D'].shift(1)) & (hist['K'] > 60))
                    if _kd_death.any():
                        _kd_d = hist[_kd_death]
                        fig.add_trace(go.Scatter(
                            x=_kd_d['date'], y=_kd_d['max'] * 1.04,
                            mode='markers', marker=dict(symbol='triangle-down', color='red', size=12,
                                                       line=dict(width=1, color='darkred')),
                            name='KD死叉(出場警示)',
                            hovertext='⚠️ KD 高檔死叉(短線過熱、考慮減碼)',
                            hoverinfo='text',
                        ), row=1, col=1)

                # ── 📐 跳空缺口:今日 low > 昨日 high(向上跳空) 或 今日 high < 昨日 low ──
                _prev_high = hist['max'].shift(1)
                _prev_low  = hist['min'].shift(1)
                _gap_up   = hist['min'] > _prev_high
                _gap_down = hist['max'] < _prev_low
                for _, _row in hist[_gap_up].iterrows():
                    fig.add_shape(type="rect",
                        x0=_row['date'] - pd.Timedelta(hours=12), x1=_row['date'] + pd.Timedelta(hours=12),
                        y0=_prev_high.loc[_row.name], y1=_row['min'],
                        fillcolor="rgba(255,99,71,0.25)", line=dict(width=0), row=1, col=1)
                for _, _row in hist[_gap_down].iterrows():
                    fig.add_shape(type="rect",
                        x0=_row['date'] - pd.Timedelta(hours=12), x1=_row['date'] + pd.Timedelta(hours=12),
                        y0=_row['max'], y1=_prev_low.loc[_row.name],
                        fillcolor="rgba(60,179,113,0.25)", line=dict(width=0), row=1, col=1)

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

                # ── 📖 線圖訊號解說(摺疊預設關,需要時才打開查看)──
                with st.expander("📖 線圖訊號解說", expanded=False):
                    st.markdown("""
**🎨 背景染色**(MA 排列狀態)
- 🔴 **淡紅底**：MA5 > MA20 > MA60(多頭排列、強勢區)
- 🟢 **淡綠底**:MA5 < MA20 < MA60(空頭排列、弱勢區)
- ⚪ **無底色**:均線糾結中(無方向)

**📈 K 線上的訊號 icon**(都是「事件點」,出現即代表那天觸發)
| icon | 意義 | 用途 |
|---|---|---|
| ⭐ | 爆量紅 K(量 > 5 期均量 × 2 且收紅) | 主力進場參考 |
| 🟡 | 爆量黑 K(量 ×2 且跌 ≥ 3%) | **主力出貨警訊** |
| 🔺 | KD 低檔金叉(K 上穿 D 且 K < 設定門檻) | **進場時機點** |
| 🔻 | KD 高檔死叉(K 下穿 D 且 K > 60) | **出場時機點** |
| 📈 | 站上季線首日(close 上穿 MA60) | 中長線轉強 |
| 📉 | 跌破季線首日(close 下穿 MA60) | **中長線轉弱,必看** |

**🎯 線段與區塊**
- 🟠 **橘色點線**:60 日(週)壓力線 — 過去 60 期最高,突破 = 站上壓力
- 🔴 **紅色虛線**:防守線(MA20 × 0.98)— 跌破即考慮停損
- 🟠 **橘色 MA5** / 🟣 **紫色 MA20** / 🟢 **綠色 MA60**:三條移動均線
- ⚪ **灰色點線**:大盤 RS(加權指數相對表現)
- 🔴 紅色半透明色塊:**向上跳空缺口**(今日 low > 昨日 high)
- 🟢 綠色半透明色塊:**向下跳空缺口**(今日 high < 昨日 low)

**💡 進場節奏 pill**(個股名稱下方)
- 🟢 可進場:K<70、5 日漲幅<8%、站上 MA20+MA60、乖離不過大
- 🟡 拉回再進:K>80 或 5 日漲幅>10% **或乖離過大**(MA20>12% / 季線>20%)→ 追高風險,pill 會標出是哪個原因
- 🔵 觀察:其他(訊號模糊、跌破均線、資料不足)

> **乖離煞車(避免追高)**:股價慢慢爬、離均線太遠時,即使 K/漲幅還沒過熱也會轉黃燈。乖離率=(股價−均線)/均線;個股波動比大盤大,故門檻(MA20>12%、季線>20%)比大盤位階的 ±8% 寬。
                    """)
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

                        # 分級語意:讓 AI 知道分數高低的實際意義
                        _score = row_data['總分']
                        if _score >= 8:
                            _score_tier = "🔥 頂級(本系統實務最高分)"
                        elif _score == 7:
                            _score_tier = "✅ 合格"
                        elif _score == 6:
                            _score_tier = "⚠️ 邊緣(大盤資料缺失自動降標)"
                        else:
                            _score_tier = ""

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
                            綜合量化總分:{row_data['總分']} / 10 分  ({_score_tier})
                            註:本系統雖以 10 分為滿分,但實務最高僅見 8 分(9~10 分要求法人雙買+大戶散戶共振+RS強+月營收YoY同時成立,極罕見),故 8 分即冠軍級訊號。
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
