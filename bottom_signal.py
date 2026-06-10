# -*- coding: utf-8 -*-
"""台股止跌自動判讀(19 項檢查 × 四級分級)

依據《止跌自動判讀_資料來源對照.md》規格實作:
- 17 項自動判定 + 2 項人工勾選(利空鈍化 / 利空解除)
- 分級:高度恐慌 🔴 → 剛降溫 🟡 → 在打底 🟠 → 止跌確認 🟢
- 閘門:VIXTWN(台指選擇權波動率指數)跌破 40;閘門未開一律「高度恐慌」

資料來源(全部收盤後可得):
  VIXTWN 日收盤   — TAIFEX /file/taifex/Dailydownload/vix/log2data/YYYYMMnew.txt
  VIXTWN 當日高   — TAIFEX /cht/7/getVixData?filesname=YYYYMMDD(分鐘檔)
  台指期近月收盤  — TAIFEX /cht/3/futDataDown(CSV,一般時段)
  P/C ratio       — TAIFEX /cht/3/pcRatioDown(CSV)
  成交金額/加權   — TWSE rwd FMTQIK(可帶月份抓歷史)
  外資現貨買賣超  — TWSE rwd BFI82U(可帶日期抓歷史)
  加權/2330 OHLC  — yfinance ^TWII / 2330.TW(GHA 上若失敗,之後補 TWSE 備援)
  外資期貨未平倉  — 沿用 market_sentiment._fetch_taifex_institutional
                    + fi_futures_history.json(需累積 ≥ 2 日才能判「回補」)

設計原則(同 market_sentiment.py):
- 任一資料源失敗 → 該項 ok=None(顯示 ❓ 資料不足),其他項照常判
- VIXTWN 是唯一閘門:月檔 + 分鐘檔雙端點互為備援,雙雙失敗時
  result["alerts"] 會帶告警訊息,呼叫端應推播通知,不可靜默
- 所有門檻集中在 CFG,方便之後用歷史資料回測調整
"""
import json
import re
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ══════════════════════════════════════════════════════════════════════
# 可調參數(回測後再校準)
# ══════════════════════════════════════════════════════════════════════
CFG = {
    "gate_level":        40.0,   # 01 閘門
    "fall_level":        38.0,   # 02 續降(嚴格版可改 35)
    "fall_lookback":     3,      # 02 過去 N 日未站回 40
    "spread_avg_days":   5,      # 03 與美 VIX 差距 vs 近 N 日均
    "spike_high_days":   10,     # 04 爆衝:今日高 > 近 N 日高
    "spike_drop_pct":    8.0,    # 04 收盤較當日高回落 > X%
    "no_low_days":       10,     # 05/16 未破前 N 日低
    "pivot_window":      3,      # 06 轉折低偵測視窗
    "hold_days":         2,      # 08 連 N 天守住
    "vol_avg_days":      20,     # 09 均量天數
    "vol_spike_mult":    1.5,    # 09 爆量倍數
    "calm_drop_pct":     1.0,    # 10 量縮止穩:跌幅 < 1%
    "basis_days":        2,      # 12 正價差連 N 日
    "fs_avg_days":       5,      # 13 外資賣超 vs 近 N 日均
    "pc_extreme_days":   10,     # 15 P/C 近 N 日極端
    "pc_extreme_ratio":  0.95,   # 15 昨日 ≥ 近 N 日最高 × 此比例 才算「曾觸極端」
    "tsmc_ma":           5,      # 17 台積站上 MA N
    "intl_days":         2,      # 18/19 美債殖利率/美元指數 與 N 日前比
    "level3_min_ok":     11,     # 止跌確認:總成立數門檻(全 21 項)
}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
}

HISTORY_FILE = "bottom_signal_history.json"
MANUAL_FILE = "bottom_manual_flags.json"
_HISTORY_KEEP = 120


# ══════════════════════════════════════════════════════════════════════
# 共用小工具
# ══════════════════════════════════════════════════════════════════════
def _get(url, **kw):
    kw.setdefault("headers", _HEADERS)
    kw.setdefault("timeout", 20)
    return requests.get(url, **kw)


def _post(url, data, **kw):
    kw.setdefault("headers", _HEADERS)
    kw.setdefault("timeout", 20)
    return requests.post(url, data=data, **kw)


def _yf_series(ticker: str, period: str = "60d", max_retries: int = 2):
    """yfinance OHLC,回單層欄位 DataFrame(open/high/low/close)或 None。"""
    import yfinance as yf
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, period=period, auto_adjust=True,
                             progress=False, threads=False)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] for c in df.columns]
                df = df.rename(columns=str.lower)
                return df[["open", "high", "low", "close"]].dropna()
        except Exception as e:
            print(f"   ⚠ yfinance {ticker} 失敗: {str(e)[:80]}")
            if attempt < max_retries - 1:
                time.sleep(20)
    return None


def _roc_to_date(s: str):
    """民國日期('115/06/09' 或 '1150609')→ datetime.date。"""
    s = s.strip().replace("/", "")
    if len(s) == 7:
        return datetime(int(s[:3]) + 1911, int(s[3:5]), int(s[5:7])).date()
    return None


# ══════════════════════════════════════════════════════════════════════
# 資料抓取(每個函式失敗回 None,不丟例外)
# ══════════════════════════════════════════════════════════════════════
def fetch_vixtwn_daily(months: int = 3) -> "pd.Series | None":
    """VIXTWN 每日收盤(主來源:期交所月檔,tab 分隔 big5)。"""
    rows = {}
    today = datetime.now()
    for k in range(months):
        # 往回推 k 個月
        y, m = today.year, today.month - k
        while m <= 0:
            y, m = y - 1, m + 12
        url = (f"https://www.taifex.com.tw/file/taifex/Dailydownload/"
               f"vix/log2data/{y}{m:02d}new.txt")
        try:
            r = _get(url)
            if r.status_code != 200:
                continue
            for ln in r.content.decode("big5", errors="replace").splitlines()[2:]:
                parts = ln.split()
                if len(parts) >= 3 and re.fullmatch(r"\d{8}", parts[0]):
                    val = float(parts[2])
                    if val > 3:          # 過濾壞 tick
                        rows[datetime.strptime(parts[0], "%Y%m%d").date()] = val
        except Exception as e:
            print(f"   ⚠ VIXTWN 月檔 {y}{m:02d} 失敗: {str(e)[:80]}")
    if not rows:
        return None
    return pd.Series(rows).sort_index()


def fetch_vixtwn_today_from_minute() -> "dict | None":
    """VIXTWN 備援/當日高:期交所分鐘檔。回 {date, high, last}。"""
    try:
        d = datetime.now().strftime("%Y%m%d")
        r = _get(f"https://www.taifex.com.tw/cht/7/getVixData?filesname={d}")
        if r.status_code != 200:
            return None
        vals, date_seen = [], None
        for ln in r.content.decode("big5", errors="replace").splitlines()[2:]:
            parts = ln.split()
            if len(parts) >= 3 and re.fullmatch(r"\d{8}", parts[0]):
                v = float(parts[2])
                if v > 3:                # 過濾開盤前的壞 tick(出現過 1.0)
                    vals.append(v)
                    date_seen = parts[0]
        if not vals:
            return None
        return {"date": datetime.strptime(date_seen, "%Y%m%d").date(),
                "high": max(vals), "last": vals[-1]}
    except Exception as e:
        print(f"   ⚠ VIXTWN 分鐘檔失敗: {str(e)[:80]}")
        return None


def fetch_tx_futures_close(days: int = 10) -> "pd.Series | None":
    """台指期近月收盤(一般時段)。回 Series(date → close)。"""
    try:
        end = datetime.now()
        start = end - timedelta(days=days + 8)
        form = {"down_type": "1", "commodity_id": "TX",
                "queryStartDate": start.strftime("%Y/%m/%d"),
                "queryEndDate": end.strftime("%Y/%m/%d")}
        r = _post("https://www.taifex.com.tw/cht/3/futDataDown", form)
        if r.status_code != 200:
            return None
        # TAIFEX CSV 每行結尾多一個逗號,需 index_col=False 防止欄位錯位
        df = pd.read_csv(StringIO(r.content.decode("big5", errors="replace")),
                         index_col=False)
        df.columns = [c.strip() for c in df.columns]
        df["契約"] = df["契約"].astype(str).str.strip()
        df["交易時段"] = df["交易時段"].astype(str).str.strip()
        df = df[(df["契約"] == "TX") & (df["交易時段"] == "一般")].copy()
        df["到期月份(週別)"] = df["到期月份(週別)"].astype(str).str.strip()
        # 只留純月份契約(排除週選 202606W2 之類),取每日最近月
        df = df[df["到期月份(週別)"].str.fullmatch(r"\d{6}")]
        df["date"] = pd.to_datetime(df["交易日期"], format="%Y/%m/%d").dt.date
        df["close"] = pd.to_numeric(df["收盤價"], errors="coerce")
        df = df.dropna(subset=["close"])
        near = df.sort_values("到期月份(週別)").groupby("date").first()
        return near["close"].sort_index()
    except Exception as e:
        print(f"   ⚠ 台指期行情失敗: {str(e)[:80]}")
        return None


def fetch_pc_ratio(days: int = 20) -> "pd.Series | None":
    """選擇權 P/C 未平倉比(%)。回 Series(date → ratio)。

    注意:TAIFEX 查詢區間上限 30 天,固定抓 28 天(約 19 個交易日)。
    """
    try:
        end = datetime.now()
        start = end - timedelta(days=28)
        form = {"queryStartDate": start.strftime("%Y/%m/%d"),
                "queryEndDate": end.strftime("%Y/%m/%d")}
        r = _post("https://www.taifex.com.tw/cht/3/pcRatioDown", form)
        if r.status_code != 200:
            return None
        # 同 futDataDown:行尾多逗號,index_col=False 防錯位
        df = pd.read_csv(StringIO(r.content.decode("big5", errors="replace")),
                         index_col=False)
        df.columns = [c.strip() for c in df.columns]
        df["date"] = pd.to_datetime(df["日期"], format="%Y/%m/%d").dt.date
        col = "買賣權未平倉量比率%"
        df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=[col]).set_index("date")[col].sort_index()
    except Exception as e:
        print(f"   ⚠ P/C ratio 失敗: {str(e)[:80]}")
        return None


def fetch_market_turnover(months: int = 2) -> "pd.DataFrame | None":
    """大盤成交金額 + 加權收盤/漲跌(TWSE FMTQIK,逐月)。"""
    frames = []
    today = datetime.now()
    for k in range(months):
        y, m = today.year, today.month - k
        while m <= 0:
            y, m = y - 1, m + 12
        url = (f"https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK"
               f"?date={y}{m:02d}01&response=json")
        try:
            r = _get(url)
            j = r.json()
            if j.get("stat") != "OK" or not j.get("data"):
                continue
            for row in j["data"]:
                d = _roc_to_date(row[0])
                if d is None:
                    continue
                frames.append({
                    "date": d,
                    "value": float(str(row[2]).replace(",", "")),       # 成交金額
                    "taiex": float(str(row[4]).replace(",", "")),       # 收盤指數
                    "change": float(str(row[5]).replace(",", "")),      # 漲跌點數
                })
            time.sleep(1.2)   # TWSE 有流量限制,放慢
        except Exception as e:
            print(f"   ⚠ FMTQIK {y}{m:02d} 失敗: {str(e)[:80]}")
    if not frames:
        return None
    df = pd.DataFrame(frames).drop_duplicates("date").set_index("date").sort_index()
    return df


def fetch_foreign_spot_net(days: int = 6) -> "pd.Series | None":
    """外資現貨買賣差額(元),近 N 個交易日(TWSE BFI82U 逐日)。"""
    rows = {}
    d = datetime.now()
    tried = 0
    while len(rows) < days and tried < days * 3:
        if d.weekday() < 5:
            url = (f"https://www.twse.com.tw/rwd/zh/fund/BFI82U"
                   f"?dayDate={d.strftime('%Y%m%d')}&type=day&response=json")
            try:
                j = _get(url).json()
                if j.get("stat") == "OK" and j.get("data"):
                    net = 0.0
                    for row in j["data"]:
                        if "外資" in str(row[0]):
                            net += float(str(row[3]).replace(",", ""))
                    rows[d.date()] = net
                time.sleep(1.2)
            except Exception as e:
                print(f"   ⚠ BFI82U {d:%Y%m%d} 失敗: {str(e)[:80]}")
            tried += 1
        d -= timedelta(days=1)
    if not rows:
        return None
    return pd.Series(rows).sort_index()


def fetch_index_ohlc_twse(months: int = 5) -> "pd.DataFrame | None":
    """加權指數每日 OHLC(TWSE MI_5MINS_HIST,逐月)。

    為什麼不用 yfinance 當主來源:GHA 主機抓 ^TWII 常失敗,
    且 yfinance 偶爾缺交易日(實測缺過 2026-06-09)。官方資料完整。
    """
    rows = []
    today = datetime.now()
    for k in range(months):
        y, m = today.year, today.month - k
        while m <= 0:
            y, m = y - 1, m + 12
        url = (f"https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST"
               f"?date={y}{m:02d}01&response=json")
        try:
            j = _get(url).json()
            if j.get("stat") == "OK" and j.get("data"):
                for r in j["data"]:
                    d = _roc_to_date(r[0])
                    if d is None:
                        continue
                    try:
                        rows.append({
                            "date": d,
                            "open":  float(str(r[1]).replace(",", "")),
                            "high":  float(str(r[2]).replace(",", "")),
                            "low":   float(str(r[3]).replace(",", "")),
                            "close": float(str(r[4]).replace(",", "")),
                        })
                    except (ValueError, IndexError):
                        continue
            time.sleep(1.2)
        except Exception as e:
            print(f"   ⚠ MI_5MINS_HIST {y}{m:02d} 失敗: {str(e)[:80]}")
    if not rows:
        return None
    return (pd.DataFrame(rows).drop_duplicates("date")
            .set_index("date").sort_index())


def fetch_stock_ohlc_twse(stock_no: str, months: int = 3) -> "pd.DataFrame | None":
    """個股每日 OHLC(TWSE STOCK_DAY,逐月)。停牌日價格為 '--',自動略過。"""
    rows = []
    today = datetime.now()
    for k in range(months):
        y, m = today.year, today.month - k
        while m <= 0:
            y, m = y - 1, m + 12
        url = (f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
               f"?date={y}{m:02d}01&stockNo={stock_no}&response=json")
        try:
            j = _get(url).json()
            if j.get("stat") == "OK" and j.get("data"):
                for r in j["data"]:
                    d = _roc_to_date(r[0])
                    if d is None:
                        continue
                    try:
                        rows.append({
                            "date": d,
                            "open":  float(str(r[3]).replace(",", "")),
                            "high":  float(str(r[4]).replace(",", "")),
                            "low":   float(str(r[5]).replace(",", "")),
                            "close": float(str(r[6]).replace(",", "")),
                        })
                    except (ValueError, IndexError):
                        continue    # '--' 等非數字(停牌)略過
            time.sleep(1.2)
        except Exception as e:
            print(f"   ⚠ STOCK_DAY {stock_no} {y}{m:02d} 失敗: {str(e)[:80]}")
    if not rows:
        return None
    return (pd.DataFrame(rows).drop_duplicates("date")
            .set_index("date").sort_index())


def fetch_fi_futures_yesterday_today(cache_dir):
    """外資期貨淨未平倉:今日(現抓)+ 昨日(讀 fi_futures_history.json)。

    回 (today_net, yesterday_net);任一拿不到回 None。
    """
    today_net = yesterday_net = None
    try:
        from market_sentiment import _fetch_taifex_institutional, _load_fi_history
        df = _fetch_taifex_institutional()
        if df is not None and not df.empty:
            mask = (df["product"] == "臺股期貨") & (df["trader"] == "外資")
            row = df[mask]
            if not row.empty:
                today_net = int(row["oi_net_vol"].iloc[0])
        history = _load_fi_history(cache_dir)
        today_str = datetime.now().strftime("%Y-%m-%d")
        past = [h["net_vol"] for h in history if h.get("date") != today_str]
        if past:
            yesterday_net = int(past[-1])
    except Exception as e:
        print(f"   ⚠ 外資期貨失敗: {str(e)[:80]}")
    return today_net, yesterday_net


# ══════════════════════════════════════════════════════════════════════
# 判定核心
# ══════════════════════════════════════════════════════════════════════
def _item(key, group, name, ok, value=""):
    return {"key": key, "group": group, "name": name, "ok": ok,
            "value": str(value), "manual": False}


def _find_pivot_lows(lows: pd.Series, window: int) -> list:
    """區域低點:比左右 window 日都低。回 [(date, low), ...]。"""
    vals = lows.values
    pivots = []
    for i in range(window, len(vals) - window):
        seg = vals[i - window:i + window + 1]
        if vals[i] == seg.min() and (seg > vals[i]).sum() >= window:
            pivots.append((lows.index[i], float(vals[i])))
    return pivots


def run_all_checks(cache_dir=None, manual_flags=None) -> dict:
    """抓資料 → 逐項判定 → 分級。回傳 dict(items / level / gate / alerts)。"""
    c = CFG
    alerts = []
    if manual_flags is None:
        manual_flags = load_manual_flags(cache_dir)

    # ── 抓資料 ────────────────────────────────────────────────
    print("📥 抓 VIXTWN…")
    vix_tw = fetch_vixtwn_daily()
    vix_intraday = fetch_vixtwn_today_from_minute()
    if vix_tw is None and vix_intraday is not None:
        # 月檔掛了,至少用分鐘檔湊出今日值(備援)
        vix_tw = pd.Series({vix_intraday["date"]: vix_intraday["last"]})
        alerts.append("VIXTWN 月檔抓取失敗,改用分鐘檔備援(僅當日值,歷史比較項目會缺)")
    if vix_tw is None:
        alerts.append("🚨 VIXTWN 兩個來源都抓不到!閘門無法判定,請檢查期交所網站")

    print("📥 抓美股 VIX / 美債 / 美元(yfinance)…")
    us_vix = _yf_series("^VIX", period="30d")
    us10y = _yf_series("^TNX", period="15d")       # 美國 10 年期公債殖利率(%)
    dxy = _yf_series("DX-Y.NYB", period="15d")     # 美元指數 DXY

    print("📥 抓加權 / 台積電 OHLC(TWSE 官方,yfinance 備援)…")
    twii = fetch_index_ohlc_twse()
    if twii is None:
        twii = _yf_series("^TWII", period="120d")
        if twii is not None:
            alerts.append("加權 OHLC 改用 yfinance 備援(TWSE 失敗,留意缺日)")
    tsmc = fetch_stock_ohlc_twse("2330")
    if tsmc is None:
        tsmc = _yf_series("2330.TW", period="60d")
        if tsmc is not None:
            alerts.append("2330 OHLC 改用 yfinance 備援(TWSE 失敗)")

    print("📥 抓期交所(台指期 / P/C)…")
    tx_close = fetch_tx_futures_close()
    pc = fetch_pc_ratio()

    print("📥 抓證交所(成交金額 / 外資現貨)…")
    turnover = fetch_market_turnover()
    f_spot = fetch_foreign_spot_net()

    print("📥 抓外資期貨…")
    fi_today, fi_yest = fetch_fi_futures_yesterday_today(cache_dir)

    items = []

    # ── 01 恐慌指數(閘門) ────────────────────────────────────
    g = "01 恐慌指數"
    if vix_tw is not None and len(vix_tw) >= 2:
        v, prev = float(vix_tw.iloc[-1]), float(vix_tw.iloc[-2])
        recent5_max = float(vix_tw.tail(6).iloc[:-1].max()) if len(vix_tw) >= 6 else prev
        ok = (v < c["gate_level"]) and (v < prev) and (v < recent5_max)
        items.append(_item("gate", g, f"跌破 {c['gate_level']:.0f} ★閘門", ok,
                           f"今 {v:.1f} / 昨 {prev:.1f}"))
    elif vix_tw is not None and len(vix_tw) == 1:
        v = float(vix_tw.iloc[-1])
        items.append(_item("gate", g, f"跌破 {c['gate_level']:.0f} ★閘門",
                           v < c["gate_level"], f"今 {v:.1f}(無昨日值,僅比 40)"))
    else:
        items.append(_item("gate", g, f"跌破 {c['gate_level']:.0f} ★閘門", None, "資料缺"))

    if vix_tw is not None and len(vix_tw) >= c["fall_lookback"] + 1:
        v = float(vix_tw.iloc[-1])
        back = vix_tw.tail(c["fall_lookback"])
        ok = (v < c["fall_level"]) and bool((back < c["gate_level"]).all())
        items.append(_item("vix_fall", g, f"續降 {c['fall_level']:.0f}", ok, f"今 {v:.1f}"))
    else:
        items.append(_item("vix_fall", g, f"續降 {c['fall_level']:.0f}", None, "資料缺"))

    if vix_tw is not None and us_vix is not None and len(vix_tw) >= c["spread_avg_days"] + 1:
        uv = us_vix["close"]
        uv.index = [d.date() for d in uv.index]
        spread = (vix_tw - uv).dropna()
        if len(spread) >= c["spread_avg_days"] + 1:
            today_sp = float(spread.iloc[-1])
            avg_sp = float(spread.iloc[-(c["spread_avg_days"] + 1):-1].mean())
            items.append(_item("vix_spread", g, "與美VIX差距收斂", today_sp < avg_sp,
                               f"差 {today_sp:.1f} / 5日均 {avg_sp:.1f}"))
        else:
            items.append(_item("vix_spread", g, "與美VIX差距收斂", None, "重疊日不足"))
    else:
        items.append(_item("vix_spread", g, "與美VIX差距收斂", None, "資料缺"))

    if vix_tw is not None and vix_intraday is not None and len(vix_tw) >= c["spike_high_days"]:
        hi = vix_intraday["high"]
        close_v = float(vix_tw.iloc[-1])
        past_hi = float(vix_tw.tail(c["spike_high_days"] + 1).iloc[:-1].max())
        drop_pct = (hi - close_v) / hi * 100 if hi > 0 else 0
        ok = (hi > past_hi) and (drop_pct > c["spike_drop_pct"])
        items.append(_item("vix_spike", g, "爆衝後急殺", ok,
                           f"日高 {hi:.1f} 收 {close_v:.1f}(回落 {drop_pct:.1f}%)"))
    else:
        items.append(_item("vix_spike", g, "爆衝後急殺", None, "資料缺"))

    # ── 02 股價走勢(加權) ────────────────────────────────────
    g = "02 股價走勢"
    n = c["no_low_days"]
    if twii is not None and len(twii) >= n + 1:
        today_low = float(twii["low"].iloc[-1])
        prior_min = float(twii["low"].iloc[-(n + 1):-1].min())
        items.append(_item("px_no_low", g, "未破前低", today_low >= prior_min,
                           f"今低 {today_low:,.0f} / 前{n}日低 {prior_min:,.0f}"))

        pivots = _find_pivot_lows(twii["low"].tail(60), c["pivot_window"])
        if len(pivots) >= 2:
            (_, p1), (_, p2) = pivots[-2], pivots[-1]
            items.append(_item("px_higher_low", g, "低點墊高", p2 > p1,
                               f"前低 {p1:,.0f} → 新低 {p2:,.0f}"))
        else:
            items.append(_item("px_higher_low", g, "低點墊高", None, "轉折點不足"))

        o, h, l, cl = (float(twii[k].iloc[-1]) for k in ("open", "high", "low", "close"))
        prev_o = float(twii["open"].iloc[-2])
        rng = max(h - l, 1e-9)
        strong = (cl > o) and (((cl - l) / rng > 0.5) or (cl > prev_o))
        items.append(_item("px_strong_k", g, "強K守收盤", strong,
                           f"收 {cl:,.0f}({'紅' if cl > o else '黑'}K,下影 {(cl - l) / rng:.0%})"))

        hd = c["hold_days"]
        base_low = float(twii["low"].iloc[-(hd + n):-hd].min())
        seg = twii.iloc[-hd:]
        held = bool((seg["low"] > base_low).all() and (seg["close"] > base_low).all())
        items.append(_item("px_hold", g, f"連{hd}天守住", held,
                           f"守 {base_low:,.0f}"))
    else:
        for k, nm in [("px_no_low", "未破前低"), ("px_higher_low", "低點墊高"),
                      ("px_strong_k", "強K守收盤"), ("px_hold", f"連{c['hold_days']}天守住")]:
            items.append(_item(k, g, nm, None, "資料缺"))

    # ── 03 成交量 ─────────────────────────────────────────────
    g = "03 成交量"
    if turnover is not None and len(turnover) >= c["vol_avg_days"] + 1:
        t = turnover
        val, prev_val = float(t["value"].iloc[-1]), float(t["value"].iloc[-2])
        avg20 = float(t["value"].iloc[-(c["vol_avg_days"] + 1):-1].mean())
        chg = float(t["change"].iloc[-1])
        taiex = float(t["taiex"].iloc[-1])
        chg_pct = chg / (taiex - chg) * 100 if taiex != chg else 0.0

        items.append(_item("vol_spike_down", g, "爆量下跌",
                           (val > avg20 * c["vol_spike_mult"]) and (chg < 0),
                           f"量 {val / 1e12:.2f} 兆({val / avg20:.2f}×均) {chg_pct:+.1f}%"))
        items.append(_item("vol_shrink", g, "量縮止穩",
                           (val < prev_val) and (chg_pct > -c["calm_drop_pct"]),
                           f"量{'縮' if val < prev_val else '增'} {chg_pct:+.1f}%"))
        items.append(_item("vol_up", g, "帶量上漲",
                           (val > prev_val) and (chg > 0),
                           f"量{'增' if val > prev_val else '縮'} {chg_pct:+.1f}%"))
    else:
        for k, nm in [("vol_spike_down", "爆量下跌"), ("vol_shrink", "量縮止穩"),
                      ("vol_up", "帶量上漲")]:
            items.append(_item(k, g, nm, None, "資料缺"))

    # ── 04 期貨與大戶 ─────────────────────────────────────────
    g = "04 期貨與大戶"
    if tx_close is not None and turnover is not None:
        tw_close = turnover["taiex"]
        basis = (tx_close - tw_close).dropna()
        if len(basis) >= c["basis_days"]:
            recent = basis.tail(c["basis_days"])
            ok = bool((recent > 0).all())
            items.append(_item("fut_basis", g, "逆價差→正價差站穩", ok,
                               f"價差 {float(basis.iloc[-1]):+,.0f}"))
        else:
            items.append(_item("fut_basis", g, "逆價差→正價差站穩", None, "重疊日不足"))
    else:
        items.append(_item("fut_basis", g, "逆價差→正價差站穩", None, "資料缺"))

    if f_spot is not None and len(f_spot) >= c["fs_avg_days"] + 1:
        net = float(f_spot.iloc[-1])
        past = f_spot.iloc[-(c["fs_avg_days"] + 1):-1]
        avg_sell = float((-past[past < 0]).mean()) if (past < 0).any() else 0.0
        ok = (net >= 0) or (avg_sell > 0 and -net < avg_sell)
        items.append(_item("f_spot", g, "外資現貨賣超收斂", ok,
                           f"今 {net / 1e8:+,.0f} 億 / 近5日均賣 {avg_sell / 1e8:,.0f} 億"))
    else:
        items.append(_item("f_spot", g, "外資現貨賣超收斂", None, "資料缺"))

    if fi_today is not None and fi_yest is not None:
        ok = (fi_today > fi_yest) or (fi_today >= 0)
        items.append(_item("fi_cover", g, "外資期貨空單回補", ok,
                           f"淨額 {fi_yest:+,} → {fi_today:+,} 口"))
    elif fi_today is not None:
        items.append(_item("fi_cover", g, "外資期貨空單回補",
                           True if fi_today >= 0 else None,
                           f"今 {fi_today:+,} 口(無昨日紀錄,需累積)"))
    else:
        items.append(_item("fi_cover", g, "外資期貨空單回補", None, "資料缺"))

    if pc is not None and len(pc) >= c["pc_extreme_days"] + 1:
        today_pc, yest_pc = float(pc.iloc[-1]), float(pc.iloc[-2])
        window_max = float(pc.iloc[-(c["pc_extreme_days"] + 1):-1].max())
        touched = yest_pc >= window_max * c["pc_extreme_ratio"]
        items.append(_item("pc_falloff", g, "P/C比從極端回落",
                           (today_pc < yest_pc) and touched,
                           f"{yest_pc:.0f}% → {today_pc:.0f}%"))
    else:
        items.append(_item("pc_falloff", g, "P/C比從極端回落", None, "資料缺"))

    # ── 05 權值龍頭 台積電 ───────────────────────────────────
    g = "05 台積電"
    if tsmc is not None and len(tsmc) >= max(n + 1, c["tsmc_ma"] + 1):
        t_low = float(tsmc["low"].iloc[-1])
        t_min = float(tsmc["low"].iloc[-(n + 1):-1].min())
        items.append(_item("tsmc_no_low", g, "台積止穩不破低", t_low >= t_min,
                           f"今低 {t_low:,.0f} / 前{n}日低 {t_min:,.0f}"))
        t_o, t_c = float(tsmc["open"].iloc[-1]), float(tsmc["close"].iloc[-1])
        ma = float(tsmc["close"].tail(c["tsmc_ma"]).mean())
        items.append(_item("tsmc_ma", g, f"台積翻紅站MA{c['tsmc_ma']}",
                           (t_c > t_o) and (t_c > ma),
                           f"收 {t_c:,.0f} / MA{c['tsmc_ma']} {ma:,.0f}"))
    else:
        items.append(_item("tsmc_no_low", g, "台積止穩不破低", None, "資料缺"))
        items.append(_item("tsmc_ma", g, f"台積翻紅站MA{c['tsmc_ma']}", None, "資料缺"))

    # ── 06 國際資金(輔助:風險偏好回溫的旁證,只加分不擋分級) ──
    g = "06 國際資金(輔助)"
    nd = c["intl_days"]
    if us10y is not None and len(us10y) >= nd + 1:
        y_now = float(us10y["close"].iloc[-1])
        y_ago = float(us10y["close"].iloc[-(nd + 1)])
        items.append(_item("us_10y", g, "美債殖利率回升", y_now > y_ago,
                           f"{y_ago:.2f}% → {y_now:.2f}%"))
    else:
        items.append(_item("us_10y", g, "美債殖利率回升", None, "資料缺"))

    if dxy is not None and len(dxy) >= nd + 1:
        d_now = float(dxy["close"].iloc[-1])
        d_ago = float(dxy["close"].iloc[-(nd + 1)])
        items.append(_item("dxy_fall", g, "美元指數回落", d_now < d_ago,
                           f"{d_ago:.1f} → {d_now:.1f}"))
    else:
        items.append(_item("dxy_fall", g, "美元指數回落", None, "資料缺"))

    # ── 07 利空消息(人工勾選) ───────────────────────────────
    g = "07 利空消息"
    for k, nm in [("news_dulled", "利空鈍化"), ("news_resolved", "利空解除")]:
        it = _item(k, g, nm, bool(manual_flags.get(k)), "人工勾選")
        it["manual"] = True
        items.append(it)

    # ── 分級 ─────────────────────────────────────────────────
    result = {
        "asof": str(vix_tw.index[-1]) if vix_tw is not None else
                datetime.now().strftime("%Y-%m-%d"),
        "items": items,
        "alerts": alerts,
        "manual_flags": manual_flags,
        "vixtwn": float(vix_tw.iloc[-1]) if vix_tw is not None else None,
        "fi_net_today": fi_today,    # 給排程腳本 persist_fi_history 用
    }
    result.update(compute_level(items))
    return result


LEVELS = [
    {"label": "高度恐慌", "icon": "🔴", "desc": "閘門未開(VIXTWN 未跌破 40)",
     "note": "恐慌指數還沒從高點降下來。先別猜底,股價就算盤中拉高也先當反彈看。"},
    {"label": "剛降溫",   "icon": "🟡", "desc": "閘門已開,等待打底訊號",
     "note": "恐慌剛開始退,但確認的訊號還不夠。多看幾天,別急著進場。"},
    {"label": "在打底",   "icon": "🟠", "desc": "K線守住 + 量能訊號出現",
     "note": "恐慌退了、又有強K守住、賣壓也差不多倒完——可以看成正在打底。"},
    {"label": "止跌確認", "icon": "🟢", "desc": "核心條件齊備,趨勢轉穩",
     "note": "該看的訊號大致到齊了(連續守住、台積止穩、壞消息退燒)。止跌站得比較穩。"},
]

# 核心三開關(止跌確認的必要條件;news_dulled 為人工勾選)
CORE_KEYS = ["vix_fall", "tsmc_no_low", "news_dulled"]

# 每一項的白話說明(UI 顯示用,文案沿用檢查表網頁)
EXPLAIN = {
    "gate": "VIXTWN 就是市場的「怕不怕」溫度計,愈高代表大家愈恐慌。要等它從高點明確往下掉、跌破 40,才代表大家開始沒那麼怕了。這是判斷止跌最重要的一關——沒過這關,股價就算盤中拉高也先當作只是反彈。",
    "vix_fall": "光跌破 40 還不夠,要看它能不能一路退到 38、35,而且不再彈回 40 以上。持續降溫,才代表恐慌真的在退,不是假摔一下又衝回去。",
    "vix_spread": "美股也有一個恐慌指數(VIX)。台股自己飆高、美股很平靜,代表是「台灣自己的事」在嚇人。當兩邊差距開始縮小,就是台灣本地的驚慌在消退。",
    "vix_spike": "見底常有個「最後一跳」:大家恐慌到極點、指數爆衝一根,接著突然急殺回落。這種「衝高後快速崩下來」通常代表恐慌一次宣洩完,是見底的典型樣子。",
    "px_no_low": "下跌要停,第一步就是「不再創新低」。今天的低點如果守在前波低點之上,代表賣壓開始撐得住了。",
    "px_higher_low": "跌勢的特徵是「低點愈來愈低」。當這次回檔的低點比上次還高(一底比一底高),就是趨勢可能要轉向的第一個結構訊號。",
    "px_strong_k": "像「長下影線」或「把前一天跌幅吃回去」的大紅K,代表盤中殺低後被買盤強力拉回。重點是收盤要守住——盤中拉高、尾盤又被殺回去的不算數。",
    "px_hold": "一根紅K可能只是反彈逃命。要連著幾天都站得住、不再破低,才能比較放心說「跌勢真的停了」,而不是曇花一現。",
    "vol_spike_down": "成交量爆出來的大跌,常代表想賣的人一次砍光、恐慌賣壓宣洩完。賣的人都賣完了,後面就比較沒有賣壓。",
    "vol_shrink": "爆量殺完後,如果隔天成交量明顯縮小、股價也跌不太下去,代表想賣的都賣完、沒人想再殺——這是賣壓枯竭、止穩的訊號。",
    "vol_up": "「有量的上漲」:量放大、價也漲,代表真的有買盤進來承接,不是沒人氣的虛彈。",
    "fut_basis": "台指期跟現貨會有價差。期貨比現貨低(逆價差)代表大戶看壞;當它回到比現貨高(正價差)並連續幾天站穩,代表大戶不再那麼看空了。",
    "f_spot": "外資是台股最大咖。看他們每天現貨買賣超,當「賣超金額明顯縮小」甚至翻買超,代表外資殺盤的力道在減弱。",
    "fi_cover": "外資在期貨押多還是押空,看他們的「淨部位」。空單回補(減少看空)、甚至翻成做多,是他們態度轉好的訊號。",
    "pc_falloff": "P/C 比衝到極端高,代表大家瘋狂買保險避險、恐慌到頂。當它從極端往回降,通常對應恐慌見頂、市場開始冷靜。",
    "tsmc_no_low": "台積電一檔就占大盤約三成,它一弱、指數就很難止跌。所以台積不破低、站穩,是大盤能不能跟著止跌的關鍵。",
    "tsmc_ma": "比「止穩」再進一步:台積電由黑翻紅、站回 5 日線這種短期均線,代表龍頭真的轉強,指數比較有機會跟著往上。",
    "us_10y": "恐慌時資金會躲進美國公債避險(殖利率被壓低)。當殖利率回升,代表資金敢離開避風港、回去買股票了——風險偏好回溫的旁證。注意:若大跌主因是美國升息或通膨,這項方向會相反,僅供輔助參考。",
    "dxy_fall": "恐慌時資金搶買美元避險(美元指數走高)。當美元指數回落,代表資金開始流出美元、回流亞洲市場,台幣比較有撐、外資也比較願意回來。輔助參考用。",
    "news_dulled": "判斷利空退燒的徵兆:同樣的壞消息再出來,股價卻不太跌了——市場開始無感,代表利空被消化得差不多。這項電腦判斷不可靠,請自行判斷後勾選。",
    "news_resolved": "比「市場無感」更直接:造成恐慌的原因本身有了結果、講清楚了、或者解除了。根源消失,股價最容易快速回神。請自行判斷後勾選。",
}


def compute_level(items) -> dict:
    """照規格 §4:閘門 → 打底 → 確認。回 {level, level_label, level_icon, n_ok}。"""
    d = {it["key"]: it for it in items}
    n_ok = sum(1 for it in items if it["ok"] is True)

    def _ok(key):
        return d.get(key, {}).get("ok") is True

    if not _ok("gate"):
        lv = 0
    else:
        base = (_ok("px_strong_k") or _ok("px_hold")) and \
               (_ok("vol_spike_down") or _ok("vol_shrink") or _ok("vol_up"))
        if base:
            core = all(_ok(k) for k in CORE_KEYS)
            follow = _ok("vol_up")
            lv = 3 if (core and follow and n_ok >= CFG["level3_min_ok"]) else 2
        else:
            lv = 1
    meta = LEVELS[lv]
    return {"level": lv, "level_label": meta["label"],
            "level_icon": meta["icon"], "n_ok": n_ok}


def apply_manual_flags(result: dict, flags: dict) -> dict:
    """UI 勾選變更後,就地更新人工項 + 重算分級(不重抓資料)。"""
    for it in result["items"]:
        if it["manual"]:
            it["ok"] = bool(flags.get(it["key"]))
    result["manual_flags"] = flags
    result.update(compute_level(result["items"]))
    return result


# ══════════════════════════════════════════════════════════════════════
# 歷史 / 人工勾選 持久化(模式同 persist_sentiment_history)
# ══════════════════════════════════════════════════════════════════════
def load_manual_flags(cache_dir) -> dict:
    if cache_dir is None:
        return {}
    f = Path(cache_dir) / MANUAL_FILE
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_manual_flags(cache_dir, flags: dict):
    if cache_dir is None:
        return
    f = Path(cache_dir) / MANUAL_FILE
    flags = dict(flags, updated=datetime.now().strftime("%Y-%m-%d %H:%M"))
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(flags, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"   ⚠ {MANUAL_FILE} 寫入失敗: {e}")


def load_bottom_history(cache_dir) -> list:
    if cache_dir is None:
        return []
    f = Path(cache_dir) / HISTORY_FILE
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return []


def persist_bottom_history(cache_dir, result: dict):
    """每日一筆:日期 / 分級 / 成立數 / 各項 ok。同日重跑覆蓋。"""
    if cache_dir is None or result is None:
        return
    f = Path(cache_dir) / HISTORY_FILE
    today_str = datetime.now().strftime("%Y-%m-%d")
    history = [h for h in load_bottom_history(cache_dir) if h.get("date") != today_str]
    history.append({
        "date": today_str,
        "level": result["level"],
        "n_ok": result["n_ok"],
        "vixtwn": result.get("vixtwn"),
        "items": {it["key"]: it["ok"] for it in result["items"]},
    })
    history = sorted(history, key=lambda h: h["date"])[-_HISTORY_KEEP:]
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"   ⚠ {HISTORY_FILE} 寫入失敗: {e}")


# ══════════════════════════════════════════════════════════════════════
# 輸出格式
# ══════════════════════════════════════════════════════════════════════
def format_bottom_for_tg(result: dict) -> str:
    """Telegram 推播格式(同規格 §4 範例)。"""
    ok_items = [it for it in result["items"] if it["ok"] is True]
    no_items = [it for it in result["items"] if it["ok"] is False]
    na_items = [it for it in result["items"] if it["ok"] is None]
    d = {it["key"]: it for it in result["items"]}
    gate = d["gate"]

    lines = [
        f"【台股止跌判讀 {result['asof']}】",
        f"分級:{result['level_label']} {result['level_icon']}",
        f"✅ 已成立({len(ok_items)}):" + "、".join(it["name"] for it in ok_items)
        if ok_items else "✅ 已成立(0)",
        f"⬜ 未成立({len(no_items)}):" + "、".join(it["name"] for it in no_items)
        if no_items else "⬜ 未成立(0)",
    ]
    if na_items:
        lines.append(f"❓ 資料缺({len(na_items)}):" +
                     "、".join(it["name"] for it in na_items))
    lines.append(f"🔑 閘門:VIXTWN {gate['value']} "
                 f"{'✓ 已開' if gate['ok'] else '✗ 未開'}")
    if not result["manual_flags"].get("news_dulled") and result["level"] >= 2:
        lines.append("⚠️ 快轉綠燈:請至 UI 確認「利空鈍化」是否勾選")
    for a in result["alerts"]:
        lines.append(f"⚠️ {a}")
    return "\n".join(lines)


def print_report(result: dict):
    """終端機完整報告(本機測試用)。"""
    print()
    print("=" * 56)
    print(f"  台股止跌判讀  {result['asof']}")
    print(f"  分級:{result['level_icon']} {result['level_label']}"
          f"(成立 {result['n_ok']}/{len(result['items'])})")
    print("=" * 56)
    cur_group = None
    for it in result["items"]:
        if it["group"] != cur_group:
            cur_group = it["group"]
            print(f"\n── {cur_group} " + "─" * (40 - len(cur_group)))
        mark = "✅" if it["ok"] is True else ("⬜" if it["ok"] is False else "❓")
        tag = "(人工)" if it["manual"] else ""
        print(f"  {mark} {it['name']:<14}{tag} {it['value']}")
    for a in result["alerts"]:
        print(f"\n⚠️  {a}")
    print()


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    cache = Path(__file__).parent / "cache"
    res = run_all_checks(cache_dir=cache)
    print_report(res)
    persist_bottom_history(cache, res)
    print("📲 Telegram 預覽:")
    print("-" * 40)
    print(format_bottom_for_tg(res))
