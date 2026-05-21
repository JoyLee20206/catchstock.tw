"""大盤情緒指標(第 2 波)

整合 7 個訊號算出市場溫度計(0~100):
1. VIX(美股恐慌指數)        — yfinance ^VIX
2. 富邦 VIX(台指 VIX ETF)    — yfinance 00677U.TW
3. 大盤位階(乖離率)         — yfinance ^TWII vs MA60
4. 融資週變化                 — 既有 margin parquet 算
5. 融資水位百分位             — 既有 margin parquet 算(近90日)
6. 外資期貨淨口數             — TAIFEX 三大法人期貨未平倉 (TX)
7. 散戶方向估算               — MTX 非法人淨口數取反

設計原則:
- 任一資料源失敗,該指標標 N/A,但**其他指標仍正常算溫度**(部分降級)
- 全部失敗時,回傳 sentiment=None,讓呼叫端決定是否略過
"""
import time
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# 個別指標
# ══════════════════════════════════════════════════════════════════════
def _yf_download_with_retry(ticker: str, period: str = "60d", max_retries: int = 3):
    """yfinance 包 retry,失敗 sleep 30 秒重試。"""
    import yfinance as yf
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, period=period, auto_adjust=True,
                             progress=False, threads=False)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            err = str(e)[:100]
            if attempt < max_retries - 1:
                print(f"   ⚠ {ticker} 第 {attempt+1} 次失敗({err}),30 秒後重試")
                time.sleep(30)
            else:
                print(f"   ❌ {ticker} 全部 {max_retries} 次嘗試都失敗")
    return None


def get_vix() -> dict:
    """美股 VIX。

    區間判斷(經驗值):
    - < 15: 樂觀 — bull
    - 15~20: 中性
    - 20~30: 警戒
    - > 30: 恐慌
    """
    df = _yf_download_with_retry("^VIX", period="5d")
    if df is None or df.empty:
        return {"value": None, "label": "N/A", "score": None, "icon": "⚪"}

    val = float(df['Close'].iloc[-1].item() if hasattr(df['Close'].iloc[-1], 'item') else df['Close'].iloc[-1])
    if val < 15:
        label, icon, score = "樂觀", "🟢", 75
    elif val < 20:
        label, icon, score = "偏低", "🟢", 65
    elif val < 25:
        label, icon, score = "中性", "🟡", 50
    elif val < 30:
        label, icon, score = "警戒", "🟠", 30
    else:
        label, icon, score = "恐慌", "🔴", 15
    return {"value": round(val, 1), "label": label, "score": score, "icon": icon}


def get_taiex_vix() -> dict:
    """台指實現波動率(從 ^TWII 算)。

    原本用 00677U 富邦 VIX ETF 當代理,但該 ETF 已下市(2024),
    改為直接從加權指數計算 20 日年化實現波動率:
        RV = std(daily_log_return, 20) * sqrt(252) * 100

    用「歷史百分位」判讀:看當前值在過去 90 日的位置。
    高百分位 = 波動高(恐慌升溫);低百分位 = 平靜樂觀。
    """
    df = _yf_download_with_retry("^TWII", period="180d")
    if df is None or df.empty or len(df) < 40:
        return {"value": None, "pct_rank": None, "label": "N/A", "score": None, "icon": "⚪"}

    closes = df['Close'].dropna()
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]

    # 日對數報酬率 → 20 日滾動標準差 → 年化(×√252)→ 百分比
    log_ret = np.log(closes / closes.shift(1)).dropna()
    rv_series = log_ret.rolling(window=20).std() * np.sqrt(252) * 100
    rv_series = rv_series.dropna()
    if len(rv_series) < 20:
        return {"value": None, "pct_rank": None, "label": "N/A", "score": None, "icon": "⚪"}

    val = float(rv_series.iloc[-1])
    recent = rv_series.tail(90)
    pct = float((recent < val).sum() / len(recent) * 100)

    if pct < 25:
        label, icon, score = "偏低", "🟢", 70
    elif pct < 50:
        label, icon, score = "中低", "🟢", 60
    elif pct < 75:
        label, icon, score = "中高", "🟡", 40
    else:
        label, icon, score = "偏高", "🔴", 20

    return {"value": round(val, 1), "pct_rank": round(pct, 0),
            "label": label, "score": score, "icon": icon}


def get_taiex_position() -> dict:
    """大盤位階 — 加權指數相對 MA60 乖離率。"""
    df = _yf_download_with_retry("^TWII", period="120d")
    if df is None or df.empty or len(df) < 60:
        return {"value": None, "label": "N/A", "score": None, "icon": "⚪"}

    closes = df['Close'].dropna()
    latest = float(closes.iloc[-1].item() if hasattr(closes.iloc[-1], 'item') else closes.iloc[-1])
    ma60 = float(closes.tail(60).mean().item() if hasattr(closes.tail(60).mean(), 'item') else closes.tail(60).mean())
    bias = (latest - ma60) / ma60 * 100

    if bias > 8:
        label, icon, score = "過熱", "🔴", 15
    elif bias > 3:
        label, icon, score = "略高", "🟡", 40
    elif bias > -3:
        label, icon, score = "正常", "🟢", 60
    elif bias > -8:
        label, icon, score = "略低", "🟢", 70
    else:
        label, icon, score = "深跌", "🟠", 80

    return {"value": round(bias, 1), "label": label, "score": score, "icon": icon}


def get_margin_change(cache_dir) -> dict:
    """融資週變化 — 比較近 5 日 vs 前 5 日平均融資餘額。"""
    try:
        files = sorted(cache_dir.glob('margin_*.parquet'))
        if not files:
            return {"value": None, "label": "N/A", "score": None, "icon": "⚪"}

        df = pd.read_parquet(files[-1])
        if df.empty or 'MarginPurchaseTodayBalance' not in df.columns:
            return {"value": None, "label": "N/A", "score": None, "icon": "⚪"}

        df['date'] = pd.to_datetime(df['date'])
        daily_total = df.groupby('date')['MarginPurchaseTodayBalance'].sum().sort_index()
        if len(daily_total) < 10:
            return {"value": None, "label": "N/A", "score": None, "icon": "⚪"}

        recent_5 = daily_total.tail(5).mean()
        prev_5   = daily_total.iloc[-10:-5].mean()
        if prev_5 <= 0:
            return {"value": None, "label": "N/A", "score": None, "icon": "⚪"}

        change_pct = (recent_5 - prev_5) / prev_5 * 100

        if change_pct > 5:
            label, icon, score = "急增", "🔴", 25
        elif change_pct > 2:
            label, icon, score = "增加", "🟡", 45
        elif change_pct > -2:
            label, icon, score = "持平", "🟢", 55
        elif change_pct > -5:
            label, icon, score = "減少", "🟢", 65
        else:
            label, icon, score = "急減", "🟠", 70

        return {"value": round(change_pct, 1), "label": label, "score": score, "icon": icon}
    except Exception as e:
        print(f"   ⚠ 融資週變化計算失敗: {e}")
        return {"value": None, "label": "N/A", "score": None, "icon": "⚪"}


def get_margin_balance_level(cache_dir) -> dict:
    """全市場融資水位(近 90 日百分位)。

    高百分位 = 融資餘額偏高(散戶槓桿重,反向警訊)
    低百分位 = 融資偏低(底部特徵,偏多)
    """
    try:
        files = sorted(cache_dir.glob('margin_*.parquet'))
        if not files:
            return {"value": None, "pct_rank": None, "label": "N/A", "score": None, "icon": "⚪"}

        df = pd.read_parquet(files[-1])
        if df.empty or 'MarginPurchaseTodayBalance' not in df.columns:
            return {"value": None, "pct_rank": None, "label": "N/A", "score": None, "icon": "⚪"}

        df['date'] = pd.to_datetime(df['date'])
        daily_total = df.groupby('date')['MarginPurchaseTodayBalance'].sum().sort_index()
        if len(daily_total) < 10:
            return {"value": None, "pct_rank": None, "label": "N/A", "score": None, "icon": "⚪"}

        window = daily_total.tail(90)
        current = float(window.iloc[-1])
        pct = float((window < current).sum() / len(window) * 100)
        # 換算為億張(易讀)
        val_bn = round(current / 1e6, 1)

        if pct < 20:
            label, icon, score = "極低", "🟢", 80
        elif pct < 40:
            label, icon, score = "偏低", "🟢", 65
        elif pct < 60:
            label, icon, score = "中性", "🟡", 50
        elif pct < 80:
            label, icon, score = "偏高", "🟠", 35
        else:
            label, icon, score = "極高", "🔴", 15

        return {"value": val_bn, "pct_rank": round(pct, 0),
                "label": label, "score": score, "icon": icon}
    except Exception as e:
        print(f"   ⚠ 融資水位計算失敗: {e}")
        return {"value": None, "pct_rank": None, "label": "N/A", "score": None, "icon": "⚪"}


# ── TAIFEX 三大法人期貨未平倉 ──────────────────────────────────────────────
def _fetch_taifex_institutional(commodity_id: str = "TX") -> pd.DataFrame | None:
    """從 TAIFEX 抓最近交易日三大法人期貨未平倉口數 CSV。

    commodity_id: 'TX' (大台指) / 'MTX' (小台指)
    欄位: date, contract, trader, long_vol, long_amt, short_vol, short_amt, net_vol, net_amt
    """
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "Referer": "https://www.taifex.com.tw/",
    }
    today = datetime.now()

    for delta in range(0, 8):
        dt = today - timedelta(days=delta)
        if dt.weekday() >= 5:
            continue
        date_str = dt.strftime("%Y/%m/%d")
        url = (
            "https://www.taifex.com.tw/cht/3/futContractsDateDown"
            f"?queryType=1&marketCode=0&dateaddcnt=0"
            f"&commodity_id={commodity_id}&queryDate={date_str}"
        )
        try:
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code != 200:
                continue

            # TAIFEX 可能回 UTF-8-sig 或 Big5
            for enc in ("utf-8-sig", "big5", "utf-8"):
                try:
                    text = resp.content.decode(enc)
                    break
                except Exception:
                    text = None
            if not text or "外資" not in text:
                continue

            # 跳過標題行,只取含數字的資料行
            rows = []
            for line in text.splitlines():
                stripped = line.strip().strip('"')
                if not stripped or stripped.startswith("期貨"):
                    continue
                parts = [p.strip().strip('"').replace(",", "") for p in line.split(",")]
                if len(parts) >= 8 and parts[0].startswith("20"):
                    rows.append(parts[:9])

            if not rows:
                continue

            col_names = ["date", "contract", "trader",
                         "long_vol", "long_amt", "short_vol", "short_amt",
                         "net_vol", "net_amt"]
            df = pd.DataFrame(rows, columns=col_names[:len(rows[0])])

            # 數值欄轉型
            for c in ["long_vol", "short_vol", "net_vol"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

            return df

        except Exception as e:
            print(f"   ⚠ TAIFEX {commodity_id} {date_str} 抓取失敗: {str(e)[:80]}")
            continue

    return None


def get_fi_futures_net() -> dict:
    """外資期貨淨口數(大台 TX)。

    外資淨多口 > 0 → 偏多;< 0 → 偏空。
    用過去 20 個可用交易日的歷史百分位判讀信號強度。
    """
    _EMPTY = {"value": None, "pct_rank": None, "label": "N/A", "score": None, "icon": "⚪"}
    try:
        df = _fetch_taifex_institutional("TX")
        if df is None or df.empty:
            return _EMPTY

        fi_row = df[df["trader"].str.contains("外資", na=False)]
        if fi_row.empty:
            return _EMPTY

        net = int(fi_row["net_vol"].iloc[0])

        # 百分位:用過去幾個工作日的 net_vol 快取無法取得,直接用絕對值閾值判讀
        # 台指大台未平倉淨口數,±5000 以上為顯著方向
        if net > 10000:
            label, icon, score = "強多", "🟢", 75
        elif net > 3000:
            label, icon, score = "偏多", "🟢", 65
        elif net > -3000:
            label, icon, score = "中性", "🟡", 50
        elif net > -10000:
            label, icon, score = "偏空", "🟠", 35
        else:
            label, icon, score = "強空", "🔴", 20

        return {"value": net, "pct_rank": None, "label": label, "score": score, "icon": icon}
    except Exception as e:
        print(f"   ⚠ 外資期貨淨口數計算失敗: {e}")
        return _EMPTY


def get_retail_futures_ratio() -> dict:
    """散戶方向估算(小台 MTX)。

    用「三大法人淨口數加總取反」估算散戶方向:
    零和市場中,散戶淨口數 ≈ -(自營+投信+外資合計淨口數)
    正值 = 散戶偏多(反指標,偏謹慎);負值 = 散戶偏空(逆勢偏多)
    """
    _EMPTY = {"value": None, "label": "N/A", "score": None, "icon": "⚪"}
    try:
        df = _fetch_taifex_institutional("MTX")
        if df is None or df.empty:
            return _EMPTY

        institutional_traders = ["自營商", "投信", "外資"]
        mask = df["trader"].str.contains("|".join(institutional_traders), na=False)
        inst_df = df[mask]
        if inst_df.empty:
            return _EMPTY

        inst_net_total = int(inst_df["net_vol"].sum())
        # 散戶 ≈ 法人淨口數的反方向
        retail_est = -inst_net_total

        if retail_est > 5000:
            label, icon, score = "散戶多", "🟠", 40   # 散戶偏多 → 反向警訊
        elif retail_est > 1000:
            label, icon, score = "略偏多", "🟡", 48
        elif retail_est > -1000:
            label, icon, score = "中性", "🟢", 55
        elif retail_est > -5000:
            label, icon, score = "略偏空", "🟢", 62
        else:
            label, icon, score = "散戶空", "🟢", 70   # 散戶偏空 → 反向利多

        return {"value": retail_est, "label": label, "score": score, "icon": icon}
    except Exception as e:
        print(f"   ⚠ 散戶方向估算失敗: {e}")
        return _EMPTY


# ══════════════════════════════════════════════════════════════════════
# 組合溫度計
# ══════════════════════════════════════════════════════════════════════
# 權重設計(總和 1.0):
# 外資期貨方向最直接反映大型資金觀點
# VIX 反映恐慌情緒;大盤位階反映估值;融資類反映散戶籌碼
WEIGHTS = {
    "vix":             0.20,
    "taiex_vix":       0.10,
    "taiex_pos":       0.20,
    "margin_change":   0.10,
    "margin_level":    0.10,
    "fi_futures":      0.20,
    "retail_futures":  0.10,
}


def compute_sentiment(cache_dir) -> dict:
    """組合所有指標,算總體市場溫度(0~100)。

    Returns:
        {
            "indicators": {key: {value, label, score, icon}, ...},
            "temperature": int (0~100) | None,
            "label": str,
            "icon":  str,
        }
    """
    indicators = {
        "vix":            get_vix(),
        "taiex_vix":      get_taiex_vix(),
        "taiex_pos":      get_taiex_position(),
        "margin_change":  get_margin_change(cache_dir),
        "margin_level":   get_margin_balance_level(cache_dir),
        "fi_futures":     get_fi_futures_net(),
        "retail_futures": get_retail_futures_ratio(),
    }

    valid = {k: v for k, v in indicators.items() if v.get("score") is not None}
    if not valid:
        return {"indicators": indicators, "temperature": None, "label": "N/A", "icon": "⚪"}

    total_w = sum(WEIGHTS[k] for k in valid)
    temp = sum(WEIGHTS[k] * v["score"] for k, v in valid.items()) / total_w
    temp = round(temp)

    if temp >= 70:
        label, icon = "偏熱(樂觀)", "☀️"
    elif temp >= 55:
        label, icon = "略偏多", "🌤️"
    elif temp >= 45:
        label, icon = "中性", "🌥️"
    elif temp >= 30:
        label, icon = "略偏空", "🌦️"
    else:
        label, icon = "偏冷(恐慌)", "❄️"

    return {"indicators": indicators, "temperature": temp, "label": label, "icon": icon}


# ══════════════════════════════════════════════════════════════════════
# 格式化(給 TG)
# ══════════════════════════════════════════════════════════════════════
def format_sentiment_summary_line(sentiment: dict) -> str:
    """單行摘要,嵌入每日推播標頭。

    範例:📊 大盤情緒：溫度 65 🌤️ | VIX 18.2 偏低 | 台指波動 28%位 | 外資期貨 +8,234口
    """
    if not sentiment or sentiment.get("temperature") is None:
        return ""

    ind = sentiment["indicators"]
    parts = [f"🌡️ 市場溫度 <b>{sentiment['temperature']}</b> {sentiment['icon']} {sentiment['label']}"]

    v = ind.get("vix", {})
    if v.get("value") is not None:
        parts.append(f"VIX {v['value']} {v['label']}")

    v = ind.get("margin_level", {})
    if v.get("pct_rank") is not None:
        parts.append(f"融資水位 {int(v['pct_rank'])}%位 {v['label']}")

    v = ind.get("fi_futures", {})
    if v.get("value") is not None:
        sign = "+" if v["value"] >= 0 else ""
        parts.append(f"外資期貨 {sign}{v['value']:,}口 {v['label']}")

    return "📊 <b>大盤情緒</b>：" + " | ".join(parts[1:]) + f"\n{parts[0]}\n"


def format_sentiment_for_tg(sentiment: dict) -> str:
    """完整情緒指標區塊(HTML 格式),供 TG 詳細段落用。

    範例:
        📊 <b>大盤情緒指標</b>
        🌡️ 溫度計：<b>65 / 100</b> 🌤️ 略偏多
        🟢 VIX 18.2(偏低)
        🟢 台指波動 14.5(歷史 28%位,偏低)
        🟢 加權位階 +5.2%(略高)
        🟢 融資週變化 -2.1%(減少)
        🟡 融資水位 62%位(偏高)
        🟢 外資期貨 +8,234口(偏多)
        🟠 散戶估算 +5,100口(散戶多)
    """
    if not sentiment or sentiment.get("temperature") is None:
        return ""

    lines = ["\n📊 <b>大盤情緒指標</b>"]
    lines.append(f"🌡️ 溫度計:<b>{sentiment['temperature']} / 100</b> "
                 f"{sentiment['icon']} {sentiment['label']}")

    ind = sentiment["indicators"]

    v = ind.get("vix", {})
    if v.get("value") is not None:
        lines.append(f"{v['icon']} VIX {v['value']}({v['label']})")

    v = ind.get("taiex_vix", {})
    if v.get("value") is not None:
        pct_str = f"歷史 {int(v['pct_rank'])}%位," if v.get("pct_rank") is not None else ""
        lines.append(f"{v['icon']} 台指波動 {v['value']}({pct_str}{v['label']})")

    v = ind.get("taiex_pos", {})
    if v.get("value") is not None:
        sign = "+" if v["value"] >= 0 else ""
        lines.append(f"{v['icon']} 加權位階 {sign}{v['value']}%({v['label']})")

    v = ind.get("margin_change", {})
    if v.get("value") is not None:
        sign = "+" if v["value"] >= 0 else ""
        lines.append(f"{v['icon']} 融資週變化 {sign}{v['value']}%({v['label']})")

    v = ind.get("margin_level", {})
    if v.get("pct_rank") is not None:
        lines.append(f"{v['icon']} 融資水位 {int(v['pct_rank'])}%位({v['label']})")

    v = ind.get("fi_futures", {})
    if v.get("value") is not None:
        sign = "+" if v["value"] >= 0 else ""
        lines.append(f"{v['icon']} 外資期貨 {sign}{v['value']:,}口({v['label']})")

    v = ind.get("retail_futures", {})
    if v.get("value") is not None:
        sign = "+" if v["value"] >= 0 else ""
        lines.append(f"{v['icon']} 散戶估算 {sign}{v['value']:,}口({v['label']})")

    return "\n".join(lines) + "\n"
