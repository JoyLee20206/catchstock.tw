"""大盤情緒指標(第 1 波)

整合 4 個訊號算出市場溫度計(0~100):
1. VIX(美股恐慌指數)        — yfinance ^VIX
2. 富邦 VIX(台指 VIX ETF)    — yfinance 00677U.TW
3. 大盤位階(乖離率)         — yfinance ^TWII vs MA60
4. 融資週變化                 — 既有 margin parquet 算

設計原則:
- 任一資料源失敗,該指標標 N/A,但**其他指標仍正常算溫度**(部分降級)
- 全部失敗時,回傳 sentiment=None,讓呼叫端決定是否略過
"""
import time
import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# 個別指標
# ══════════════════════════════════════════════════════════════════════
def _yf_download_with_retry(ticker: str, period: str = "60d", max_retries: int = 3):
    """yfinance 包 retry,失敗 sleep 30 秒重試。
    這對應之前 ^TWII 也是 429 fail 的問題 ── 這次起所有 yfinance 都有保護。
    """
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

    區間判斷(經驗值,並非鐵則):
    - < 15: 樂觀(熱) — bull
    - 15~20: 中性
    - 20~30: 警戒
    - > 30: 恐慌(冷)
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
    """台指 VIX(用 00677U 富邦 VIX ETF 當代理)。

    用「歷史百分位」判讀:看當前值在過去 60 日的位置。
    高百分位 = 恐慌情緒升高;低百分位 = 樂觀。
    """
    df = _yf_download_with_retry("00677U.TW", period="90d")
    if df is None or df.empty or len(df) < 20:
        return {"value": None, "pct_rank": None, "label": "N/A", "score": None, "icon": "⚪"}

    closes = df['Close'].dropna()
    val = float(closes.iloc[-1].item() if hasattr(closes.iloc[-1], 'item') else closes.iloc[-1])

    # 在過去 60 日中的百分位(0~100,越高代表越偏高)
    recent = closes.tail(60)
    pct = float((recent < val).sum() / len(recent) * 100)

    if pct < 25:
        label, icon, score = "偏低", "🟢", 70
    elif pct < 50:
        label, icon, score = "中低", "🟢", 60
    elif pct < 75:
        label, icon, score = "中高", "🟡", 40
    else:
        label, icon, score = "偏高", "🔴", 20

    return {"value": round(val, 2), "pct_rank": round(pct, 0),
            "label": label, "score": score, "icon": icon}


def get_taiex_position() -> dict:
    """大盤位階 — 加權指數相對 MA60 乖離率。

    這個訊號和你 screening0515 的「twii_bias」概念一致,但這邊獨立抓避免耦合。
    """
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
        label, icon, score = "深跌", "🟠", 80  # 深跌反而代表「逢低買進」機會

    return {"value": round(bias, 1), "label": label, "score": score, "icon": icon}


def get_margin_change(cache_dir) -> dict:
    """融資週變化 — 用既有 margin parquet 算。

    比較「最近 5 個交易日總融資餘額」vs「再往前 5 個交易日總融資餘額」。

    意義:
    - 融資週增 > 5% → 散戶積極做多(若市場已高,警訊)
    - 融資週減 > 5% → 散戶撤退(若市場已低,接近底部)
    - 介於中間 → 中性
    """
    try:
        files = sorted(cache_dir.glob('margin_*.parquet'))
        if not files:
            return {"value": None, "label": "N/A", "score": None, "icon": "⚪"}

        df = pd.read_parquet(files[-1])
        if df.empty or 'MarginPurchaseTodayBalance' not in df.columns:
            return {"value": None, "label": "N/A", "score": None, "icon": "⚪"}

        df['date'] = pd.to_datetime(df['date'])
        # 每日全市場總融資餘額(加總所有股票)
        daily_total = df.groupby('date')['MarginPurchaseTodayBalance'].sum().sort_index()
        if len(daily_total) < 10:
            return {"value": None, "label": "N/A", "score": None, "icon": "⚪"}

        recent_5 = daily_total.tail(5).mean()
        prev_5   = daily_total.iloc[-10:-5].mean()
        if prev_5 <= 0:
            return {"value": None, "label": "N/A", "score": None, "icon": "⚪"}

        change_pct = (recent_5 - prev_5) / prev_5 * 100

        # 注意:這個訊號的「好壞」要看當前位階,單獨看融資變化不能直接判斷
        # 這裡先給「散戶情緒方向」分數,溫度計加總時融資只當輔助訊號
        if change_pct > 5:
            label, icon, score = "急增", "🔴", 25     # 散戶過熱(反指標)
        elif change_pct > 2:
            label, icon, score = "增加", "🟡", 45
        elif change_pct > -2:
            label, icon, score = "持平", "🟢", 55
        elif change_pct > -5:
            label, icon, score = "減少", "🟢", 65
        else:
            label, icon, score = "急減", "🟠", 70     # 散戶撤退(可能接近底部)

        return {"value": round(change_pct, 1), "label": label, "score": score, "icon": icon}
    except Exception as e:
        print(f"   ⚠ 融資週變化計算失敗: {e}")
        return {"value": None, "label": "N/A", "score": None, "icon": "⚪"}


# ══════════════════════════════════════════════════════════════════════
# 組合溫度計
# ══════════════════════════════════════════════════════════════════════
# 權重設計(總和 1.0):
# VIX 跟富邦VIX 是「恐慌情緒」直接指標,權重最大
# 位階偏熱會增加修正風險
# 融資是輔助訊號,權重最小(且方向跟散戶相反,單獨判斷不可靠)
WEIGHTS = {
    "vix":           0.30,
    "taiex_vix":     0.30,
    "taiex_pos":     0.25,
    "margin_change": 0.15,
}


def compute_sentiment(cache_dir) -> dict:
    """組合所有指標,算總體市場溫度(0~100)。

    Returns:
        {
            "indicators": {
                "vix":           {value, label, score, icon},
                "taiex_vix":     {value, pct_rank, label, score, icon},
                "taiex_pos":     {value, label, score, icon},
                "margin_change": {value, label, score, icon},
            },
            "temperature": int (0~100) | None,
            "label": "極冷" | "偏冷" | "中性" | "偏熱" | "過熱",
            "icon":  "🥶" | "❄️" | "🌤️" | "☀️" | "🔥",
        }
    """
    indicators = {
        "vix":           get_vix(),
        "taiex_vix":     get_taiex_vix(),
        "taiex_pos":     get_taiex_position(),
        "margin_change": get_margin_change(cache_dir),
    }

    # 加權平均(只用有 score 的指標,並對權重做歸一化)
    valid = {k: v for k, v in indicators.items() if v.get("score") is not None}
    if not valid:
        return {"indicators": indicators, "temperature": None, "label": "N/A", "icon": "⚪"}

    total_w = sum(WEIGHTS[k] for k in valid)
    temp = sum(WEIGHTS[k] * v["score"] for k, v in valid.items()) / total_w
    temp = round(temp)

    # 溫度 → 文字標籤(直覺對應,分數高 = 偏多/熱、分數低 = 偏空/冷)
    # 注意這跟「漲跌」無直接關聯,而是「市場情緒」
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
def format_sentiment_for_tg(sentiment: dict) -> str:
    """組 TG 訊息區塊(HTML 格式)。

    範例輸出:
        📊 <b>大盤情緒指標</b>
        🌡️ 溫度計:<b>65 / 100</b> ☀️ 略偏多
        🟢 VIX 18.2(偏低)
        🟢 富邦VIX 14.5(歷史 28%位,偏低)
        🟡 加權位階 +5.2%(略高)
        🟢 融資週變化 -2.1%(減少)
    """
    if not sentiment or sentiment.get("temperature") is None:
        return ""  # 全部 fail 時不顯示這段

    lines = ["\n📊 <b>大盤情緒指標</b>"]
    lines.append(f"🌡️ 溫度計:<b>{sentiment['temperature']} / 100</b> "
                 f"{sentiment['icon']} {sentiment['label']}")

    ind = sentiment["indicators"]

    # VIX
    v = ind.get("vix", {})
    if v.get("value") is not None:
        lines.append(f"{v['icon']} VIX {v['value']}({v['label']})")

    # 富邦 VIX
    v = ind.get("taiex_vix", {})
    if v.get("value") is not None:
        pct_str = f",歷史 {int(v['pct_rank'])}% 位" if v.get("pct_rank") is not None else ""
        lines.append(f"{v['icon']} 富邦VIX {v['value']}{pct_str}({v['label']})")

    # 加權位階
    v = ind.get("taiex_pos", {})
    if v.get("value") is not None:
        sign = "+" if v["value"] >= 0 else ""
        lines.append(f"{v['icon']} 加權位階 {sign}{v['value']}%({v['label']})")

    # 融資週變化
    v = ind.get("margin_change", {})
    if v.get("value") is not None:
        sign = "+" if v["value"] >= 0 else ""
        lines.append(f"{v['icon']} 融資週變化 {sign}{v['value']}%({v['label']})")

    return "\n".join(lines) + "\n"
