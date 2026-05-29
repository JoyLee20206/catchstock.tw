"""大盤情緒指標(精簡版 — 6 訊號)

整合 6 個訊號算出市場溫度計(0~100):
1. VIX(美股恐慌指數)        — yfinance ^VIX
2. 台指實現波動率            — yfinance ^TWII (20 日年化 std × √252,
                              原 00677U 富邦 VIX ETF 2024 下市改用實現波動率)
3. 大盤位階(乖離率)         — yfinance ^TWII vs MA60
4. 融資水位百分位             — 既有 margin parquet 算(近 90 日)
5. 外資期貨淨口數             — TAIFEX 大台「臺股期貨」未平倉
                              累積 ≥ 20 日後自動切換到 90 日歷史百分位;
                              過渡期用絕對門檻 ±15k/±40k(2025~2026 放寬版)
6. 散戶方向估算               — TAIFEX 微型臺指期貨非法人淨口數取反
                              (微台散戶占比 ~ 90%,小型臺指期貨為備援)
                              一律輸出 0~100% 部位指數(反指標);
                              累積 ≥ 20 日後自動切換到歷史百分位

已下架:
- 融資週變化(get_margin_change) — 與融資水位重疊度高,函式保留供直接調用

設計原則:
- 任一資料源失敗,該指標標 N/A,但**其他指標仍正常算溫度**(部分降級)
- 全部失敗時,回傳 sentiment=None,讓呼叫端決定是否略過
- 期貨門檻 / 散戶來源 隨市場結構演進,需定期校準
"""
import time
import json
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

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
# 模組級記憶體快取:同一次 process 內,大台 / 微台 兩支函式共用一次 fetch
_TAIFEX_CACHE = {"df": None, "ts": 0.0}
_TAIFEX_TTL_SEC = 1800  # 30 分鐘


def _fetch_taifex_institutional() -> "pd.DataFrame | None":
    """從 TAIFEX 抓最近交易日「三大法人期貨未平倉口數」(POST 表單)。

    歷史上的 CSV GET 端點現在會回「查無資料」alert,改用 POST 抓 HTML 表格。
    一次回傳全部商品 × 三身份別(自營商 / 投信 / 外資),呼叫端自行篩商品。

    回傳 DataFrame,欄位:
        product:     商品名稱(例:臺股期貨 / 小型臺指期貨 / 微型臺指期貨)
        trader:      身份別(自營商 / 投信 / 外資)
        oi_net_vol:  未平倉多空淨額口數(正=淨多、負=淨空)
    """
    import requests

    # 模組級快取(同一進程 30 分鐘內共用)
    now_ts = time.time()
    if _TAIFEX_CACHE["df"] is not None and (now_ts - _TAIFEX_CACHE["ts"]) < _TAIFEX_TTL_SEC:
        return _TAIFEX_CACHE["df"]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
        "Referer": "https://www.taifex.com.tw/cht/3/futContractsDate",
    }
    url = "https://www.taifex.com.tw/cht/3/futContractsDate"
    today = datetime.now()

    for delta in range(0, 8):
        dt = today - timedelta(days=delta)
        if dt.weekday() >= 5:
            continue
        date_str = dt.strftime("%Y/%m/%d")
        form = {
            "queryType":   "1",
            "marketCode":  "0",
            "dateaddcnt":  "",
            "commodity_id": "TXF",      # 任填一個有效商品,回應會列出全部商品
            "queryDate":   date_str,
        }
        try:
            resp = requests.post(url, data=form, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue

            text = resp.content.decode("utf-8", errors="replace")
            if "外資" not in text or "臺股期貨" not in text:
                continue

            tables = pd.read_html(StringIO(text))
            if not tables:
                continue
            tb = tables[0]

            # 多層 header → 攤平為單層
            if isinstance(tb.columns, pd.MultiIndex):
                tb.columns = [c[-1] if isinstance(c, tuple) else c for c in tb.columns]

            # 標準格式 15 欄:
            #   0=序號, 1=商品名稱, 2=身份別,
            #   3~8=交易口數欄,
            #   9~14=未平倉餘額欄(多方口、多方金額、空方口、空方金額、多空淨額口、多空淨額金額)
            if tb.shape[1] < 14:
                continue

            df = pd.DataFrame({
                "product":    tb.iloc[:, 1].astype(str).str.strip(),
                "trader":     tb.iloc[:, 2].astype(str).str.strip(),
                "oi_net_vol": pd.to_numeric(tb.iloc[:, 13], errors="coerce"),
            })
            # 只留三大法人列
            df = df[df["trader"].isin(["自營商", "投信", "外資"])].copy()
            df = df.dropna(subset=["oi_net_vol"])
            df["oi_net_vol"] = df["oi_net_vol"].astype(int)

            if df.empty:
                continue

            _TAIFEX_CACHE["df"] = df
            _TAIFEX_CACHE["ts"] = now_ts
            return df

        except Exception as e:
            print(f"   ⚠ TAIFEX {date_str} POST 失敗: {str(e)[:100]}")
            continue

    return None


# ── 外資期貨歷史持久化(給「90 日百分位」評分) ─────────────────────────
# 為什麼要做:外資期貨淨口數絕對門檻會隨市場規模演進(2018 ±15k、2023 ±20k、
# 2026 ±30k+),每隔一段時間就要重新校準。改用百分位制後永久不用調。
# 累積期間(< 20 日)用放寬的絕對門檻(±15k/±40k)當 fallback。
_FI_HISTORY_FILE = "fi_futures_history.json"
_FI_HISTORY_KEEP = 90
_FI_HISTORY_MIN  = 20


def _load_fi_history(cache_dir):
    if cache_dir is None:
        return []
    f = Path(cache_dir) / _FI_HISTORY_FILE
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return []


def persist_fi_history(cache_dir, net_vol):
    """公開 API:寫入今日外資期貨淨口數,由 UI 在快取外呼叫。

    同日重跑覆蓋,自動保留最近 90 日。設計同 persist_sentiment_history,
    避免被 Streamlit @st.cache_data 擋住而漏寫。
    """
    if cache_dir is None:
        return
    f = Path(cache_dir) / _FI_HISTORY_FILE
    today_str = datetime.now().strftime("%Y-%m-%d")
    history = _load_fi_history(cache_dir)
    history = [h for h in history if h.get("date") != today_str]
    history.append({"date": today_str, "net_vol": int(net_vol)})
    history = sorted(history, key=lambda h: h["date"])[-_FI_HISTORY_KEEP:]
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"   ⚠ fi_futures_history.json 寫入失敗: {e}")


def get_fi_futures_net(cache_dir=None) -> dict:
    """外資期貨淨口數(大台 臺股期貨)+ 雙模式評分。

    評分模式(自動切換):
      - 累積歷史 ≥ 20 日 → 用「過去 90 日百分位」評分(主要)
            高百分位 = 外資相對近 3 個月更偏多 → 樂觀
            低百分位 = 外資相對近 3 個月更偏空 → 恐慌
      - 累積歷史 < 20 日 → 用絕對門檻 ±15k/±40k(放寬版,過渡用)
            避免 2026 規模下「強空」被誤判得太頻繁

    寫檔由呼叫端負責(persist_fi_history),這裡只讀 + 計算。
    """
    _EMPTY = {"value": None, "pct_rank": None, "n_days": 0, "mode": None,
              "label": "N/A", "score": None, "icon": "⚪"}
    try:
        df = _fetch_taifex_institutional()
        if df is None or df.empty:
            return _EMPTY

        # 篩臺股期貨且排除小型/微型(大台精確比對)
        mask = (df["product"] == "臺股期貨") & (df["trader"] == "外資")
        row = df[mask]
        if row.empty:
            return _EMPTY

        net = int(row["oi_net_vol"].iloc[0])

        # 讀過去歷史(不含今天) + 虛擬補上今天的值,讓 n_days 與百分位的分母都包含今日
        # (寫檔仍由 caller 透過 persist_fi_history 負責,維持讀寫分離)
        history = _load_fi_history(cache_dir)
        _today_str = datetime.now().strftime("%Y-%m-%d")
        past_vols = [h["net_vol"] for h in history if h.get("date") != _today_str]
        all_vols = past_vols + [net]   # 今天併入用來計算
        n_days = len(all_vols)

        if n_days >= _FI_HISTORY_MIN:
            # ① 百分位制
            arr = sorted(all_vols)
            below = sum(1 for v in arr if v < net)
            pct = float(below / len(arr) * 100)
            mode = "percentile"
            if pct > 80:
                label, icon, score = "強多", "🟢", 75
            elif pct > 60:
                label, icon, score = "偏多", "🟢", 65
            elif pct >= 40:
                label, icon, score = "中性", "🟡", 50
            elif pct >= 20:
                label, icon, score = "偏空", "🟠", 35
            else:
                label, icon, score = "強空", "🔴", 20
        else:
            # ② 絕對門檻(放寬版,過渡用)
            pct = None
            mode = "absolute"
            if net > 40000:
                label, icon, score = "強多", "🟢", 75
            elif net > 15000:
                label, icon, score = "偏多", "🟢", 65
            elif net > -15000:
                label, icon, score = "中性", "🟡", 50
            elif net > -40000:
                label, icon, score = "偏空", "🟠", 35
            else:
                label, icon, score = "強空", "🔴", 20

        return {"value": net,
                "pct_rank": round(pct, 0) if pct is not None else None,
                "n_days":   n_days,
                "mode":     mode,
                "label": label, "score": score, "icon": icon}
    except Exception as e:
        print(f"   ⚠ 外資期貨淨口數計算失敗: {e}")
        return _EMPTY


# ── 散戶期貨歷史(用於百分位計算)─────────────────────────────────────────
# 每次成功抓到當日散戶淨口數時,就 append 到這個 JSON 檔,
# 累積到 ≥ 20 日後改用「歷史百分位」評分(更穩健、不需手動校準門檻)。
_RETAIL_HISTORY_FILE = "retail_futures_history.json"
_RETAIL_HISTORY_KEEP = 90    # 只保留最近 90 日
_RETAIL_HISTORY_MIN  = 20    # 低於此筆數時,百分位不可靠 → 退回絕對門檻評分


def _load_retail_history(cache_dir):
    f = Path(cache_dir) / _RETAIL_HISTORY_FILE
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_retail_history(cache_dir, history):
    f = Path(cache_dir) / _RETAIL_HISTORY_FILE
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"   ⚠ retail history 寫入失敗: {e}")


def persist_retail_history(cache_dir, retail_net, source):
    """公開 API:寫入今日散戶淨口數,由 UI 在快取外呼叫。

    同日重跑覆蓋,自動保留最近 90 日。設計同 persist_sentiment_history /
    persist_fi_history,避免被 Streamlit @st.cache_data 擋住而漏寫。
    """
    if cache_dir is None:
        return
    today_str = datetime.now().strftime("%Y-%m-%d")
    history = _load_retail_history(cache_dir)
    history = [h for h in history if h.get("date") != today_str]
    history.append({"date": today_str, "retail_net": int(retail_net), "source": source})
    history = sorted(history, key=lambda h: h["date"])[-_RETAIL_HISTORY_KEEP:]
    _save_retail_history(cache_dir, history)


# 線性歸一化半幅(2025~2026 校準):±30k 對應 0% / 100%
_RETAIL_LINEAR_FULLSCALE = 30000


def _retail_linear_pct(retail_net: int) -> float:
    """把散戶淨口數線性映射到 0~100%(50% = 中性)。
        retail_net = 0      → 50%
        retail_net = +30000 → 100% (極多)
        retail_net = -30000 →   0% (極空)
    超過 ±30k 會夾到 0% 或 100%。
    歷史累積 ≥ 20 日後改用真實 90 日百分位取代此估算。
    """
    pct = 50.0 + (retail_net / _RETAIL_LINEAR_FULLSCALE) * 50.0
    return max(0.0, min(100.0, pct))


def _retail_label_by_pct(pct: float):
    """把 0~100% 映射到標籤 / 圖示 / 分數(反指標 — % 高代表散戶多 → 警訊)。"""
    if pct > 80:
        return "散戶極多", "🔴", 25
    elif pct > 60:
        return "散戶偏多", "🟠", 40
    elif pct >= 40:
        return "中性", "🟡", 55
    elif pct >= 20:
        return "散戶偏空", "🟢", 65
    else:
        return "散戶極空", "🟢", 75


def get_retail_futures_ratio(cache_dir=None) -> dict:
    """散戶方向估算(微型臺指期貨 — 最純散戶代理)。

    用「三大法人淨口數加總取反」估算散戶方向(零和市場近似):
        散戶淨口數 ≈ -(自營商 + 投信 + 外資 合計淨口數)

    一律輸出 **散戶部位 0~100%**(50 = 中性、100 = 極多、0 = 極空),
    使用者介面語意統一,內部依歷史長度自動切換量測方式:
      - 累積歷史 ≥ 20 日 → **歷史百分位**(更穩、永不需校準) → pct = % rank
      - 累積歷史 < 20 日 → **線性歸一化**(以 ±30k 為半幅做轉換,作過渡用)

    主來源:微型臺指期貨(散戶占比 ~ 90%,訊號最純)
    備援:  小型臺指期貨(MXF,散戶占比 ~ 50%)
    歷史檔:cache/retail_futures_history.json,自動累積維持最近 90 日

    回傳 dict 多出 `mode` 欄位:"percentile" / "linear" / None,
    UI 可據此標示「歷史百分位」或「線性估算」。
    """
    _EMPTY = {"value": None, "pct": None, "n_days": 0, "mode": None,
              "label": "N/A", "score": None, "icon": "⚪", "source": None}
    try:
        df = _fetch_taifex_institutional()
        if df is None or df.empty:
            return _EMPTY

        # 微台優先(最純散戶),沒資料才退回小台
        source = None
        for target in ("微型臺指期貨", "小型臺指期貨"):
            sub = df[df["product"] == target]
            if not sub.empty:
                source = target
                break
        else:
            return _EMPTY

        inst_net_total = int(sub["oi_net_vol"].sum())
        retail_est = -inst_net_total

        # 讀歷史(不寫,寫檔由 UI 端 persist_retail_history 負責 — 讀寫分離 pattern)
        # 跟 sentiment / fi_futures 一致,避免被 Streamlit @st.cache_data 擋住
        history = _load_retail_history(cache_dir) if cache_dir is not None else []

        # 只用相同 source 的歷史(避免微台/小台混算),且**虛擬包含今天**讓 n_days 從 day 1 即為 1
        _today_str = datetime.now().strftime("%Y-%m-%d")
        past_vols = [h["retail_net"] for h in history
                     if h.get("source") == source and h.get("date") != _today_str]
        same_source_hist = past_vols + [retail_est]
        n_days = len(same_source_hist)

        if n_days >= _RETAIL_HISTORY_MIN:
            # 模式 A:歷史百分位
            arr = sorted(same_source_hist)
            below = sum(1 for v in arr if v < retail_est)
            pct = float(below / len(arr) * 100)
            mode = "percentile"
        else:
            # 模式 B:線性歸一化(過渡用)
            pct = _retail_linear_pct(retail_est)
            mode = "linear"

        label, icon, score = _retail_label_by_pct(pct)

        return {"value": retail_est, "pct": round(pct, 0), "n_days": n_days,
                "mode": mode, "label": label, "score": score, "icon": icon,
                "source": source}

    except Exception as e:
        print(f"   ⚠ 散戶方向估算失敗: {e}")
        return _EMPTY


# ══════════════════════════════════════════════════════════════════════
# 組合溫度計
# ══════════════════════════════════════════════════════════════════════
# 權重設計(compute_sentiment 會把實際有效指標再做歸一化,
# 因此這些絕對值的相對比例才是重點,不需強制 sum = 1.0):
#   主軸 (×3 @ 0.20)  : VIX / 大盤位階 / 外資期貨
#   輔助 (×3 @ 0.10)  : 台指波動率 / 融資水位 / 散戶估算
# 拿掉 margin_change(與 margin_level 重疊度高,後者反映「絕對位階」更穩);
# get_margin_change() 函式仍保留供需要短期動能訊號的人直接調用。
WEIGHTS = {
    "vix":             0.20,
    "taiex_vix":       0.10,
    "taiex_pos":       0.20,
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
        "margin_level":   get_margin_balance_level(cache_dir),
        "fi_futures":     get_fi_futures_net(cache_dir),
        "retail_futures": get_retail_futures_ratio(cache_dir),
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

    # ⚠ 注意:這裡**不寫檔**。寫檔(side effect)由呼叫端在拿到結果後執行,
    # 避免 @st.cache_data 快取住結果後副作用永遠不跑。
    # 寫檔請呼叫 persist_sentiment_history(cache_dir, temp, label)。
    return {"indicators": indicators, "temperature": temp, "label": label, "icon": icon}


# ── 大盤情緒歷史持久化(給「7 日趨勢」metric / 趨勢圖) ───────────────────
# 每次成功算出溫度時由 caller 呼叫一次,同日重跑覆蓋,保留最近 90 日。
# 設計原則:寫檔是副作用,**不放在 compute_sentiment 內**,避免被 Streamlit 快取擋住。
_SENTIMENT_HISTORY_FILE = "sentiment_history.json"
_SENTIMENT_HISTORY_KEEP = 90


def persist_sentiment_history(cache_dir, temp, label):
    """寫入今日溫度到 cache/sentiment_history.json(公開 API)。

    呼叫端在拿到 compute_sentiment 結果後呼叫一次。
    每天首次寫入會 append;同日重複呼叫會覆蓋當日紀錄(不重複)。
    """
    if cache_dir is None:
        return
    f = Path(cache_dir) / _SENTIMENT_HISTORY_FILE
    today_str = datetime.now().strftime("%Y-%m-%d")
    history = []
    if f.exists():
        try:
            history = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            history = []
    # 覆蓋今日紀錄,避免同日多次寫入重複
    history = [h for h in history if h.get("date") != today_str]
    history.append({"date": today_str, "temp": int(temp), "label": label})
    history = sorted(history, key=lambda h: h["date"])[-_SENTIMENT_HISTORY_KEEP:]
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"   ⚠ sentiment_history.json 寫入失敗: {e}")


def load_sentiment_history(cache_dir):
    """讀取大盤情緒歷史,給 UI 算趨勢。回傳 list[{date,temp,label}] 由舊到新。"""
    if cache_dir is None:
        return []
    f = Path(cache_dir) / _SENTIMENT_HISTORY_FILE
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════
# 溫度計有效性回測(溫度 vs ^TWII 後續 N 日報酬)
# ══════════════════════════════════════════════════════════════════════
# 溫度區間(lo 含、hi 不含),由熱到冷;與 compute_sentiment 的 label 門檻一致
_SENT_BANDS = [
    ("偏熱 ≥70",    70, 1000),
    ("略偏多 55–69", 55, 70),
    ("中性 45–54",   45, 55),
    ("略偏空 30–44", 30, 45),
    ("偏冷 <30",    -1, 30),
]


def _load_twii_close(cache_dir):
    """讀 ^TWII 收盤序列(index=date, 由舊到新)。失敗回 None。"""
    try:
        files = sorted(Path(cache_dir).glob("twii_*.parquet"))
        if not files:
            return None
        df = pd.read_parquet(files[-1])
        if "date" not in df.columns or "Close" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df["Close"].dropna()
    except Exception as e:
        print(f"   ⚠ 讀取 twii close 失敗: {e}")
        return None


def _twii_forward_return(twii_close, entry_date, n_days: int):
    """^TWII 從 entry_date 起算 n_days 個交易日的報酬率(%)。不足回 None。"""
    idx = twii_close.index.searchsorted(pd.Timestamp(entry_date))
    if idx >= len(twii_close):
        return None
    target = idx + n_days
    if target >= len(twii_close):
        return None
    e = twii_close.iloc[idx]
    t = twii_close.iloc[target]
    if e > 0 and pd.notna(e) and pd.notna(t):
        return (t - e) / e * 100
    return None


def backtest_sentiment(cache_dir, n_days_list=(5, 10)) -> dict:
    """回測「市場溫度」對 ^TWII 後續報酬的預測力。

    對 sentiment_history 內每個歷史溫度,取 ^TWII 後續 N 日報酬,
    依溫度高低分組統計(各組勝率/平均報酬),並算溫度與後續報酬的相關係數。

    Returns:
        {
          "n_days_list": [5, 10],
          "bands": [ {"band","n_total","5d":{"n","win_rate","avg"},"10d":{...}}, ... ],
          "corr":  {5: r, 10: r},      # 溫度 vs 後續報酬 的 Pearson 相關(None=樣本不足)
          "n_eval": {5: 可用樣本數, 10: ...},
        }
        資料不足回 {"error": "..."}。
    """
    hist = load_sentiment_history(cache_dir)
    if not hist or len(hist) < 5:
        return {"error": "情緒歷史不足(至少需 5 日),持續累積後再看。"}

    twii = _load_twii_close(cache_dir)
    if twii is None or twii.empty:
        return {"error": "找不到 ^TWII 收盤資料(twii_*.parquet),無法回測。"}

    # 每筆歷史 → (temp, {n: fwd_return})
    rows = []
    for h in hist:
        temp = h.get("temp")
        date = h.get("date")
        if temp is None or not date:
            continue
        fwd = {n: _twii_forward_return(twii, date, n) for n in n_days_list}
        rows.append({"temp": temp, "fwd": fwd})

    # 分組統計
    bands_out = []
    for name, lo, hi in _SENT_BANDS:
        members = [r for r in rows if lo <= r["temp"] < hi]
        entry = {"band": name, "n_total": len(members)}
        for n in n_days_list:
            vals = [r["fwd"][n] for r in members if r["fwd"][n] is not None]
            if vals:
                wins = sum(1 for v in vals if v > 0)
                entry[f"{n}d"] = {
                    "n": len(vals),
                    "win_rate": wins / len(vals),
                    "avg": sum(vals) / len(vals),
                }
            else:
                entry[f"{n}d"] = None
        bands_out.append(entry)

    # 相關係數(溫度 vs 後續報酬)
    corr, n_eval = {}, {}
    for n in n_days_list:
        pairs = [(r["temp"], r["fwd"][n]) for r in rows if r["fwd"][n] is not None]
        n_eval[n] = len(pairs)
        if len(pairs) >= 3:
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            # 任一序列零變異時 corrcoef 會回 nan → 視為 None
            if len(set(xs)) > 1 and len(set(ys)) > 1:
                r = float(np.corrcoef(xs, ys)[0, 1])
                corr[n] = r if not np.isnan(r) else None
            else:
                corr[n] = None
        else:
            corr[n] = None

    return {
        "n_days_list": list(n_days_list),
        "bands": bands_out,
        "corr": corr,
        "n_eval": n_eval,
    }


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
        🌡️ 溫度計:<b>65 / 100</b> 🌤️ 略偏多
        🟢 VIX 18.2(偏低)
        🟢 台指波動 14.5(歷史 28%位,偏低)
        🟢 加權位階 +5.2%(略高)
        🟡 融資水位 62%位(偏高)
        🟢 外資期貨 +8,234口(偏多)
        🟠 散戶估算 72%(散戶偏多)
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

    v = ind.get("margin_level", {})
    if v.get("pct_rank") is not None:
        lines.append(f"{v['icon']} 融資水位 {int(v['pct_rank'])}%位({v['label']})")

    v = ind.get("fi_futures", {})
    if v.get("value") is not None:
        sign = "+" if v["value"] >= 0 else ""
        lines.append(f"{v['icon']} 外資期貨 {sign}{v['value']:,}口({v['label']})")

    v = ind.get("retail_futures", {})
    if v.get("pct") is not None:
        lines.append(f"{v['icon']} 散戶估算 {int(v['pct'])}%({v['label']})")

    return "\n".join(lines) + "\n"
