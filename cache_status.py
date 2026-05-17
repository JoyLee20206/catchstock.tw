"""cache 新鮮度判斷共用模組

UI 與 Telegram 共用同一套判斷邏輯,避免兩邊顯示不一致
(網頁紅燈但 TG 沒事、或 TG 警告但網頁說 OK)。
"""
import pandas as pd


def cache_freshness(cache_date) -> dict:
    """判斷 cache 新鮮度,回傳結構化結果。

    Args:
        cache_date: cache 最新一筆資料的日期(pd.Timestamp / datetime / None / NaT)

    Returns:
        dict with keys:
        - level: "missing" | "ok" | "info" | "warn" | "error"
        - msg:   給人看的中文訊息(一句話)
        - days:  距今天差幾天(int);若 cache_date 不合法則為 None

    Level 對照:
        missing — cache_date 為 None / NaT(找不到 cache 檔)
        ok      — 當天資料或週末延遲 ≤ 2 天
        info    — 1-3 天,可能是國定假日
        warn    — 4-5 天,可能 fetch 排程出問題
        error   — ≥ 6 天,需要立即處理
    """
    if cache_date is None or pd.isna(cache_date):
        return {"level": "missing", "msg": "找不到 daily 快取,請先執行 fetch_cache.py", "days": None}

    now_tpe = pd.Timestamp.now(tz="Asia/Taipei").replace(tzinfo=None)
    today_naive = now_tpe.normalize()

    cache_dt = pd.Timestamp(cache_date)
    if cache_dt.tz is not None:
        cache_dt = cache_dt.tz_localize(None)
    cache_dt = cache_dt.normalize()

    age = (today_naive - cache_dt).days
    date_str = cache_dt.strftime("%Y-%m-%d")
    is_weekend = today_naive.weekday() >= 5  # 5=週六, 6=週日
    now_hour = now_tpe.hour

    # 當天資料
    if age == 0:
        if now_hour < 15:
            return {
                "level": "info",
                "msg": f"目前資料為 {date_str}。今日盤後數據預計 15:30 自動更新。",
                "days": 0,
            }
        return {
            "level": "ok",
            "msg": f"cache 最新日期: {date_str}(今日最新數據)",
            "days": 0,
        }

    # 週末延遲(週五收盤後到週日,age 會是 1~2)
    if is_weekend and age <= 2:
        return {
            "level": "ok",
            "msg": f"cache 最新日期: {date_str}(週末未開盤,此已為最新交易日)",
            "days": age,
        }

    # 平日 1~3 天(可能是國定假日)
    if age <= 3:
        return {
            "level": "info",
            "msg": f"cache 最新日期: {date_str}({age} 天前,若遇國定假日未開盤即為最新資料)",
            "days": age,
        }

    # 4-5 天(警告)
    if age <= 5:
        return {
            "level": "warn",
            "msg": f"cache 最新日期: {date_str}({age} 天前,可能 fetch_cache 排程失敗)",
            "days": age,
        }

    # ≥ 6 天(嚴重)
    return {
        "level": "error",
        "msg": f"cache 最新日期: {date_str}({age} 天前)⚠ 過舊,請立即更新",
        "days": age,
    }
