"""策略績效追蹤

對 picks_history 內的每筆選股,從 daily parquet 拉出後續 N 日 close,
算出「入選後 N 日報酬」、「勝率」、「平均報酬」、「最大回檔」等指標。

提供 UI「策略績效」分頁需要的 dataframe。
"""
import pandas as pd
from picks_history import get_picks


# ══════════════════════════════════════════════════════════════════════
# 載入 close 矩陣(優化:一次讀檔,所有 pick 共用)
# ══════════════════════════════════════════════════════════════════════
def _load_close_matrix(cache_dir):
    """讀最新 daily parquet,組成 close pivot table。

    Returns:
        DataFrame index=date(Timestamp), columns=stock_id(str), values=close
        失敗回 None
    """
    try:
        files = sorted(cache_dir.glob('daily_*.parquet'))
        if not files:
            return None
        df = pd.read_parquet(files[-1], columns=['stock_id', 'date', 'close'])
        if df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'])
        df['stock_id'] = df['stock_id'].astype(str)
        # 同一 (date, stock_id) 可能有重複(上市/上櫃撞號),取最後一筆
        df = df.drop_duplicates(subset=['date', 'stock_id'], keep='last')
        return df.pivot(index='date', columns='stock_id', values='close')
    except Exception as e:
        print(f"⚠ 讀取 close 矩陣失敗: {e}")
        return None


def _forward_return(close_matrix, sid: str, entry_date, n_days: int):
    """算「entry_date 後 n_days 個交易日的報酬率(%)」。

    若 entry_date 不在矩陣 / 該股不在矩陣 / 後續天數不足,回 None。
    """
    if close_matrix is None or sid not in close_matrix.columns:
        return None
    sid_series = close_matrix[sid].dropna()
    if sid_series.empty:
        return None

    # 找 entry_date 對應的索引位置(用 searchsorted 找最接近的交易日)
    entry_ts = pd.Timestamp(entry_date)
    dates = sid_series.index
    # 取「>= entry_date 的第一個交易日」
    idx_arr = dates.searchsorted(entry_ts)
    if idx_arr >= len(dates):
        return None
    entry_idx = idx_arr
    target_idx = entry_idx + n_days
    if target_idx >= len(dates):
        return None  # 還沒到 N 日後

    entry_close = sid_series.iloc[entry_idx]
    target_close = sid_series.iloc[target_idx]
    if entry_close <= 0 or pd.isna(entry_close) or pd.isna(target_close):
        return None
    return (target_close - entry_close) / entry_close * 100


# ══════════════════════════════════════════════════════════════════════
# 主要計算函式
# ══════════════════════════════════════════════════════════════════════
def compute_performance(history: list, cache_dir, n_days_list=(5, 10, 20)) -> dict:
    """對歷史所有 pick 算 N 日後績效。

    Args:
        history: load_history() 結果
        cache_dir: pathlib.Path 指向 CACHE_DIR
        n_days_list: 要算的後續日數(交易日)

    Returns:
        {
            "samples": [...],         # 每筆 pick + 後續報酬,給 UI 直方圖
            "overall": {"n":..., "win_rate_5d":..., "avg_return_5d":..., ...},
            "by_score": {7: {...}, 8: {...}, 9: {...}, 10: {...}},
            "summary_text": "..."     # 一行摘要給 TG 用(可選)
        }
        資料不足回 {"samples": [], "overall": {}, "by_score": {}}
    """
    close_matrix = _load_close_matrix(cache_dir)
    if close_matrix is None:
        return {"samples": [], "overall": {}, "by_score": {}, "error": "無法讀取 daily 快取"}

    samples = []
    for entry in history:
        if entry.get("date") == "legacy":
            continue  # 沒日期的資料不能算績效
        try:
            entry_date = pd.Timestamp(entry["date"])
        except Exception:
            continue
        for pick in get_picks(entry):
            sid = str(pick.get("sid", ""))
            score = pick.get("score")
            if not sid:
                continue
            row = {"date": entry["date"], "sid": sid, "score": score}
            for n in n_days_list:
                row[f"return_{n}d"] = _forward_return(close_matrix, sid, entry_date, n)
            samples.append(row)

    # ── 整體統計 ──
    overall = {}
    if samples:
        for n in n_days_list:
            valid = [s[f"return_{n}d"] for s in samples if s[f"return_{n}d"] is not None]
            if valid:
                wins = sum(1 for v in valid if v > 0)
                overall[f"n_{n}d"] = len(valid)
                overall[f"win_rate_{n}d"] = wins / len(valid)
                overall[f"avg_return_{n}d"] = sum(valid) / len(valid)
                overall[f"median_return_{n}d"] = sorted(valid)[len(valid) // 2]
                overall[f"max_return_{n}d"] = max(valid)
                overall[f"min_return_{n}d"] = min(valid)

    # ── 分數區間統計 ──
    by_score = {}
    for score in sorted({s["score"] for s in samples if s["score"] is not None}):
        score_samples = [s for s in samples if s["score"] == score]
        stat = {"n_picks": len(score_samples)}
        for n in n_days_list:
            valid = [s[f"return_{n}d"] for s in score_samples if s[f"return_{n}d"] is not None]
            if valid:
                wins = sum(1 for v in valid if v > 0)
                stat[f"n_{n}d"] = len(valid)
                stat[f"win_rate_{n}d"] = wins / len(valid)
                stat[f"avg_return_{n}d"] = sum(valid) / len(valid)
        by_score[score] = stat

    return {"samples": samples, "overall": overall, "by_score": by_score}


def format_performance_summary(perf: dict) -> str:
    """產生一行 TG 用的績效摘要。"""
    o = perf.get("overall", {})
    if not o or "win_rate_5d" not in o:
        return ""
    return (
        f"📊 近 30 日策略績效(5 日後):"
        f"勝率 {o['win_rate_5d']*100:.0f}% / "
        f"平均報酬 {o['avg_return_5d']:+.2f}% / "
        f"樣本 {o['n_5d']} 筆"
    )
