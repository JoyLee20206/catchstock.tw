"""多日入選歷史共用模組(v2)

Schema v2:[{"date": "YYYY-MM-DD", "picks": [{"sid", "score", "close", "industry"}, ...]}]
Schema v1:[{"date": "YYYY-MM-DD", "sids": ["2330", ...]}]  ← 向下相容
Legacy:   ["2330", ...]                                    ← 向下相容

新增 close / score / industry 欄位後可支援:
- 策略績效追蹤(performance.py 需要 close)
- 產業輪動(industry_rotation.py 需要 industry)
- 分數區間勝率分析(需要 score)

寫入永遠用 v2;讀取自動偵測格式並提供統一存取介面。
"""
import os
import json


HISTORY_FILE = "cache/previous_picks.json"
HISTORY_DAYS = 30  # 績效追蹤需要 N 天歷史,擴大到 30 天(原 7 天)


# ══════════════════════════════════════════════════════════════════════
# 統一存取介面(處理 v1/v2/legacy 三種格式)
# ══════════════════════════════════════════════════════════════════════
def get_sids(entry: dict) -> list:
    """從 entry 提取 sid 字串清單,自動處理新舊格式。"""
    if "picks" in entry:
        return [str(p["sid"]) for p in entry["picks"] if "sid" in p]
    return [str(s) for s in entry.get("sids", [])]


def get_picks(entry: dict) -> list:
    """從 entry 提取 picks dict 清單(含 score/close/industry)。
    若 entry 是舊版只有 sids,回傳 [{"sid": ..., "score": None, ...}] 補空。
    """
    if "picks" in entry:
        return entry["picks"]
    return [{"sid": str(s), "score": None, "close": None, "industry": None}
            for s in entry.get("sids", [])]


# ══════════════════════════════════════════════════════════════════════
# 讀寫
# ══════════════════════════════════════════════════════════════════════
def load_history() -> list:
    """讀取多日歷史。回傳混合 v1/v2 entries 也保留原樣,讀取端用 get_sids/get_picks 抽取。"""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data and isinstance(data[0], str):
            # legacy flat list → 包成單筆,下次寫入會被剔除
            return [{"date": "legacy", "sids": data}]
        if isinstance(data, list):
            return data
    except Exception as e:
        print(f"讀取歷史名單失敗,忽略比對: {e}")
    return []


def save_history(history: list) -> None:
    """寫回歷史,只保留最近 HISTORY_DAYS 天。"""
    os.makedirs("cache", exist_ok=True)
    trimmed = history[-HISTORY_DAYS:]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, ensure_ascii=False, indent=2)


def build_picks_from_df(df) -> list:
    """從 screening DataFrame 抽出 picks list(v2 schema)。
    df 預期含欄位:代號、總分、現價、產業
    """
    if df is None or df.empty:
        return []
    import pandas as pd
    picks = []
    for _, row in df.iterrows():
        try:
            sid = str(row['代號'])
            score_val = row.get('總分')
            close_val = row.get('現價')
            ind_val   = row.get('產業')
            picks.append({
                "sid": sid,
                "score": int(score_val) if pd.notna(score_val) else None,
                "close": float(close_val) if pd.notna(close_val) else None,
                "industry": str(ind_val) if pd.notna(ind_val) and str(ind_val).strip() else None,
            })
        except Exception:
            continue
    return picks


# ══════════════════════════════════════════════════════════════════════
# 統計
# ══════════════════════════════════════════════════════════════════════
def compute_streak(sid: str, history: list) -> int:
    """計算 sid 從今日往前推的連續入選天數。
    使用前提:history 尚未加入今日 entry。
    """
    streak = 1  # 今天本來就含此 sid(呼叫端責任)
    for entry in reversed(history):
        if sid in get_sids(entry):
            streak += 1
        else:
            break
    return streak


def compute_hot_picks(history: list, top_n: int = 10) -> list:
    """過去 N 天的入選熱度榜。

    Returns:
        [{
            "sid", "hits", "max_streak", "active_streak",
            "in_latest": bool,
            "total_days": int,
        }, ...]
    """
    if not history:
        return []

    all_sids = set()
    for entry in history:
        all_sids.update(get_sids(entry))

    latest_set = set(get_sids(history[-1])) if history else set()
    n_entries = len(history)
    results = []

    for sid in all_sids:
        per_day = [sid in get_sids(entry) for entry in history]
        hits = sum(per_day)

        max_streak = 0
        cur = 0
        for v in per_day:
            cur = cur + 1 if v else 0
            if cur > max_streak:
                max_streak = cur

        active = 0
        for v in reversed(per_day):
            if v:
                active += 1
            else:
                break

        results.append({
            "sid": sid,
            "hits": hits,
            "max_streak": max_streak,
            "active_streak": active,
            "in_latest": sid in latest_set,
            "total_days": n_entries,
        })

    results.sort(key=lambda x: (-x["hits"], -x["max_streak"], x["sid"]))
    return results[:top_n]
