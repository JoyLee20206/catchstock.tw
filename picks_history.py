"""多日入選歷史共用模組

讀寫 cache/previous_picks.json,並提供 streak 統計給 UI 熱度榜使用。
向下相容舊的 flat list 格式([sid, sid, ...]),自動轉新格式。
"""
import os
import json


HISTORY_FILE = "cache/previous_picks.json"
HISTORY_DAYS = 7  # 保留最近 N 天


# ══════════════════════════════════════════════════════════════════════
# 讀寫
# ══════════════════════════════════════════════════════════════════════
def load_history() -> list:
    """讀取多日歷史。

    新格式: [{"date": "YYYY-MM-DD", "sids": ["2330", ...]}, ...]  依日期遞增排序
    舊格式: ["2330", ...]                                       自動包成單筆 "legacy"
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data and isinstance(data[0], str):
            # 舊版 flat list → 包成單筆 legacy,下次寫入時會被剔除
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


# ══════════════════════════════════════════════════════════════════════
# 統計
# ══════════════════════════════════════════════════════════════════════
def compute_streak(sid: str, history: list) -> int:
    """計算 sid 從今日往前推的連續入選天數。

    使用前提:history 尚未加入今日的 entry(否則會多算 1 天)。
    回傳值 = 1 表示今天才出現(新進),>= 3 才有持續性意義。
    """
    streak = 1  # 今天本來就含此 sid(呼叫端責任)
    for entry in reversed(history):
        if sid in entry.get("sids", []):
            streak += 1
        else:
            break
    return streak


def compute_hot_picks(history: list, top_n: int = 10) -> list:
    """過去 N 天的入選熱度榜。

    Args:
        history: load_history() 的結果(可含今日,也可不含,皆會正確處理)
        top_n: 回傳前幾名

    Returns:
        [{
            "sid": str,
            "hits": int,            # 期間內總入選天數
            "max_streak": int,      # 期間內最長連續天數
            "active_streak": int,   # 期間結尾的連續天數(若最後一日未入選 = 0)
            "in_latest": bool,      # 是否在最後一筆中(用於 UI 標「★ 今日」)
        }, ...]

        排序鍵: hits desc, max_streak desc, sid asc
    """
    if not history:
        return []

    # 蒐集所有出現過的 sid
    all_sids = set()
    for entry in history:
        all_sids.update(entry.get("sids", []))

    latest_set = set(history[-1].get("sids", [])) if history else set()
    n_entries = len(history)
    results = []

    for sid in all_sids:
        # 把每日入選與否轉成 boolean list
        per_day = [sid in entry.get("sids", []) for entry in history]
        hits = sum(per_day)

        # 期間內最長連續
        max_streak = 0
        cur = 0
        for v in per_day:
            cur = cur + 1 if v else 0
            if cur > max_streak:
                max_streak = cur

        # 期間結尾的連續(從尾往前數)
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
