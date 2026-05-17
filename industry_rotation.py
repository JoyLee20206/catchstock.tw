"""產業輪動追蹤

對比「最近 N 天」vs「再往前 N 天」各產業上榜次數變化,
找出資金正在流入/流出的產業。
"""
from collections import Counter
from picks_history import get_picks


def compute_industry_rotation(history: list, recent_days: int = 7, prev_days: int = 7) -> list:
    """產業熱度變化排行。

    Args:
        history: load_history() 結果(依日期遞增排序)
        recent_days: 最近 N 天當「現在」
        prev_days: 再往前 N 天當「對比基期」

    Returns:
        [{
            "industry": str,
            "recent_count": int,    # 最近 N 天該產業上榜總次數
            "prev_count": int,      # 對比期該產業上榜總次數
            "change": int,          # recent - prev
            "direction": "up"|"down"|"flat",
        }, ...]
        依 abs(change) 由大到小排序
    """
    if not history:
        return []

    # 過濾 legacy(沒日期的)
    dated = [h for h in history if h.get("date") != "legacy"]
    if len(dated) < 2:
        return []

    # 切兩段:最近 recent_days 天 vs 再前 prev_days 天
    recent_slice = dated[-recent_days:]
    prev_slice   = dated[-(recent_days + prev_days):-recent_days] if len(dated) > recent_days else []

    def count_industries(entries):
        c = Counter()
        for e in entries:
            for pick in get_picks(e):
                ind = pick.get("industry")
                if ind and isinstance(ind, str) and ind.strip():
                    c[ind] += 1
        return c

    recent_c = count_industries(recent_slice)
    prev_c   = count_industries(prev_slice)

    all_industries = set(recent_c.keys()) | set(prev_c.keys())
    results = []
    for ind in all_industries:
        r = recent_c.get(ind, 0)
        p = prev_c.get(ind, 0)
        change = r - p
        if change > 2:
            direction = "up"
        elif change < -2:
            direction = "down"
        else:
            direction = "flat"
        results.append({
            "industry": ind,
            "recent_count": r,
            "prev_count": p,
            "change": change,
            "direction": direction,
        })

    # 排序:abs(change) desc;若 change=0 則 recent_count desc
    results.sort(key=lambda x: (-abs(x["change"]), -x["recent_count"]))
    return results
