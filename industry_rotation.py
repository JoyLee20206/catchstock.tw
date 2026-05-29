"""產業輪動追蹤

對比「最近 N 天」vs「再往前 N 天」各產業上榜次數變化,
找出資金正在流入/流出的產業。
"""
from collections import Counter
from picks_history import get_picks


# TWSE 產業分類代碼 → 中文名稱對映(跟 fetch_cache.py 保持一致)
# 為什麼這裡也要有:舊版 picks_history.json 把 OpenAPI 的數字代碼直接存進去,
# 現在讀出來要 normalize 才不會看到「24」「25」之類的純數字 chip。
_TWSE_INDUSTRY_CODE_MAP = {
    "01": "水泥工業",      "02": "食品工業",      "03": "塑膠工業",
    "04": "紡織纖維",      "05": "電機機械",      "06": "電器電纜",
    "08": "玻璃陶瓷",      "09": "造紙工業",      "10": "鋼鐵工業",
    "11": "橡膠工業",      "12": "汽車工業",      "14": "建材營造",
    "15": "航運業",        "16": "觀光事業",      "17": "金融保險",
    "18": "貿易百貨",      "20": "其他業",        "21": "化學工業",
    "22": "生技醫療業",    "23": "油電燃氣業",    "24": "半導體業",
    "25": "電腦及週邊設備業","26": "光電業",      "27": "通信網路業",
    "28": "電子零組件業",  "29": "電子通路業",    "30": "資訊服務業",
    "31": "其他電子業",    "32": "文化創意業",    "33": "農業科技業",
    "34": "電子商務業",    "38": "觀光餐旅",      "39": "居家生活",
    "40": "數位雲端",      "41": "運動休閒",      "80": "管理股票",
}


def _normalize_industry(raw):
    """把舊歷史 picks 的「產業別」標準化為中文名稱。
    純數字代碼 → 查表中文;查不到原樣回傳;已是中文 → 原樣。
    """
    s = (raw or "").strip()
    if not s:
        return ""
    if s.isdigit() and len(s) <= 3:
        return _TWSE_INDUSTRY_CODE_MAP.get(s.zfill(2), s)
    return s


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
                # 用 _normalize_industry 把舊歷史的數字代碼也轉成中文,
                # 同樣產業的不同寫法(例 "24" 與 "半導體業")會自動合併
                ind = _normalize_industry(pick.get("industry"))
                if ind:
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
