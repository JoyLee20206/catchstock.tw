"""多日入選歷史共用模組(v2)

Schema v2:[{"date": "YYYY-MM-DD", "picks": [{"sid", "score", "close", "industry", "sig"?}, ...]}]
           "sig" (選用):{投信,外資,雙買,券,大戶,散戶,技術,KD,營收,RS} 各 0/1,供訊號歸因分析
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
HISTORY_DAYS = 365  # 保留一年:讓績效/歸因/大盤濾網/出場回測能「跨多空」累積。
                    # (原 30 天上限會讓所有回測永遠長不大、跨不過一次大盤回檔)
                    # 熱度榜/輪動只看近期,改由 compute_hot_picks 的 window 參數控制,不受此影響。


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
    df 預期含欄位:代號、總分、現價、產業 + 10 個計分細項旗標。

    新增 "sig":保存 10 個計分細項旗標(0/1),供日後「訊號歸因分析」
    —— 拆解法人/籌碼/技術/基本/大盤各條件對後續報酬的貢獻。
    歷史無法回溯(舊資料沒存),故從導入此版本起開始累積。
    """
    if df is None or df.empty:
        return []
    import pandas as pd

    # 細項旗標 → 穩定 key(欄名可能變動,這裡固定對外 schema)。
    # 8 個欄名固定;KD / 營收 欄名含動態參數,改用前綴比對。
    _STABLE_SIG = {
        "投信": "投信買超", "外資": "外資買超", "雙買": "投信+外資雙買",
        "券": "券相關", "大戶": "400張大戶上升", "散戶": "散戶下降",
        "技術": "技術面", "RS": "RS優於大盤",
    }
    _kd_col  = next((c for c in df.columns if str(c).startswith("KD低檔金叉")), None)
    _rev_col = next((c for c in df.columns if str(c).startswith("連月營收達標")), None)

    def _flag(row, col):
        if col is None or col not in row:
            return None
        v = row.get(col)
        try:
            return int(v) if pd.notna(v) else None
        except (ValueError, TypeError):
            return None

    picks = []
    for _, row in df.iterrows():
        try:
            sid = str(row['代號'])
            score_val = row.get('總分')
            close_val = row.get('現價')
            ind_val   = row.get('產業')
            sig = {k: _flag(row, col) for k, col in _STABLE_SIG.items()}
            sig["KD"] = _flag(row, _kd_col)
            sig["營收"] = _flag(row, _rev_col)
            picks.append({
                "sid": sid,
                "score": int(score_val) if pd.notna(score_val) else None,
                "close": float(close_val) if pd.notna(close_val) else None,
                "industry": str(ind_val) if pd.notna(ind_val) and str(ind_val).strip() else None,
                "sig": sig,
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


def compute_hot_picks(history: list, top_n: int = 10, window: int | None = None) -> list:
    """過去 N 天的入選熱度榜。

    Args:
        window: 只看最近 N 個交易日(None=全部)。保留歷史可達一年,
                但熱度榜應反映「近期」強勢,故 UI 會傳入較短視窗(如 20)。

    Returns:
        [{
            "sid", "hits", "max_streak", "active_streak",
            "in_latest": bool,
            "total_days": int,   # = 實際納入計算的天數(套用 window 後)
        }, ...]
    """
    if not history:
        return []
    if window is not None and window > 0:
        history = history[-window:]

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
