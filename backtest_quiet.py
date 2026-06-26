# -*- coding: utf-8 -*-
# ============================================================
# backtest_quiet.py — 「籌碼沉澱 / 冷門股」規則的『對照組回測』(自適應版)
# ------------------------------------------------------------
# 沿用 backtest.py 的進出場引擎(訊號日隔日開盤 + 0.1% 滑價進場、持有 N 個交易日
# 後收盤出場、同股去重)。三~四組用相同再平衡日、相同進出場,只差「進場條件」。
#
# ★ 自適應(這版的重點):
#   - 每次跑都讀「最新累積到的」daily / holders 快取,並在開頭印出資料涵蓋範圍。
#     雲端資料愈積愈長,這支不用改就自動用更長的歷史跑。
#   - 「散戶連續減少」那一關會【資料夠才自動納入】:holders 累積到足夠週數、
#     且交集後交易筆數夠(≥ MIN_RETAIL_TRADES)才把它加成第 4 關;否則自動退回
#     3 關(量縮+站穩月線+流動性)並印出原因。判斷邏輯全自動,不用手動切換。
#
# 散戶定義與主程式 screening0515.py 一致:SMALL_HOLDER_LEVELS=[2,3,4] 比例加總,
# 近 3 週累計變化 ≤ -0.15% 視為「散戶在減少」。
#
# ⚠ 限制:本機 daily 約 116 個交易日、holders 僅 5 週 → 散戶關通常會自動跳過。
#         把資料夾換成雲端累積較長的快取,散戶關就會自己亮起來。
# ============================================================
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from backtest import _load_daily_matrix, compute_signal_returns, summarize, SLIPPAGE_PCT

CACHE = Path("cache")

# ---------------- 價量參數 ----------------
MA_PERIOD        = 20      # 月線
STABLE_DAYS      = 3       # 站穩 = 最近幾日收盤都 ≥ MA20
MAX_ABOVE_MA_PCT = 20.0    # 不追高(對齊 screening0515.py)
MIN_AVG_LOT      = 500     # 日均量門檻(張)
WEEK_DAYS        = 5       # 一週約 5 個交易日
VOL_DOWN_WEEKS   = 3       # 週量連續下滑幾「週」(用 5 日塊代理)
# ---------------- 散戶參數(與 screening0515.py 一致) ----------------
SMALL_HOLDER_LEVELS    = [2, 3, 4]
HOLDER_LOOKBACK_WEEKS  = 4         # 4 週才能算「近 3 週累計變化」
SMALL_3W_CHANGE_MAX    = -0.15     # 散戶近 3 週累計持股比例變化 ≤ 此值
MIN_HOLDER_WEEKS       = 6         # holders 至少要這麼多「不同週」才考慮納入散戶關
MIN_RETAIL_TRADES      = 20        # 加散戶關後,最長持有期至少要這麼多筆才算可信
# ---------------- 回測參數 ----------------
HOLD_HORIZONS    = [20, 40]   # 持有交易日數:≈1個月、≈2個月
REBAL_EVERY      = 5          # 每隔幾個交易日取一次訊號(降低重疊、加速)
# --------------------------------------------------


def build_condition_matrices(close, volume):
    """回 (站穩月線&不追高, 流動性, 量縮) 三個 bool 矩陣 (date×stock)。"""
    ma = close.rolling(MA_PERIOD).mean()
    above = (close >= ma)
    stable = above.rolling(STABLE_DAYS).min() == 1            # 連 STABLE_DAYS 日站上
    not_chase = close <= ma * (1 + MAX_ABOVE_MA_PCT / 100.0)  # 不追高
    base_trend = stable & not_chase

    avg_lot = (volume / 1000.0).rolling(20).mean()
    liquid = avg_lot >= MIN_AVG_LOT

    wk = (volume / 1000.0).rolling(WEEK_DAYS).mean()
    vol_down = pd.DataFrame(True, index=close.index, columns=close.columns)
    for k in range(VOL_DOWN_WEEKS):
        vol_down &= (wk.shift(k * WEEK_DAYS) < wk.shift((k + 1) * WEEK_DAYS))

    return base_trend, liquid, vol_down


def load_retail_decline_daily(close, cache_dir=None):
    """讀 holders,建『散戶比例近 3 週累計下降』的 daily bool 矩陣(週訊號 ffill 到每日)。
    回 (daily_bool_or_None, 不同週數)。定義同 screening0515.py(levels [2,3,4])。"""
    base = Path(cache_dir) if cache_dir is not None else CACHE
    files = sorted(base.glob("holders_*.parquet"))
    if not files:
        return None, 0
    df = pd.read_parquet(files[-1])
    if df.empty or "HoldingSharesLevel" not in df.columns:
        return None, 0
    df["date"] = pd.to_datetime(df["date"])
    df["stock_id"] = df["stock_id"].astype(str)
    df["HoldingSharesLevel"] = pd.to_numeric(df["HoldingSharesLevel"], errors="coerce")
    df["percent"] = pd.to_numeric(
        df["percent"].astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce"
    )
    df = df.dropna(subset=["HoldingSharesLevel", "percent"])
    small = df[df["HoldingSharesLevel"].isin(SMALL_HOLDER_LEVELS)]
    if small.empty:
        return None, 0

    wk = small.groupby(["date", "stock_id"])["percent"].sum().unstack().sort_index()
    n_weeks = len(wk.index)
    if n_weeks < HOLDER_LOOKBACK_WEEKS:
        return None, n_weeks

    # 近 3 週累計變化 = 本週 - (3 週前) ;不足 3 週前的列為 NaN → 視為無訊號
    change = wk - wk.shift(HOLDER_LOOKBACK_WEEKS - 1)
    decl_weekly = change < SMALL_3W_CHANGE_MAX

    # 週訊號 ffill 到每日:每個交易日對應「≤ 它的最近一個 holders 週」(無前視)
    decl = decl_weekly.reindex(columns=close.columns)
    decl_daily = decl.reindex(close.index, method="ffill").fillna(False).astype(bool)
    return decl_daily, n_weeks


def rebal_mask(index):
    keep = np.zeros(len(index), dtype=bool)
    keep[::REBAL_EVERY] = True
    return pd.Series(keep, index=index)


def backtest(cache_dir=None):
    """跑對照組回測,回結構化結果(供 UI / 其他程式 import)。
    回 dict:{meta:{...}, horizons:[{hold,months,rows,verdict}], primary_name} 或 {"error":...}。"""
    cd = Path(cache_dir) if cache_dir is not None else CACHE
    m = _load_daily_matrix(cd)
    if m is None:
        return {"error": "讀不到 daily 快取"}
    close, volume, open_ = m["close"], m["volume"], m.get("open")

    etf = [c for c in close.columns if str(c).startswith("00")]
    if etf:
        close = close.drop(columns=etf)
        volume = volume.drop(columns=[c for c in etf if c in volume.columns])
        if open_ is not None:
            open_ = open_.drop(columns=[c for c in etf if c in open_.columns])

    base_trend, liquid, vol_down = build_condition_matrices(close, volume)
    rmask = rebal_mask(close.index)
    on_rebal = lambda mat: mat.where(rmask, other=False)

    # ── 散戶關:逐持有期分別判定 ──
    #   先看 holders 週數夠不夠(不夠則所有持有期都不可能納入);
    #   再「對每個持有期」各自用該期的樣本筆數決定要不要納入散戶關。
    #   原因:愈長的持有期愈難湊樣本(要往前留更多未來日),
    #   逐期判定可讓短持有期(20日)先納入,不必等 40 日也湊滿。
    retail_daily, n_weeks = load_retail_decline_daily(close, cache_dir)
    holders_ok = retail_daily is not None and n_weeks >= MIN_HOLDER_WEEKS

    horizons = []
    any_active = False
    for hold in HOLD_HORIZONS:
        # 1) 本持有期是否納入散戶關
        retail_active, retail_note = False, ""
        if not holders_ok:
            retail_note = f"散戶關跳過:holders 僅 {n_weeks} 週(需 ≥ {MIN_HOLDER_WEEKS})"
        else:
            retail_mat = on_rebal(base_trend & liquid & vol_down & retail_daily)
            n_r = len(compute_signal_returns(retail_mat, close, hold,
                                             dedup_within_hold=True, open_matrix=open_))
            if n_r >= MIN_RETAIL_TRADES:
                retail_active = True
                retail_note = f"散戶關納入 ✅(本期樣本 {n_r} 筆 ≥ {MIN_RETAIL_TRADES})"
            else:
                retail_note = f"散戶關跳過:本期僅 {n_r} 筆(< {MIN_RETAIL_TRADES})樣本太少"
        any_active = any_active or retail_active

        # 2) 依本期是否納入散戶,組出比較組
        if retail_active:
            primary_name = "訊號組(4關:+散戶↓)"
            groups = [
                (primary_name,                on_rebal(base_trend & liquid & vol_down & retail_daily)),
                ("參考·3關(沒散戶)",          on_rebal(base_trend & liquid & vol_down)),
                ("對照A(全市場/流動性)",       on_rebal(liquid)),
                ("對照B(站穩月線·沒量縮)",     on_rebal(base_trend & liquid & ~vol_down)),
            ]
        else:
            primary_name = "訊號組(3關:量縮+站穩月線)"
            groups = [
                (primary_name,                on_rebal(base_trend & liquid & vol_down)),
                ("對照A(全市場/流動性)",       on_rebal(liquid)),
                ("對照B(站穩月線·沒量縮)",     on_rebal(base_trend & liquid & ~vol_down)),
            ]

        # 3) 算各組統計
        rows, stats_by = [], {}
        for name, mat in groups:
            tr = compute_signal_returns(mat, close, hold, dedup_within_hold=True, open_matrix=open_)
            st = summarize(tr, hold_days=hold)
            stats_by[name] = st
            if st["n"] == 0:
                rows.append({"組別": name, "筆數": 0}); continue
            rows.append({
                "組別": name, "筆數": st["n"],
                "勝率%": round(st["win_rate"] * 100, 1),
                "平均報酬%": round(st["avg_return"], 2),
                "中位數%": round(st["median_return"], 2),
                "波動(std)": round(st["std"], 2),
                "最大回檔%": round(st["mdd"], 1),
            })
        verdict = None
        s, a = stats_by.get(primary_name), stats_by.get("對照A(全市場/流動性)")
        if s and a and s["n"] and a["n"]:
            dwin = (s["win_rate"] - a["win_rate"]) * 100
            dret = s["avg_return"] - a["avg_return"]
            verdict = {
                "win_delta_pp": round(dwin, 1),
                "ret_delta_pp": round(dret, 2),
                "beats_market": bool(dret > 0 and dwin > 0),
                "signal_win": round(s["win_rate"] * 100, 1),
                "market_win": round(a["win_rate"] * 100, 1),
            }
        horizons.append({"hold": hold, "months": round(hold / 20, 1),
                         "rows": rows, "verdict": verdict,
                         "retail_active": retail_active, "retail_note": retail_note,
                         "primary_name": primary_name})

    if not holders_ok:
        retail_msg = f"holders 僅 {n_weeks} 週(< {MIN_HOLDER_WEEKS})→ 各持有期一律只測 3 關"
    else:
        retail_msg = (f"holders {n_weeks} 週(≥ {MIN_HOLDER_WEEKS} ✅)→ 散戶關「逐持有期」判定,"
                      f"各期樣本夠才納入(見下方各表上方標註)")
    meta = {
        "asof": close.index.max().strftime("%Y-%m-%d"),
        "start": close.index.min().strftime("%Y-%m-%d"),
        "n_days": len(close.index),
        "slippage": SLIPPAGE_PCT,
        "retail_any_active": any_active,
        "n_weeks": n_weeks,
        "retail_msg": retail_msg,
    }
    return {"meta": meta, "horizons": horizons}


def run():
    res = backtest()
    if "error" in res:
        print(res["error"]); sys.exit(1)
    mt = res["meta"]
    print(f">>> 籌碼沉澱規則 對照組回測(自適應) | daily {mt['start']} ~ {mt['asof']} "
          f"({mt['n_days']} 交易日) | 滑價 {mt['slippage']}% | 隔日開盤進場")
    print(f">>> {mt['retail_msg']}\n")
    for h in res["horizons"]:
        print(f"━━━ 持有 {h['hold']} 交易日(≈{h['months']} 個月) | {h['retail_note']} ━━━")
        print(pd.DataFrame(h["rows"]).to_string(index=False))
        v = h["verdict"]
        if v:
            verdict = "有贏過大盤" if v["beats_market"] else "沒有明顯贏過大盤"
            print(f"  → {h['primary_name']} vs 大盤:勝率 {v['win_delta_pp']:+.1f}pp、"
                  f"平均報酬 {v['ret_delta_pp']:+.2f}pp → {verdict}")
            print(f"     (文章宣稱兩個月勝率 68%;訊號組 {v['signal_win']:.1f}%、大盤基準 {v['market_win']:.1f}%)")
        print()


if __name__ == "__main__":
    run()
