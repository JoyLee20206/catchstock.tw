# -*- coding: utf-8 -*-
# ============================================================
# screening_quiet.py — 「籌碼沉澱 / 冷門股」篩選
# ------------------------------------------------------------
# 出處:某篇選股邏輯文章。把它的規則落地成可執行篩選(僅篩選,不含回測)。
# 同時符合以下四關才入選:
#   1. 週轉率連續 3 週下滑   → 換手降溫(短線客退場)
#   2. 散戶持股比例連續減少   → 浮額被洗(沿用主程式 TDCC 做法)
#   3. 站穩月線(收盤 ≥ MA20) → 沒人亂殺、底部撐住
#   4. 日均量 ≥ 500 張        → 量能足夠、數據不易被操縱
#
# ⚠ 關於「週轉率」的代理做法(很重要,看一下):
#   真週轉率 = 成交量 ÷ 流通在外股數。本專案的快取沒有「流通股數」欄位,
#   但對「同一檔股票」來說,流通股數在幾週內幾乎是常數,除以常數不會改變
#   「上升還是下降」的方向 → 所以「週轉率連 3 週下滑」== 「週成交量連 3 週下滑」。
#   本程式即以「週平均成交量的連續下滑」代理週轉率趨勢,趨勢判讀完全等價。
#   (只有跨股比較絕對週轉率時才需要真流通股數,本規則不需要。)
#
# ⚠ 關於「散戶人數減少」:
#   集保 TDCC 分散表只有「各持股分級的占股比例(percent)」,沒有「人數」欄位。
#   主程式 screening.py 一向用「散戶分級的持股比例下降」當代理,這裡完全沿用,
#   保持與你既有工具一致(SMALL_HOLDER_LEVELS = [2,3,4])。
# ============================================================
import glob
import pandas as pd
from datetime import datetime, timedelta, timezone

TPE_TZ = timezone(timedelta(hours=8))
CACHE = "cache"

# ---------------- 參數(想調就改這裡) ----------------
VOL_DOWN_WEEKS      = 3        # 週轉率(週量)要連續下滑幾週
STABLE_DAYS         = 3        # 「站穩月線」= 最近幾個交易日收盤都 ≥ MA20
MA_PERIOD           = 20       # 月線
MAX_ABOVE_MA_PCT    = 20.0     # 不追高:收盤高出月線超過此 % 視為過熱,剔除
                               #   (對齊主程式 screening0515.py 的 CHASE_ABOVE_MA20_MAX_PCT=20;
                               #    也才符合文章「貼著月線的冷門股」精神,排除先暴衝完才縮量的)
MIN_AVG_LOT         = 500      # 日均量門檻(張),近 20 日均量
SMALL_HOLDER_LEVELS = [2, 3, 4]    # 散戶分級(與 screening.py 一致;Level 1 零股雜訊不納入)
HOLDER_MIN_WEEKS    = 4        # 散戶要算「近 3 週累計變化」至少需 4 週資料
SMALL_3W_CHANGE_MAX = -0.15    # 散戶近 3 週累計持股比例變化 ≤ 此值(越負=跑越多)
SMALL_1W_CHANGE_MAX = -0.05    # Fallback:資料不足 4 週時改用 1 週比較
EXCLUDE_ETF         = True     # 排除 00 開頭的 ETF(策略針對個股籌碼)
# -----------------------------------------------------


def _latest(pattern, cache_dir=None):
    base = str(cache_dir) if cache_dir is not None else CACHE
    fs = sorted(glob.glob(f"{base}/{pattern}"))
    if not fs:
        raise FileNotFoundError(f"找不到快取:{pattern}")
    return fs[-1]


def load_data(cache_dir=None):
    daily = pd.read_parquet(_latest("daily_*.parquet", cache_dir))
    daily["date"] = pd.to_datetime(daily["date"])
    daily["stock_id"] = daily["stock_id"].astype(str)
    daily = daily.sort_values(["stock_id", "date"])

    holders = pd.read_parquet(_latest("holders_*.parquet", cache_dir))
    holders["stock_id"] = holders["stock_id"].astype(str)
    holders["HoldingSharesLevel"] = pd.to_numeric(holders["HoldingSharesLevel"], errors="coerce")
    holders["percent"] = pd.to_numeric(
        holders["percent"].astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )
    holders = holders.dropna(subset=["HoldingSharesLevel", "percent"])

    try:
        info = pd.read_parquet(_latest("info_*.parquet", cache_dir))
        info["stock_id"] = info["stock_id"].astype(str)
        name_map = dict(zip(info["stock_id"], info["stock_name"]))
    except FileNotFoundError:
        name_map = {}
    return daily, holders, name_map


def small_holder_change(holders):
    """每檔散戶持股比例的「近 3 週累計變化」(資料不足則退化成 1 週)。
    回 {stock_id: (change, weeks_used, latest_pct)}。change<0 = 散戶在減少。"""
    out = {}
    small = holders[holders["HoldingSharesLevel"].isin(SMALL_HOLDER_LEVELS)]
    for sid, g in small.groupby("stock_id"):
        s = g.groupby("date")["percent"].sum().sort_index()
        if len(s) >= HOLDER_MIN_WEEKS:
            change = s.iloc[-1] - s.iloc[-HOLDER_MIN_WEEKS]   # 4 週前→本週 = 3 週累計
            weeks = HOLDER_MIN_WEEKS - 1
        elif len(s) >= 2:
            change = s.iloc[-1] - s.iloc[-2]
            weeks = 1
        else:
            continue
        out[sid] = (round(float(change), 3), weeks, round(float(s.iloc[-1]), 2))
    return out


def weekly_volume_declining(g):
    """g = 單檔 daily(已按日期排序)。回 (是否連續下滑, 最近數週均量list)。
    以『週平均日成交量(張)』連續下滑代理週轉率下滑。用週平均→可吃部分週。"""
    v = g.set_index("date")["Trading_Volume"] / 1000.0   # 股→張
    wk = v.resample("W-FRI").mean().dropna()              # 每週平均日量
    if len(wk) < VOL_DOWN_WEEKS + 1:
        return False, []
    last = wk.iloc[-(VOL_DOWN_WEEKS + 1):]                # 取 N+1 週看 N 個下滑
    declining = all(last.iloc[i] > last.iloc[i + 1] for i in range(len(last) - 1))
    return declining, [round(float(x), 1) for x in last]


def screen(cache_dir=None):
    """跑篩選,回 (df, meta)。可被 UI / 其他程式 import 呼叫。
    df 可能為空 DataFrame;meta 含 asof / holders_date / n / conditions 文字。"""
    daily, holders, name_map = load_data(cache_dir)
    asof = daily["date"].max().strftime("%Y-%m-%d")
    holders_date = str(pd.to_datetime(holders["date"]).max().date())

    small_chg = small_holder_change(holders)
    rows = []
    for sid, g in daily.groupby("stock_id"):
        if EXCLUDE_ETF and sid.startswith("00"):
            continue
        if len(g) < MA_PERIOD + 5:
            continue

        closes = g["close"].astype(float)
        vols_lot = g["Trading_Volume"].astype(float) / 1000.0

        # 關 4:日均量 ≥ 門檻(近 20 日)
        avg_lot = float(vols_lot.tail(20).mean())
        if avg_lot < MIN_AVG_LOT:
            continue

        # 關 3:站穩月線(最近 STABLE_DAYS 日收盤都 ≥ MA20)且不過熱(不追高)
        ma20 = float(closes.tail(MA_PERIOD).mean())
        recent_closes = closes.tail(STABLE_DAYS)
        if not (recent_closes >= ma20).all():
            continue
        if (float(closes.iloc[-1]) / ma20 - 1) * 100 > MAX_ABOVE_MA_PCT:
            continue   # 離月線太遠 = 已暴衝過,非「貼著線的冷門股」

        # 關 1:週轉率(週量)連續下滑
        declining, wk_vols = weekly_volume_declining(g)
        if not declining:
            continue

        # 關 2:散戶比例連續減少
        sc = small_chg.get(sid)
        if sc is None:
            continue
        change, weeks, latest_pct = sc
        thr = SMALL_3W_CHANGE_MAX if weeks == 3 else SMALL_1W_CHANGE_MAX
        if change >= thr:
            continue

        price = float(closes.iloc[-1])
        rows.append({
            "代號": sid,
            "名稱": name_map.get(sid, ""),
            "收盤": round(price, 2),
            f"MA{MA_PERIOD}": round(ma20, 2),
            "離月線%": round((price / ma20 - 1) * 100, 1),
            "日均量(張)": round(avg_lot),
            "週量趨勢(張/日)": " → ".join(str(x) for x in wk_vols),
            "週量降幅%": round((wk_vols[-1] / wk_vols[0] - 1) * 100, 1) if wk_vols[0] else None,
            "散戶比例變化%": change,
            "散戶比例%": latest_pct,
            "散戶週數": weeks,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # 排序:越「被遺忘」越前面 → 週量降幅最大(最負)優先,其次散戶跑最兇
        df = df.sort_values(["週量降幅%", "散戶比例變化%"]).reset_index(drop=True)
        df.index += 1
    meta = {
        "asof": asof,
        "holders_date": holders_date,
        "n": len(df),
        "conditions": (f"週轉率(週量)連 {VOL_DOWN_WEEKS} 週↓ + 散戶比例連續↓ + "
                       f"收盤站穩 MA{MA_PERIOD}({STABLE_DAYS}日,不追高 ≤+{MAX_ABOVE_MA_PCT:.0f}%) + "
                       f"日均量≥{MIN_AVG_LOT}張"),
    }
    return df, meta


def run():
    df, meta = screen()
    print(f">>> 籌碼沉澱/冷門股篩選 | 資料截止 {meta['asof']} | 集保最新 {meta['holders_date']}")
    print(f">>> 條件:{meta['conditions']}\n")
    if df.empty:
        print("本次無股票同時符合四關。")
        return None
    print(f"✅ 共 {len(df)} 檔同時符合四關:\n")
    print(df.to_string())
    stamp = datetime.now(TPE_TZ).strftime("%Y%m%d_%H%M")
    out = f"籌碼沉澱篩選_{stamp}.xlsx"
    df.to_excel(out, index_label="排名")
    print(f"\n已存檔:{out}")
    return df


if __name__ == "__main__":
    run()
