"""策略績效追蹤

對 picks_history 內的每筆選股,從 daily parquet 拉出後續 N 日 close,
算出「入選後 N 日報酬」、「勝率」、「平均報酬」、「最大回檔」等指標。

提供 UI「策略績效」分頁需要的 dataframe。
"""
import pandas as pd
from picks_history import get_picks

# 台股來回交易成本估計(%):手續費買賣各 ~0.1425% + 賣出交易稅 0.3%,約 0.5%
TRADE_COST_PCT = 0.5
# 市價單滑價估計(%):以隔日開盤掛市價,實際成交通常略高於開盤報價
SLIPPAGE_PCT = 0.1


# ══════════════════════════════════════════════════════════════════════
# 載入價格矩陣(優化:一次讀檔,所有 pick 共用)
# ══════════════════════════════════════════════════════════════════════
def _load_price_matrices(cache_dir):
    """讀最新 daily parquet,組成 close + open pivot tables。

    Returns:
        {"close": DataFrame, "open": DataFrame | None}
        index=date(Timestamp), columns=stock_id(str)
        失敗回 None
    """
    try:
        files = sorted(cache_dir.glob('daily_*.parquet'))
        if not files:
            return None
        try:
            df = pd.read_parquet(files[-1], columns=['stock_id', 'date', 'close', 'open'])
        except Exception:
            df = pd.read_parquet(files[-1], columns=['stock_id', 'date', 'close'])
        if df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'])
        df['stock_id'] = df['stock_id'].astype(str)
        df = df.drop_duplicates(subset=['date', 'stock_id'], keep='last')
        close = df.pivot(index='date', columns='stock_id', values='close')
        open_ = (
            df.pivot(index='date', columns='stock_id', values='open')
            if 'open' in df.columns else None
        )
        return {"close": close, "open": open_}
    except Exception as e:
        print(f"⚠ 讀取 price 矩陣失敗: {e}")
        return None


def _forward_return(matrices, sid: str, entry_date, n_days: int):
    """算「entry_date 訊號日隔日開盤進場、持有 n_days 個交易日的報酬率(%)」。

    進場以訊號日(T)的隔日(T+1)開盤 + SLIPPAGE_PCT 滑價為基準,
    消除「用訊號當日收盤當進場價」的前視偏誤。
    若無開盤資料則回退為訊號日收盤(舊行為,計算口徑不中斷)。
    """
    if matrices is None:
        return None
    close_matrix = matrices["close"]
    open_matrix = matrices.get("open")

    if sid not in close_matrix.columns:
        return None
    close_s = close_matrix[sid].dropna()
    if close_s.empty:
        return None

    entry_ts = pd.Timestamp(entry_date)
    dates = close_s.index
    idx = dates.searchsorted(entry_ts)
    if idx >= len(dates):
        return None

    next_idx = idx + 1          # T+1:隔日
    target_idx = idx + n_days   # T+n:出場日
    if next_idx >= len(dates) or target_idx >= len(dates):
        return None

    target_close = close_s.iloc[target_idx]
    if pd.isna(target_close):
        return None

    # 取隔日開盤;無資料時回退為訊號日收盤
    entry_open = float('nan')
    if open_matrix is not None and sid in open_matrix.columns:
        next_date = dates[next_idx]
        if next_date in open_matrix.index:
            entry_open = open_matrix.loc[next_date, sid]

    if pd.isna(entry_open) or entry_open <= 0:
        entry_open = close_s.iloc[idx]   # fallback: T close

    actual_entry = entry_open * (1 + SLIPPAGE_PCT / 100)
    if actual_entry <= 0 or pd.isna(actual_entry):
        return None
    return (target_close - actual_entry) / actual_entry * 100


def _risk_metrics(returns_in_date_order: list, hold_days: int) -> dict:
    """從「依日期排序的報酬序列」算風險指標,口徑對齊 backtest.py 的 summarize()。

    - mdd:    最大回檔(%),假設等權、序列交易、複利累積的資金曲線最大跌幅(負數)
    - sharpe: 年化夏普值 = 平均/標準差 × √(252/hold_days)
    - std:    報酬標準差(母體,ddof=0,與 backtest 的 np.std 一致)

    序列 < 2 筆時 std/sharpe 回 0;空序列全回 0。
    """
    rets = returns_in_date_order
    if not rets:
        return {"mdd": 0.0, "sharpe": 0.0, "std": 0.0}

    # 最大回檔:複利資金曲線從高點到低點的最大跌幅
    # 口徑對齊 backtest.py:高點從「第一筆交易後的資金」起算(等同 np.maximum.accumulate)
    equity_curve = []
    e = 1.0
    for r in rets:
        e *= (1 + r / 100.0)
        equity_curve.append(e)
    peak, mdd = equity_curve[0], 0.0
    for e in equity_curve:
        if e > peak:
            peak = e
        if peak <= 0:
            # 資金曲線歸零/轉負(報酬 ≤ -100%,通常是資料異常)→ 視為 -100% 回檔,不再除零
            mdd = min(mdd, -1.0)
            continue
        dd = (e - peak) / peak
        if dd < mdd:
            mdd = dd
    mdd_pct = mdd * 100.0

    # 標準差(母體)+ 年化 Sharpe
    mean = sum(rets) / len(rets)
    if len(rets) > 1:
        var = sum((x - mean) ** 2 for x in rets) / len(rets)
        std = var ** 0.5
    else:
        std = 0.0
    if std > 0 and hold_days > 0:
        sharpe = (mean / std) * (252.0 / hold_days) ** 0.5
    else:
        sharpe = 0.0

    return {"mdd": mdd_pct, "sharpe": sharpe, "std": std}


def _nonoverlap_mdd(series: list, hold_days: int) -> float:
    """非重疊取樣的最大回檔(%)。

    「每入選日平均報酬」是 hold_days 天的重疊窗口——天天選股時,相鄰入選日的
    持有期高度重疊,直接連續複利會把同一段行情重複計入,把資金曲線回檔嚴重灌大
    (例:一段實際只跌 10% 出頭的回檔,串出來的 MDD 可達 -35%~-40%)。

    改成每隔 hold_days 取一筆(相鄰取樣點的持有期才不重疊,等同序列獨立交易),
    再算複利 MDD。對所有 hold_days 種起始相位各算一次取平均,避免單一起點的偶然。
    """
    if hold_days <= 1 or len(series) < 2:
        return _risk_metrics(series, hold_days)["mdd"]
    mdds = []
    for phase in range(min(hold_days, len(series))):
        sub = series[phase::hold_days]
        if len(sub) >= 2:
            mdds.append(_risk_metrics(sub, hold_days)["mdd"])
    if not mdds:
        return _risk_metrics(series, hold_days)["mdd"]
    return sum(mdds) / len(mdds)


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
    matrices = _load_price_matrices(cache_dir)
    if matrices is None:
        return {"samples": [], "overall": {}, "by_score": {}, "error": "無法讀取 daily 快取"}
    close_matrix = matrices["close"]

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
                row[f"return_{n}d"] = _forward_return(matrices, sid, entry_date, n)
            samples.append(row)

    # ── 整體統計 ──
    overall = {}
    if samples:
        for n in n_days_list:
            valid = [s[f"return_{n}d"] for s in samples if s[f"return_{n}d"] is not None]
            if valid:
                wins = sum(1 for v in valid if v > 0)
                gains = [v for v in valid if v > 0]
                losses = [v for v in valid if v <= 0]
                overall[f"n_{n}d"] = len(valid)
                overall[f"win_rate_{n}d"] = wins / len(valid)
                overall[f"avg_return_{n}d"] = sum(valid) / len(valid)
                overall[f"median_return_{n}d"] = sorted(valid)[len(valid) // 2]
                overall[f"max_return_{n}d"] = max(valid)
                overall[f"min_return_{n}d"] = min(valid)
                overall[f"avg_gain_{n}d"] = sum(gains) / len(gains) if gains else 0.0
                overall[f"avg_loss_{n}d"] = sum(losses) / len(losses) if losses else 0.0
                # 損益比:總獲利 / 總虧損絕對值;無虧損時為 inf
                total_loss = sum(losses)
                overall[f"profit_factor_{n}d"] = (
                    sum(gains) / abs(total_loss) if total_loss < 0 else float('inf')
                )
                # 淨期望值 = 平均報酬 - 來回交易成本(扣成本後實際落袋)
                overall[f"net_expectancy_{n}d"] = sum(valid) / len(valid) - TRADE_COST_PCT
                # 風險指標(最大回檔、夏普值):
                # 用「每個入選日的平均報酬」序列,而非逐筆 pick。
                # 原因:逐筆把同日多檔當成連續交易、且重疊的 N 日窗口會被重複複利,
                #       會把資金曲線波動嚴重灌水(同 compute_equity_curve 的註解)。
                #       收斂成每日一點後,口徑與「vs 大盤」走勢圖一致、數字才真實。
                _by_date = {}
                for s in samples:
                    r = s[f"return_{n}d"]
                    if r is not None:
                        _by_date.setdefault(s["date"], []).append(r)
                daily_avg = [sum(v) / len(v) for _, v in sorted(_by_date.items())]
                _risk = _risk_metrics(daily_avg, n)
                # MDD 改非重疊取樣,與 check_system_health 同口徑:收斂成每日一點仍是
                # N 日重疊窗口(天天選股時相鄰日持有期重疊),直接複利會把同段下跌重複
                # 計入、灌大回檔。_nonoverlap_mdd 每隔 N 取一筆再算,才是真實序列交易回檔。
                # (sharpe/std 是離散度指標、不會像 MDD 那樣累乘放大,仍用原序列)
                overall[f"mdd_{n}d"] = _nonoverlap_mdd(daily_avg, n)
                overall[f"sharpe_{n}d"] = _risk["sharpe"]
                overall[f"std_{n}d"] = _risk["std"]
                overall[f"risk_n_{n}d"] = len(daily_avg)   # 風險指標的有效「天數」樣本

    # ── 分數區間統計 ──
    by_score = {}
    for score in sorted({s["score"] for s in samples if s["score"] is not None}):
        score_samples = [s for s in samples if s["score"] == score]
        stat = {"n_picks": len(score_samples)}
        for n in n_days_list:
            valid = [s[f"return_{n}d"] for s in score_samples if s[f"return_{n}d"] is not None]
            if valid:
                wins = sum(1 for v in valid if v > 0)
                gains = [v for v in valid if v > 0]
                losses = [v for v in valid if v <= 0]
                stat[f"n_{n}d"] = len(valid)
                stat[f"win_rate_{n}d"] = wins / len(valid)
                stat[f"avg_return_{n}d"] = sum(valid) / len(valid)
                stat[f"avg_gain_{n}d"] = sum(gains) / len(gains) if gains else 0.0
                stat[f"avg_loss_{n}d"] = sum(losses) / len(losses) if losses else 0.0
                total_loss = sum(losses)
                stat[f"profit_factor_{n}d"] = (
                    sum(gains) / abs(total_loss) if total_loss < 0 else float('inf')
                )
        by_score[score] = stat

    # ── 同樣本比較(common sample):不同持有期鎖定同一批 pick ──
    # 各持有期的有效樣本天生不同(新 pick 只活得出短天期報酬),直接互比會變成
    # 「3 日含最近的單、10 日只剩早期的單」各說各話(出場回測 vs 持有天數表打架的根源)。
    # 取「最長且已有資料的持有期」算得出報酬的那批 pick,所有持有期都只統計這批
    # → 同一批股票、同一段時期,跨持有期才可直接比較。
    overall_common = {}
    if samples:
        common_base_n = None
        for n in sorted(n_days_list, reverse=True):
            if any(s[f"return_{n}d"] is not None for s in samples):
                common_base_n = n
                break
        if common_base_n:
            common = [s for s in samples if s[f"return_{common_base_n}d"] is not None]
            overall_common["base_n"] = common_base_n
            overall_common["n_picks"] = len(common)
            for n in n_days_list:
                valid = [s[f"return_{n}d"] for s in common if s[f"return_{n}d"] is not None]
                if not valid:
                    continue
                wins = sum(1 for v in valid if v > 0)
                gains = [v for v in valid if v > 0]
                losses = [v for v in valid if v <= 0]
                total_loss = sum(losses)
                overall_common[f"n_{n}d"] = len(valid)
                overall_common[f"win_rate_{n}d"] = wins / len(valid)
                overall_common[f"avg_return_{n}d"] = sum(valid) / len(valid)
                overall_common[f"net_expectancy_{n}d"] = sum(valid) / len(valid) - TRADE_COST_PCT
                overall_common[f"profit_factor_{n}d"] = (
                    sum(gains) / abs(total_loss) if total_loss < 0 else float('inf')
                )

    return {"samples": samples, "overall": overall, "by_score": by_score,
            "overall_common": overall_common}


def _load_twii_regime(cache_dir, ma_days: int = 60):
    """讀 ^TWII,回傳 (close 序列, MA 序列)。判斷各日是否站上季線用。失敗回 None。"""
    from pathlib import Path
    try:
        files = sorted(Path(cache_dir).glob("twii_*.parquet"))
        if not files:
            return None
        df = pd.read_parquet(files[-1])
        if "date" not in df.columns or "Close" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        close = df["Close"].dropna()
        if len(close) < ma_days:
            return None
        ma = close.rolling(ma_days).mean()
        return close, ma
    except Exception as e:
        print(f"⚠ 讀取 twii regime 失敗: {e}")
        return None


def backtest_market_filter(history: list, cache_dir, hold_days: int = 5, ma_days: int = 60) -> dict:
    """回測「大盤濾網」:只在特定大盤狀態進場,淨期望值會不會更好?

    對每筆已滿持有期的 pick,判斷其入選日的大盤狀態(站上/跌破季線),
    比較「全部進場(基準)」vs「只在多頭」vs「只在空頭」的勝率/淨期望值。
    若情緒歷史足夠,額外比較「只在溫度 ≥ 50」vs「< 50」。

    Returns:
        {
          "hold_days","ma_days","n_total",
          "scenarios": [{"name","stat":{n,win_rate,avg,net_exp}|None}, ...],
          "temp_block": {"base","warm","cool","n_temp"}|None,
        }
        資料不足回 {"error": "..."}。
    """
    matrices = _load_price_matrices(cache_dir)
    if matrices is None:
        return {"error": "無法讀取 daily 快取"}
    close_matrix = matrices["close"]
    twii = _load_twii_regime(cache_dir, ma_days)
    if twii is None:
        return {"error": f"找不到 ^TWII 或資料不足 {ma_days} 日,無法判斷大盤狀態"}
    close_t, ma_t = twii

    # 情緒歷史(選用):date -> temp
    temp_by_date = {}
    try:
        from market_sentiment import load_sentiment_history
        for h in load_sentiment_history(cache_dir):
            if h.get("temp") is not None and h.get("date"):
                temp_by_date[h["date"]] = h["temp"]
    except Exception:
        pass

    def _regime_at(entry_ts):
        """該入選日是否站上季線。資料不足回 None。"""
        idx = close_t.index.searchsorted(entry_ts, side="right") - 1
        if idx < 0 or idx >= len(close_t):
            return None
        mav = ma_t.iloc[idx]
        if pd.isna(mav):
            return None
        return bool(close_t.iloc[idx] > mav)

    recs = []
    for entry in history:
        if entry.get("date") == "legacy":
            continue
        try:
            entry_ts = pd.Timestamp(entry["date"])
        except Exception:
            continue
        bullish = _regime_at(entry_ts)
        temp = temp_by_date.get(entry["date"])
        for pick in get_picks(entry):
            sid = str(pick.get("sid", ""))
            if not sid:
                continue
            ret = _forward_return(matrices, sid, entry_ts, hold_days)
            if ret is None:
                continue
            recs.append({"ret": ret, "bullish": bullish, "temp": temp})

    if not recs:
        return {"error": "尚無足夠已滿持有期的樣本"}

    def _stat(subset):
        vals = [r["ret"] for r in subset]
        if not vals:
            return None
        wins = sum(1 for v in vals if v > 0)
        avg = sum(vals) / len(vals)
        return {"n": len(vals), "win_rate": wins / len(vals),
                "avg": avg, "net_exp": avg - TRADE_COST_PCT}

    bull = [r for r in recs if r["bullish"] is True]
    bear = [r for r in recs if r["bullish"] is False]
    scenarios = [
        {"name": "全部進場(基準)",          "stat": _stat(recs)},
        {"name": f"只在多頭(站上 MA{ma_days})", "stat": _stat(bull)},
        {"name": f"只在空頭(跌破 MA{ma_days})", "stat": _stat(bear)},
    ]

    temp_recs = [r for r in recs if r["temp"] is not None]
    temp_block = None
    if len(temp_recs) >= 5:
        temp_block = {
            "base": _stat(temp_recs),
            "warm": _stat([r for r in temp_recs if r["temp"] >= 50]),
            "cool": _stat([r for r in temp_recs if r["temp"] < 50]),
            "n_temp": len(temp_recs),
        }

    return {
        "hold_days": hold_days, "ma_days": ma_days, "n_total": len(recs),
        "scenarios": scenarios, "temp_block": temp_block,
    }


def _price_path(matrices, sid: str, entry_date, max_hold: int):
    """進場後每日報酬路徑 [r_day1, ..., r_dayMax](%)。

    進場以訊號日(T)隔日(T+1)開盤 + SLIPPAGE_PCT 滑價估算;
    路徑 r_dayk = (close[T+k] - actual_entry) / actual_entry。
    需要完整 max_hold 個交易日(不足回 None),確保各出場策略評估同一批 pick。
    """
    if matrices is None:
        return None
    close_matrix = matrices["close"]
    open_matrix = matrices.get("open")

    if sid not in close_matrix.columns:
        return None
    s = close_matrix[sid].dropna()
    if s.empty:
        return None
    dates = s.index
    idx = dates.searchsorted(pd.Timestamp(entry_date))
    if idx >= len(dates) or idx + max_hold >= len(dates):
        return None

    # 取隔日開盤作為進場價;無資料時回退為當日收盤
    entry_open = float('nan')
    if open_matrix is not None and sid in open_matrix.columns and idx + 1 < len(dates):
        next_date = dates[idx + 1]
        if next_date in open_matrix.index:
            entry_open = open_matrix.loc[next_date, sid]
    if pd.isna(entry_open) or entry_open <= 0:
        entry_open = s.iloc[idx]   # fallback: T close

    actual_entry = entry_open * (1 + SLIPPAGE_PCT / 100)
    if pd.isna(actual_entry) or actual_entry <= 0:
        return None

    rets = []
    for k in range(1, max_hold + 1):
        c = s.iloc[idx + k]
        if pd.isna(c):
            return None
        rets.append((c - actual_entry) / actual_entry * 100)
    return rets


def backtest_exit_rules(history: list, cache_dir, max_hold: int = 10) -> dict:
    """出場規則回測:固定持有 vs 停損 / 停利 / 移動停損,哪種最賺?

    對每筆「有完整 max_hold 日路徑」的 pick,模擬各出場策略的實現報酬與持有天數,
    比較淨期望值、勝率、平均持有天數、最差單筆、日均報酬(效率)。

    Returns:
        {"max_hold","n","strategies":[{name,win_rate,avg,net_exp,avg_days,worst,daily}, ...]}
        資料不足回 {"error": ...}。
    """
    matrices = _load_price_matrices(cache_dir)
    if matrices is None:
        return {"error": "無法讀取 daily 快取"}

    paths = []
    for entry in history:
        if entry.get("date") == "legacy":
            continue
        try:
            entry_ts = pd.Timestamp(entry["date"])
        except Exception:
            continue
        for pick in get_picks(entry):
            sid = str(pick.get("sid", ""))
            if not sid:
                continue
            p = _price_path(matrices, sid, entry_ts, max_hold)
            if p:
                paths.append(p)

    if not paths:
        return {"error": f"尚無滿 {max_hold} 個交易日的樣本(pick 要夠老才算得出完整路徑)。"}

    # ── 各出場策略:回傳 (實現報酬%, 持有天數) ──
    def _fixed(path, n):
        n = min(n, len(path))
        return path[n - 1], n

    def _stop(path, stop_pct):           # 跌破 -stop_pct% 即出場,否則持滿
        for d, r in enumerate(path, start=1):
            if r <= -stop_pct:
                return r, d
        return path[-1], len(path)

    def _take_profit(path, tp_pct):      # 漲到 +tp_pct% 即出場,否則持滿
        for d, r in enumerate(path, start=1):
            if r >= tp_pct:
                return r, d
        return path[-1], len(path)

    def _trailing(path, draw_pct):       # 從波段高點(價格)回落 draw_pct% 出場
        peak_ratio = 1.0
        for d, r in enumerate(path, start=1):
            ratio = 1 + r / 100.0
            if ratio > peak_ratio:
                peak_ratio = ratio
            if ratio <= peak_ratio * (1 - draw_pct / 100.0):
                return r, d
        return path[-1], len(path)

    _STRATS = [
        (f"固定持有 {min(5, max_hold)} 日",  lambda p: _fixed(p, 5)),
        (f"固定持有 {max_hold} 日",          lambda p: _fixed(p, max_hold)),
        ("停損 -5%(否則持滿)",              lambda p: _stop(p, 5)),
        ("停損 -8%(否則持滿)",              lambda p: _stop(p, 8)),
        ("移動停損 8%(高點回落)",           lambda p: _trailing(p, 8)),
        ("停利 +10%(否則持滿)",             lambda p: _take_profit(p, 10)),
    ]

    strategies = []
    for name, fn in _STRATS:
        rets, days = [], []
        for path in paths:
            r, d = fn(path)
            rets.append(r)
            days.append(d)
        n = len(rets)
        wins = sum(1 for r in rets if r > 0)
        avg = sum(rets) / n
        net_exp = avg - TRADE_COST_PCT
        avg_days = sum(days) / n
        strategies.append({
            "name": name, "win_rate": wins / n, "avg": avg,
            "net_exp": net_exp, "avg_days": avg_days,
            "worst": min(rets),
            "daily": net_exp / avg_days if avg_days > 0 else 0.0,
        })

    return {"max_hold": max_hold, "n": len(paths), "strategies": strategies}


# 10 個計分細項的穩定 key(對應 build_picks_from_df 寫入的 "sig")
_SIG_KEYS = ["投信", "外資", "雙買", "券", "大戶", "散戶", "技術", "KD", "營收", "RS"]


def attribute_signals(history: list, cache_dir, hold_days: int = 5) -> dict:
    """訊號歸因:拆解 10 個計分細項各自對後續報酬的貢獻。

    對每筆「有存細項(sig)且已滿持有期」的 pick,
    依每個訊號「有觸發(=1) vs 沒觸發(=0)」分組,比較後續 N 日平均報酬/勝率。
    edge = 觸發組平均 − 未觸發組平均;edge 為正代表該訊號確實加值。

    注意:歷史 pick 在導入 sig 記錄前不含細項,故只用有 sig 的樣本;
    需累積一段時間才有參考價值。

    Returns:
        {
          "hold_days","n_eval","n_with_sig",
          "per_signal": [{"key","on":{n,win_rate,avg}|None,"off":{...}|None,"edge":float|None}, ...],
        }
        無法讀快取回 {"error": ...};尚無 sig 樣本回 {"n_eval":0,...} 供 UI 顯示累積中。
    """
    matrices = _load_price_matrices(cache_dir)
    if matrices is None:
        return {"error": "無法讀取 daily 快取"}
    close_matrix = matrices["close"]

    recs = []          # (sig_dict, ret)
    n_with_sig = 0     # 有 sig 欄位的 pick 數(不論是否滿持有期)
    for entry in history:
        if entry.get("date") == "legacy":
            continue
        try:
            entry_ts = pd.Timestamp(entry["date"])
        except Exception:
            continue
        for pick in get_picks(entry):
            sid = str(pick.get("sid", ""))
            if not sid:
                continue
            sig = pick.get("sig")
            if not isinstance(sig, dict):
                continue
            n_with_sig += 1
            ret = _forward_return(matrices, sid, entry_ts, hold_days)
            if ret is None:
                continue
            recs.append((sig, ret))

    def _agg(vals):
        if not vals:
            return None
        wins = sum(1 for v in vals if v > 0)
        return {"n": len(vals), "win_rate": wins / len(vals), "avg": sum(vals) / len(vals)}

    per_signal = []
    for key in _SIG_KEYS:
        on  = [ret for sig, ret in recs if sig.get(key) == 1]
        off = [ret for sig, ret in recs if sig.get(key) == 0]
        s_on, s_off = _agg(on), _agg(off)
        edge = (s_on["avg"] - s_off["avg"]) if (s_on and s_off) else None
        per_signal.append({"key": key, "on": s_on, "off": s_off, "edge": edge})

    return {"hold_days": hold_days, "n_eval": len(recs),
            "n_with_sig": n_with_sig, "per_signal": per_signal}


def compute_equity_curve(history: list, cache_dir, hold_days: int = 5) -> dict:
    """算「系統 picks vs ^TWII 大盤」的逐日 N 日後報酬對照。

    對每個入選日:
      - 系統:當日所有 picks 的平均 N 日後報酬(%)
      - 大盤:同期間 ^TWII 的 N 日後報酬(%)

    重要:不再對重疊的 N 日窗口做複利累加(那會把真實 ~10% 漲幅吹成 +50%)。
    改成回傳每日的「點報酬」,UI 上畫成兩條走勢線、4 卡顯示平均值。

    Args:
        history: load_history() 結果
        cache_dir: pathlib.Path 指向 CACHE_DIR
        hold_days: 持有交易日數(預設 5)

    Returns:
        {
          "dates":          [entry_date_str, ...],
          "pick_returns":   [系統平均 N 日報酬 (%), ...],
          "twii_returns":   [大盤 N 日報酬 (%), ...],
          "n_days":         樣本日數,
          "avg_pick":       系統平均 N 日報酬,
          "avg_twii":       大盤平均 N 日報酬,
          "alpha":          系統 - 大盤(百分點,直接相減),
          "win_days":       系統當日 > 大盤的天數,
          "hold_days":      N(回傳供 UI 標示),
        }
        資料不足回 None
    """
    from pathlib import Path

    matrices = _load_price_matrices(cache_dir)
    if matrices is None:
        return None
    close_matrix = matrices["close"]

    # 讀 ^TWII close + open(優先用 fetch_cache 落地的 parquet)
    # open 用於讓大盤對照與 picks 同口徑:都在 T+1 開盤進場(去除前視偏誤),避免 alpha 比較不公平
    twii_close = None
    twii_open = None
    try:
        twii_files = sorted(Path(cache_dir).glob("twii_*.parquet"))
        if twii_files:
            df_twii = pd.read_parquet(twii_files[-1])
            if "date" in df_twii.columns:
                df_twii["date"] = pd.to_datetime(df_twii["date"])
                df_twii = df_twii.set_index("date").sort_index()
            if "Close" in df_twii.columns:
                twii_close = df_twii["Close"].dropna()
            if "Open" in df_twii.columns and twii_close is not None:
                twii_open = df_twii["Open"].reindex(twii_close.index)
    except Exception as e:
        print(f"⚠ 讀取 twii parquet 失敗(equity curve 無大盤對照): {e}")

    rows = []
    for entry in history:
        if entry.get("date") == "legacy":
            continue
        try:
            entry_date = pd.Timestamp(entry["date"])
        except Exception:
            continue
        picks = get_picks(entry)

        # 系統:平均 picks 報酬
        valid_returns = []
        for pick in picks:
            sid = str(pick.get("sid", ""))
            if sid:
                ret = _forward_return(matrices, sid, entry_date, hold_days)
                if ret is not None:
                    valid_returns.append(ret)
        if not valid_returns:
            continue
        avg_ret = sum(valid_returns) / len(valid_returns)

        # 大盤:^TWII 同期間報酬。與 picks 同口徑:T+1 開盤進場、T+N 收盤出場(去除前視偏誤)。
        # 指數為對照基準、不可直接交易 → 不加滑價(picks 端的 SLIPPAGE 是真實交易成本,基準不該也扣)。
        twii_ret = None
        if twii_close is not None:
            idx = twii_close.index.searchsorted(entry_date)   # T
            next_idx = idx + 1                                 # T+1(進場日,對齊 picks)
            target_idx = idx + hold_days                       # T+N(出場日)
            if next_idx < len(twii_close) and target_idx < len(twii_close):
                # 進場價:T+1 開盤(無開盤資料則退回 T+1 收盤);出場價:T+N 收盤
                entry_p = None
                if twii_open is not None:
                    _o = twii_open.iloc[next_idx]
                    if pd.notna(_o) and _o > 0:
                        entry_p = _o
                if entry_p is None:
                    entry_p = twii_close.iloc[next_idx]
                target_p = twii_close.iloc[target_idx]
                if entry_p is not None and entry_p > 0 and pd.notna(entry_p) and pd.notna(target_p):
                    twii_ret = (target_p - entry_p) / entry_p * 100
        if twii_ret is None:
            twii_ret = 0.0  # 找不到大盤資料時當 0,避免下方統計噴掉

        rows.append({
            "date":         entry["date"],
            "pick_ret":     avg_ret,
            "twii_ret":     twii_ret,
            "n_picks":      len(valid_returns),
        })

    if not rows:
        return None

    pick_returns = [r["pick_ret"] for r in rows]
    twii_returns = [r["twii_ret"] for r in rows]
    avg_pick = sum(pick_returns) / len(pick_returns)
    avg_twii = sum(twii_returns) / len(twii_returns)
    win_days = sum(1 for r in rows if r["pick_ret"] > r["twii_ret"])

    return {
        "dates":          [r["date"] for r in rows],
        "pick_returns":   [round(v, 2) for v in pick_returns],
        "twii_returns":   [round(v, 2) for v in twii_returns],
        "n_days":         len(rows),
        "avg_pick":       round(avg_pick, 2),
        "avg_twii":       round(avg_twii, 2),
        "alpha":          round(avg_pick - avg_twii, 2),
        "win_days":       win_days,
        "hold_days":      hold_days,
    }


def compute_per_stock_performance(history: list, cache_dir, hold_days: int = 5,
                                  min_picks: int = 1) -> list:
    """個股層級績效:把每筆 pick 依「個股」分組,算每檔入選後 N 日報酬的統計。

    回答「系統選的『哪些股票』真的賺/賠?」——彙總層級看不出個股優劣,這裡拆到單檔。
    進場口徑與其他回測一致(訊號日隔日開盤 + 滑價,走 _forward_return)。

    Args:
        history: load_history() 結果
        cache_dir: CACHE_DIR
        hold_days: 持有交易日數
        min_picks: 至少入選幾次(有完整報酬)才列入,過濾單筆雜訊

    Returns:
        list[dict] 依平均報酬高→低排序:
          {sid, n, win_rate, avg, best, worst, avg_score, last_date}
        無法讀快取或無樣本回 []。
    """
    matrices = _load_price_matrices(cache_dir)
    if matrices is None:
        return []

    rec = {}   # sid -> list of (ret, score, date_str)
    for entry in history:
        if entry.get("date") == "legacy":
            continue
        try:
            entry_ts = pd.Timestamp(entry["date"])
        except Exception:
            continue
        for pick in get_picks(entry):
            sid = str(pick.get("sid", ""))
            if not sid:
                continue
            ret = _forward_return(matrices, sid, entry_ts, hold_days)
            if ret is None:
                continue
            rec.setdefault(sid, []).append((ret, pick.get("score"), entry["date"]))

    out = []
    for sid, items in rec.items():
        rets = [r for r, _, _ in items]
        if len(rets) < min_picks:
            continue
        wins = sum(1 for r in rets if r > 0)
        scores = [s for _, s, _ in items if s is not None]
        out.append({
            "sid":       sid,
            "n":         len(rets),
            "win_rate":  wins / len(rets),
            "avg":       sum(rets) / len(rets),
            "best":      max(rets),
            "worst":     min(rets),
            "avg_score": (sum(scores) / len(scores)) if scores else None,
            "last_date": max(d for _, _, d in items),
        })
    out.sort(key=lambda x: (-x["avg"], -x["n"]))
    return out


def check_system_health(history: list, cache_dir, hold_days: int = 5,
                        recent_window: int = 20) -> dict:
    """系統失效監控:策略近期的 edge 還在嗎?

    用「每入選日平均報酬」序列(對齊風險指標口徑,避免重疊窗口灌水),
    比較『近期 vs 全期』+ 當前回檔,判斷策略是否退化/失效。

    狀態(由嚴到鬆):
      🔴 fail  近期淨期望值 < 0 —— 近期選股扣成本後是賠的,edge 可能失效
      🟡 warn  近期仍正,但 (回檔過深 / 近期勝率 < 40% / 近期淨期望值 < 全期一半) 任一
      🟢 ok    近期表現正常
      ⏳ insufficient  樣本太少(全期 < 20 或近期 < 8 個入選日)不評估,避免誤報

    Returns:
        {"status","label","reason","recent_net_exp","all_net_exp",
         "recent_win_rate","mdd_recent","n_recent","n_all","hold_days"}
    """
    MDD_WARN   = -10.0   # 當前回檔(每入選日平均報酬複利曲線)深於此 % → 警戒
    WIN_WARN   = 0.40    # 近期(日)勝率低於此 → 警戒
    DECAY_FRAC = 0.5     # 近期淨期望值低於全期的此比例(且全期為正)→ 警戒
    MIN_ALL    = 20      # 全期最少入選日
    MIN_RECENT = 8       # 近期視窗最少入選日

    matrices = _load_price_matrices(cache_dir)
    if matrices is None:
        return {"status": "insufficient", "label": "⏳ 累積中",
                "reason": "無法讀取 daily 快取", "n_all": 0, "n_recent": 0,
                "hold_days": hold_days}

    # 每入選日平均報酬(date 排序)
    by_date = {}
    for entry in history:
        if entry.get("date") == "legacy":
            continue
        try:
            entry_ts = pd.Timestamp(entry["date"])
        except Exception:
            continue
        rets = []
        for pick in get_picks(entry):
            sid = str(pick.get("sid", ""))
            if not sid:
                continue
            r = _forward_return(matrices, sid, entry_ts, hold_days)
            if r is not None:
                rets.append(r)
        if rets:
            by_date[entry["date"]] = sum(rets) / len(rets)

    sorted_items = sorted(by_date.items())
    series_dates = [d for d, _ in sorted_items]
    series = [v for _, v in sorted_items]   # 依日期排序的每日平均報酬
    n_all = len(series)
    recent = series[-recent_window:] if recent_window > 0 else series
    n_recent = len(recent)

    base = {"n_all": n_all, "n_recent": n_recent, "hold_days": hold_days,
            "recent_net_exp": None, "all_net_exp": None,
            "recent_win_rate": None, "mdd_recent": None,
            "series_dates": series_dates, "series_returns": series}

    if n_all < MIN_ALL or n_recent < MIN_RECENT:
        base.update({"status": "insufficient", "label": "⏳ 累積中",
                     "reason": f"樣本不足(全期 {n_all}/{MIN_ALL} 入選日、近期 {n_recent}/{MIN_RECENT}),"
                               f"暫不評估失效,避免年輕/全多頭資料誤報。",
                     "series_dates": series_dates, "series_returns": series})
        return base

    all_net_exp    = sum(series) / n_all - TRADE_COST_PCT
    recent_net_exp = sum(recent) / n_recent - TRADE_COST_PCT
    recent_win     = sum(1 for r in recent if r > 0) / n_recent
    mdd_recent     = _nonoverlap_mdd(recent, hold_days)   # 近期回檔(%,非重疊取樣,不灌水)
    base.update({"recent_net_exp": recent_net_exp, "all_net_exp": all_net_exp,
                 "recent_win_rate": recent_win, "mdd_recent": mdd_recent})

    if recent_net_exp < 0:
        base.update({"status": "fail", "label": "🔴 失效警報",
                     "reason": f"近 {n_recent} 個入選日的淨期望值轉為 {recent_net_exp:+.2f}%(扣成本後賠錢)——"
                               f"策略 edge 可能失效,建議暫停新進場、回頭檢查訊號與大盤環境。"})
        return base

    warns = []
    if mdd_recent < MDD_WARN:
        warns.append(f"近期回檔 {mdd_recent:.1f}%(深於 {MDD_WARN:.0f}%)")
    if recent_win < WIN_WARN:
        warns.append(f"近期勝率 {recent_win*100:.0f}%(低於 {WIN_WARN*100:.0f}%)")
    if all_net_exp > 0 and recent_net_exp < all_net_exp * DECAY_FRAC:
        warns.append(f"近期淨期望值 {recent_net_exp:+.2f}% 不到全期 {all_net_exp:+.2f}% 的一半(衰退)")

    if warns:
        base.update({"status": "warn", "label": "🟡 警戒",
                     "reason": "、".join(warns) + "。edge 仍在但轉弱,建議收緊停損、降低部位。"})
    else:
        # 全期也為正且近期與其相當 → 才說「與全期一致」;否則(如全期負、近期轉正)只陳述近期,避免誤導
        if all_net_exp > 0:
            _cmp = f"與全期({all_net_exp:+.2f}%)相當" if recent_net_exp >= all_net_exp * DECAY_FRAC else f"(全期 {all_net_exp:+.2f}%)"
        else:
            _cmp = f"已優於全期({all_net_exp:+.2f}%)"
        base.update({"status": "ok", "label": "🟢 正常",
                     "reason": f"近 {n_recent} 個入選日淨期望值 {recent_net_exp:+.2f}%、勝率 {recent_win*100:.0f}%,"
                               f"{_cmp},edge 維持中。"})
    return base


def format_performance_summary(perf: dict) -> str:
    """產生一行 TG 用的績效摘要。"""
    o = perf.get("overall", {})
    if not o or "win_rate_5d" not in o:
        return ""
    pf = o.get("profit_factor_5d")
    pf_str = "∞" if pf == float("inf") else (f"{pf:.2f}" if pf is not None else "—")
    exp = o.get("net_expectancy_5d", 0.0)
    return (
        f"📊 近 30 日策略績效(5 日後):"
        f"勝率 {o['win_rate_5d']*100:.0f}% / "
        f"淨期望值 {exp:+.2f}% / "
        f"損益比 {pf_str} / "
        f"樣本 {o['n_5d']} 筆"
    )
