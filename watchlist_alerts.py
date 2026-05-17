"""自選股告警

對自選股清單做技術 / 籌碼面檢查,任一條件觸發就產生警示。

檢查條件:
1. 跌破 MA20(防守位失守)
2. 跌破 MA60(季線崩跌)
3. KD 死叉(技術轉弱)
4. 外資連 3 日賣超(籌碼鬆動)

預期被 telegram_notify.py 在主流程中呼叫,將結果整合到推播訊息。
"""
import pandas as pd


def _calc_kd_for_alert(high, low, close, n: int = 9):
    """簡化版 KD 計算(只算最後兩天的 K/D 用來判斷死叉)。
    回傳 (k_prev, d_prev, k_now, d_now);資料不足回 (None, None, None, None)。
    """
    if len(close) < n + 2:
        return None, None, None, None
    rsv_list = []
    for i in range(n - 1, len(close)):
        period_high = high.iloc[i - n + 1: i + 1].max()
        period_low  = low.iloc[i - n + 1: i + 1].min()
        if period_high == period_low:
            rsv = 50.0
        else:
            rsv = (close.iloc[i] - period_low) / (period_high - period_low) * 100
        rsv_list.append(rsv)

    k_list = [50.0]
    d_list = [50.0]
    for rsv in rsv_list:
        k = (2 / 3) * k_list[-1] + (1 / 3) * rsv
        d = (2 / 3) * d_list[-1] + (1 / 3) * k
        k_list.append(k)
        d_list.append(d)

    if len(k_list) < 3:
        return None, None, None, None
    return k_list[-2], d_list[-2], k_list[-1], d_list[-1]


# ══════════════════════════════════════════════════════════════════════
# 個別檢查函式
# ══════════════════════════════════════════════════════════════════════
def _load_stock_history(cache_dir, sid: str, n_days: int = 100):
    """從 daily parquet 拉個股最近 N 天。"""
    try:
        files = sorted(cache_dir.glob('daily_*.parquet'))
        if not files:
            return None
        df = pd.read_parquet(files[-1], filters=[('stock_id', '==', str(sid))])
        if df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').tail(n_days).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"⚠ 讀 {sid} history 失敗: {e}")
        return None


def _load_institutional(cache_dir, sid: str, n_days: int = 5):
    """從 institutional parquet 拉個股最近 N 天三大法人。
    Returns: DataFrame[date, Foreign_Investor(net_lots), Investment_Trust(net_lots)]
    """
    try:
        files = sorted(cache_dir.glob('institutional_*.parquet'))
        if not files:
            return None
        df = pd.read_parquet(files[-1], filters=[('stock_id', '==', str(sid))])
        if df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'])
        df['net_lots'] = (df['buy'] - df['sell']) / 1000.0
        pv = df.pivot_table(index='date', columns='name', values='net_lots', aggfunc='sum').reset_index()
        for c in ['Foreign_Investor', 'Investment_Trust']:
            if c not in pv.columns:
                pv[c] = 0.0
        return pv.sort_values('date').tail(n_days).reset_index(drop=True)
    except Exception as e:
        print(f"⚠ 讀 {sid} 法人失敗: {e}")
        return None


def check_single_stock(cache_dir, sid: str) -> list:
    """對單一個股做全套檢查,回傳該股的警示清單(可多筆)。"""
    alerts = []
    hist = _load_stock_history(cache_dir, sid, n_days=80)
    if hist is None or len(hist) < 21:
        return []

    high_col = 'max' if 'max' in hist.columns else 'high'
    low_col  = 'min' if 'min' in hist.columns else 'low'
    close = hist['close']
    latest = float(close.iloc[-1])
    prev   = float(close.iloc[-2]) if len(close) >= 2 else latest

    # 1. 跌破 MA20(前一日仍站上、今日跌破)─ 新跌破才警示,避免每天重複
    if len(close) >= 21:
        ma20_now  = float(close.tail(20).mean())
        ma20_prev = float(close.iloc[-21:-1].mean())
        if prev >= ma20_prev and latest < ma20_now:
            alerts.append({
                "sid": sid, "type": "MA20_BREAK",
                "msg": f"跌破 MA20 ({ma20_now:.1f},現價 {latest:.1f})",
            })

    # 2. 跌破 MA60
    if len(close) >= 61:
        ma60_now  = float(close.tail(60).mean())
        ma60_prev = float(close.iloc[-61:-1].mean())
        if prev >= ma60_prev and latest < ma60_now:
            alerts.append({
                "sid": sid, "type": "MA60_BREAK",
                "msg": f"跌破季線 MA60 ({ma60_now:.1f},現價 {latest:.1f})",
            })

    # 3. KD 死叉(且 K 值 > 50,代表從高檔死叉,風險訊號)
    k_prev, d_prev, k_now, d_now = _calc_kd_for_alert(hist[high_col], hist[low_col], close)
    if all(x is not None for x in [k_prev, d_prev, k_now, d_now]):
        if k_prev >= d_prev and k_now < d_now and k_now > 50:
            alerts.append({
                "sid": sid, "type": "KD_DEATH_CROSS",
                "msg": f"高檔 KD 死叉(K={k_now:.0f}, D={d_now:.0f})",
            })

    # 4. 外資連 3 日賣超
    inst = _load_institutional(cache_dir, sid, n_days=5)
    if inst is not None and len(inst) >= 3:
        foreign_recent3 = inst['Foreign_Investor'].tail(3)
        if (foreign_recent3 < 0).all() and foreign_recent3.sum() < -50:  # 至少累計賣超 50 張
            total = foreign_recent3.sum()
            alerts.append({
                "sid": sid, "type": "FOREIGN_SELL_3D",
                "msg": f"外資連 3 日賣超(累計 {total:.0f} 張)",
            })

    return alerts


def check_watchlist(cache_dir, watchlist: list) -> list:
    """對整份 watchlist 跑檢查。

    Args:
        cache_dir: pathlib.Path 指向 CACHE_DIR
        watchlist: ["2330", "2454", ...] 自選股代號清單

    Returns:
        [{"sid": str, "type": str, "msg": str}, ...] 全部觸發的警示
    """
    if not watchlist:
        return []
    all_alerts = []
    for sid in watchlist:
        try:
            all_alerts.extend(check_single_stock(cache_dir, str(sid)))
        except Exception as e:
            print(f"⚠ 檢查 {sid} 失敗: {e}")
    return all_alerts


# ══════════════════════════════════════════════════════════════════════
# 給 TG 用的格式化
# ══════════════════════════════════════════════════════════════════════
# 警示類型 → emoji 對映(台股紅警示)
ALERT_ICON = {
    "MA20_BREAK":      "🔴",
    "MA60_BREAK":      "🔴",
    "KD_DEATH_CROSS":  "🟡",
    "FOREIGN_SELL_3D": "🟠",
}


def format_alerts_for_tg(alerts: list, name_map: dict = None) -> str:
    """組成 TG 訊息片段(HTML 格式)。"""
    if not alerts:
        return ""
    name_map = name_map or {}
    # 依 sid 分組:同一檔股票多警示時集中顯示
    by_sid = {}
    for a in alerts:
        by_sid.setdefault(a["sid"], []).append(a)

    lines = [f"\n⚠️ <b>自選股警示 ({len(by_sid)} 檔)</b>"]
    for sid in sorted(by_sid.keys()):
        name = name_map.get(sid, "")
        for a in by_sid[sid]:
            icon = ALERT_ICON.get(a["type"], "⚠️")
            lines.append(
                f"{icon} <a href='https://tw.stock.yahoo.com/quote/{sid}'>{sid}</a> {name}:{a['msg']}"
            )
    return "\n".join(lines) + "\n"
