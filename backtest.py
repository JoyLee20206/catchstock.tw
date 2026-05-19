"""訊號回測引擎(MVP)

對 daily / institutional parquet 內每個交易日掃描三個技術訊號的觸發,
算出觸發後 N 日的報酬率,提供勝率/平均報酬等回測指標。

訊號定義與 screening0515.py 完全一致(從同一檔的常數抄出),
避免線上選股跟回測使用兩套不同邏輯。

三個訊號:
- breakout : 量價齊揚突破 60 日新高
- foreign  : 外資連 5 日買超(總淨額 > 0 且 ≥ 2 日買)
- kd_cross : KD 低檔金叉(近 5 日內,K<30 起漲,今日 K<80 維持金叉)
"""
import pandas as pd
import numpy as np


# ── 訊號參數(與 screening0515.py 同步) ───────────────────────────────
HIGH_BREAK_DAYS    = 60
HIGH_TOLERANCE     = 0.995
BREAKOUT_VOL_RATIO = 1.5
LOOKBACK_DAYS      = 5
FI_MIN_BUY_DAYS    = 2
KD_LOOKBACK        = 5
KD_LOW_FROM        = 30
KD_HIGH_CAP_NOW    = 80

SIGNAL_LABELS = {
    "breakout":         "量價齊揚突破(60 日新高)",
    "foreign":          "外資連 5 日買超",
    "kd_cross":         "KD 低檔金叉",
    "investment_trust": "投信連 5 日買超",
    "rs_market":        "RS 優於大盤(20 日)",
}


# ══════════════════════════════════════════════════════════════════════
# 載入資料(優化:整批讀進矩陣,避免逐股查詢)
# ══════════════════════════════════════════════════════════════════════
def _load_daily_matrix(cache_dir):
    """讀最新 daily parquet,組成 pivot tables。

    Returns:
        dict {
            'close':   DataFrame index=date, columns=stock_id, values=close,
            'high':    同上 但 values=high,
            'low':     同上 但 values=low,
            'volume':  同上 但 values=Trading_Volume,
        }
        失敗回 None
    """
    try:
        files = sorted(cache_dir.glob('daily_*.parquet'))
        if not files:
            return None
        df = pd.read_parquet(files[-1])
        if df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'])
        df['stock_id'] = df['stock_id'].astype(str)
        df = df.drop_duplicates(subset=['date', 'stock_id'], keep='last')

        high_col   = 'max' if 'max' in df.columns else 'high'
        low_col    = 'min' if 'min' in df.columns else 'low'
        volume_col = 'Trading_Volume' if 'Trading_Volume' in df.columns else 'volume'

        return {
            'close':  df.pivot(index='date', columns='stock_id', values='close'),
            'high':   df.pivot(index='date', columns='stock_id', values=high_col),
            'low':    df.pivot(index='date', columns='stock_id', values=low_col),
            'volume': df.pivot(index='date', columns='stock_id', values=volume_col),
        }
    except Exception as e:
        print(f"⚠ 讀取 daily matrix 失敗: {e}")
        return None


def _load_it_net_matrix(cache_dir):
    """投信每日淨買超 matrix(股),邏輯與 _load_foreign_net_matrix 一致但抓投信。"""
    try:
        files = sorted(cache_dir.glob('institutional_*.parquet'))
        if not files:
            return None
        df = pd.read_parquet(files[-1])
        if df.empty:
            return None
        mask = df['name'].str.contains('Investment_Trust|投信', na=False)
        it = df[mask].copy()
        if it.empty:
            return None
        it['date']     = pd.to_datetime(it['date'])
        it['stock_id'] = it['stock_id'].astype(str)
        it['net']      = it['buy'] - it['sell']
        it = it.groupby(['date', 'stock_id'], as_index=False)['net'].sum()
        return it.pivot(index='date', columns='stock_id', values='net')
    except Exception as e:
        print(f"⚠ 讀取投信 matrix 失敗: {e}")
        return None


def _load_foreign_net_matrix(cache_dir):
    """讀 institutional parquet,組成「外資每日淨買超(股數)」pivot table。

    Returns:
        DataFrame index=date, columns=stock_id, values=net_lots(股)
        失敗或無資料回 None
    """
    try:
        files = sorted(cache_dir.glob('institutional_*.parquet'))
        if not files:
            return None
        df = pd.read_parquet(files[-1])
        if df.empty:
            return None
        # 與 screening0515 邏輯一致:外資但排除 Dealer
        mask = (
            df['name'].str.contains('Foreign_Investor|外資', na=False) &
            ~df['name'].str.contains('Dealer', na=False)
        )
        fi = df[mask].copy()
        if fi.empty:
            return None
        fi['date'] = pd.to_datetime(fi['date'])
        fi['stock_id'] = fi['stock_id'].astype(str)
        fi['net'] = fi['buy'] - fi['sell']
        # 同 (date, stock_id) 可能有多個外資子名稱(外資/陸資合計等),加總
        fi = fi.groupby(['date', 'stock_id'], as_index=False)['net'].sum()
        return fi.pivot(index='date', columns='stock_id', values='net')
    except Exception as e:
        print(f"⚠ 讀取外資 matrix 失敗: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# 訊號偵測(向量化,在 close/high/low/volume matrix 上算)
# ══════════════════════════════════════════════════════════════════════
def detect_breakout_signals(matrices):
    """量價齊揚突破:close ≥ 前 60 日新高 × 0.995 且 量 ≥ MA20 × 1.5

    Returns: DataFrame index=date, columns=stock_id, values=bool
    """
    close = matrices['close']
    vol   = matrices['volume']

    # 前 60 日新高(不含今日):shift(1) 排除今日,再 rolling max
    rolling_high = close.shift(1).rolling(window=HIGH_BREAK_DAYS, min_periods=HIGH_BREAK_DAYS).max()
    vol_ma20     = vol.rolling(window=20, min_periods=20).mean()

    price_break = close >= rolling_high * HIGH_TOLERANCE
    vol_surge   = vol >= vol_ma20 * BREAKOUT_VOL_RATIO

    return price_break & vol_surge


def detect_investment_trust_signals(matrices, it_matrix):
    """投信連 5 日買超:近 5 日淨額 > 0 且 ≥ 2 日買超(邏輯與 detect_foreign_signals 對稱)"""
    close = matrices['close']
    if it_matrix is None:
        return pd.DataFrame(False, index=close.index, columns=close.columns)
    it_aligned   = it_matrix.reindex(index=close.index, columns=close.columns)
    rolling_sum  = it_aligned.rolling(window=LOOKBACK_DAYS, min_periods=LOOKBACK_DAYS).sum()
    rolling_pos  = (it_aligned > 0).rolling(window=LOOKBACK_DAYS, min_periods=LOOKBACK_DAYS).sum()
    return (rolling_sum > 0) & (rolling_pos >= FI_MIN_BUY_DAYS)


def detect_rs_market_signals(matrices, window: int = 20):
    """RS 優於大盤:個股近 N 日報酬 > 跨股票中位數同期報酬。
    用「市場中位報酬」代替 ^TWII 作為市場代理,避免依賴外部資料,且整批向量化。
    """
    close  = matrices['close']
    ret_n  = close.pct_change(window)
    market = ret_n.median(axis=1)               # 每日跨股票報酬中位數 = 市場代理
    return ret_n.gt(market, axis=0) & ret_n.notna()


def detect_foreign_signals(matrices, fi_matrix):
    """外資連 5 日買超:近 5 日淨額 > 0 且 ≥ 2 日買超

    Returns: DataFrame index=date, columns=stock_id, values=bool;若 fi_matrix=None 回全 False
    """
    close = matrices['close']
    if fi_matrix is None:
        return pd.DataFrame(False, index=close.index, columns=close.columns)

    # 對齊 close 的日期/股票範圍(fi_matrix 可能少股票,reindex 後缺的填 NaN)
    fi_aligned = fi_matrix.reindex(index=close.index, columns=close.columns)

    # 近 5 日 (含當日)的滾動加總、滾動買超日數
    rolling_sum  = fi_aligned.rolling(window=LOOKBACK_DAYS, min_periods=LOOKBACK_DAYS).sum()
    rolling_pos  = (fi_aligned > 0).rolling(window=LOOKBACK_DAYS, min_periods=LOOKBACK_DAYS).sum()

    return (rolling_sum > 0) & (rolling_pos >= FI_MIN_BUY_DAYS)


def _calc_kd_matrix(matrices, n_period=9):
    """向量化計算 K/D 矩陣。

    台股慣例:KD 用 RSV 平滑(α=1/3)
    K_t = 2/3 * K_{t-1} + 1/3 * RSV_t
    D_t = 2/3 * D_{t-1} + 1/3 * K_t
    初值 K_0 = D_0 = 50

    Returns: (K_matrix, D_matrix) — 與 close 同 shape,前 n_period-1 列為 NaN
    """
    close = matrices['close']
    high  = matrices['high']
    low   = matrices['low']

    # 滾動 9 日內最高最低
    period_high = high.rolling(window=n_period, min_periods=n_period).max()
    period_low  = low.rolling(window=n_period, min_periods=n_period).min()

    rsv = (close - period_low) / (period_high - period_low) * 100
    rsv = rsv.where(period_high != period_low, 50.0)  # 高低相等時 RSV=50

    # K/D 遞迴平滑:無法純 vectorize,用 numpy 沿時間軸跑
    K = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    D = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    # 找每檔股票第一個有效 RSV 的位置(即「最早能算 K 的那天」)
    # 之前的 K/D 都是 NaN;從那天起,初始 K_prev = D_prev = 50
    rsv_arr = rsv.values
    K_arr   = np.full_like(rsv_arr, np.nan, dtype=float)
    D_arr   = np.full_like(rsv_arr, np.nan, dtype=float)
    n_rows, n_cols = rsv_arr.shape

    for j in range(n_cols):
        k_prev = 50.0
        d_prev = 50.0
        started = False
        for i in range(n_rows):
            if not np.isnan(rsv_arr[i, j]):
                k = (2/3) * k_prev + (1/3) * rsv_arr[i, j]
                d = (2/3) * d_prev + (1/3) * k
                K_arr[i, j] = k
                D_arr[i, j] = d
                k_prev = k
                d_prev = d
                started = True
            elif started:
                # 中間出現 NaN(交易暫停日)→ 維持上次值,不更新
                K_arr[i, j] = np.nan
                D_arr[i, j] = np.nan

    K[:] = K_arr
    D[:] = D_arr
    return K, D


def detect_kd_cross_signals(matrices):
    """KD 低檔金叉:
    - 近 KD_LOOKBACK 日內(1~5 日前,不含今天)某日「昨 K ≤ 昨 D 且 今 K > 今 D」
    - 該交叉日的 K < KD_LOW_FROM
    - 今日 K > D 且 K < KD_HIGH_CAP_NOW(維持金叉、未過熱)

    Returns: DataFrame index=date, columns=stock_id, values=bool
    """
    K, D = _calc_kd_matrix(matrices)
    close = matrices['close']

    # 今日金叉狀態:K > D
    cross_now = K > D
    # 今日未超買
    not_overbought = K < KD_HIGH_CAP_NOW

    # 每日的金叉發生事件:昨 K ≤ 昨 D 且今 K > 今 D
    K_prev = K.shift(1)
    D_prev = D.shift(1)
    cross_event   = (K_prev <= D_prev) & (K > D)
    low_cross_event = cross_event & (K < KD_LOW_FROM)  # 低檔金叉事件

    # 近 KD_LOOKBACK 日(不含今天)內是否有發生低檔金叉
    # 用 shift(1) 排除今天,再 rolling sum
    had_low_cross = (
        low_cross_event.shift(1)
                       .rolling(window=KD_LOOKBACK, min_periods=1)
                       .sum() > 0
    )

    return cross_now & not_overbought & had_low_cross & K.notna()


# ══════════════════════════════════════════════════════════════════════
# 多訊號合併
# ══════════════════════════════════════════════════════════════════════
def combine_signals(sig_dict: dict, selected: list, mode: str = "and") -> pd.DataFrame:
    """合併多個訊號 matrix。
    mode='and' → 同日全部觸發才算(交集,訊號變少但更可信)
    mode='or'  → 同日任一觸發即算(聯集,訊號變多)
    """
    if not selected:
        return None
    mats = [sig_dict[s] for s in selected if s in sig_dict]
    if not mats:
        return None
    result = mats[0].copy().fillna(False).astype(bool)
    for m in mats[1:]:
        m_bool = m.fillna(False).astype(bool)
        result = (result & m_bool) if mode == "and" else (result | m_bool)
    return result


# ══════════════════════════════════════════════════════════════════════
# 主流程:對任一訊號矩陣算 N 日後報酬
# ══════════════════════════════════════════════════════════════════════
def compute_signal_returns(signal_matrix, close_matrix, hold_days: int) -> pd.DataFrame:
    """對訊號觸發點,算「進場後 hold_days 個交易日的報酬率」。

    Args:
        signal_matrix: bool DataFrame index=date, columns=stock_id (True 表觸發)
        close_matrix:  收盤價 DataFrame 同 shape
        hold_days:     持有交易日數

    Returns:
        DataFrame[date, stock_id, entry_close, exit_close, return_pct]
        return_pct = (exit_close - entry_close) / entry_close * 100
    """
    if signal_matrix is None or signal_matrix.empty:
        return pd.DataFrame()

    # entry = 觸發當日收盤(假設收盤掛單),exit = 持有後 N 個交易日的收盤
    exit_close = close_matrix.shift(-hold_days)

    # 把訊號矩陣 stack 成長表
    sig_long = signal_matrix.stack()
    sig_long = sig_long[sig_long].reset_index()
    sig_long.columns = ['date', 'stock_id', '_flag']

    if sig_long.empty:
        return pd.DataFrame()

    entry_long = close_matrix.stack().reset_index()
    entry_long.columns = ['date', 'stock_id', 'entry_close']

    exit_long = exit_close.stack().reset_index()
    exit_long.columns = ['date', 'stock_id', 'exit_close']

    merged = (
        sig_long[['date', 'stock_id']]
        .merge(entry_long, on=['date', 'stock_id'], how='left')
        .merge(exit_long,  on=['date', 'stock_id'], how='left')
    )
    merged = merged.dropna(subset=['entry_close', 'exit_close'])
    merged = merged[merged['entry_close'] > 0]

    merged['return_pct'] = (merged['exit_close'] - merged['entry_close']) / merged['entry_close'] * 100
    # 過濾離譜值(資料壞掉造成的 1000% 漲跌)
    merged = merged[merged['return_pct'].abs() < 100]

    return merged.sort_values('date').reset_index(drop=True)


def summarize(trades: pd.DataFrame) -> dict:
    """對一組交易紀錄算統計摘要。"""
    if trades is None or trades.empty:
        return {"n": 0}
    rets = trades['return_pct']
    wins = (rets > 0).sum()
    return {
        "n":            len(trades),
        "win_rate":     wins / len(trades),
        "avg_return":   rets.mean(),
        "median_return": rets.median(),
        "max_return":   rets.max(),
        "min_return":   rets.min(),
        "std":          rets.std(),
    }


# ══════════════════════════════════════════════════════════════════════
# 對外主介面
# ══════════════════════════════════════════════════════════════════════
def run_backtest(cache_dir, signal="breakout", hold_days: int = 10,
                 date_range: tuple = None, combine_mode: str = "and") -> dict:
    """跑一次回測。

    Args:
        cache_dir:  pathlib.Path 指向 CACHE_DIR
        signal:     'breakout' / 'foreign' / 'kd_cross'
        hold_days:  持有交易日數(5/10/20)
        date_range: (start_date, end_date) 兩個 pd.Timestamp;None = 全部

    Returns:
        {
            "signal": 'breakout',
            "hold_days": 10,
            "trades": DataFrame[date, stock_id, entry_close, exit_close, return_pct],
            "stats": {"n": ..., "win_rate": ..., ...},
            "all_signals_stats": {  # 三個訊號的 stats,用於對照圖
                "breakout":  {...},
                "foreign":   {...},
                "kd_cross":  {...},
            },
            "error": str (僅失敗時),
        }
    """
    matrices = _load_daily_matrix(cache_dir)
    if matrices is None:
        return {"error": "無法讀取 daily 快取", "trades": pd.DataFrame(), "stats": {"n": 0}}

    fi_matrix = _load_foreign_net_matrix(cache_dir)
    it_matrix = _load_it_net_matrix(cache_dir)

    # 計算 5 個訊號矩陣(對照圖一次算齊)
    sig_matrices = {
        "breakout":         detect_breakout_signals(matrices),
        "foreign":          detect_foreign_signals(matrices, fi_matrix),
        "kd_cross":         detect_kd_cross_signals(matrices),
        "investment_trust": detect_investment_trust_signals(matrices, it_matrix),
        "rs_market":        detect_rs_market_signals(matrices),
    }

    # signal 支援 str(向後相容) 或 list(多選)
    selected_list = [signal] if isinstance(signal, str) else list(signal)

    def _slice_by_date(mat):
        if date_range is None:
            return mat
        start, end = date_range
        return mat.loc[(mat.index >= start) & (mat.index <= end)]

    # 對每個訊號算交易與摘要(供對照圖)
    all_signals_stats = {}
    for sig_name, sig_mat in sig_matrices.items():
        trades = compute_signal_returns(_slice_by_date(sig_mat), matrices['close'], hold_days)
        all_signals_stats[sig_name] = summarize(trades)

    # 算「選定組合」的交易
    if len(selected_list) == 1 and selected_list[0] in sig_matrices:
        # 單一訊號:直接用上面已算好的
        combined_mat = _slice_by_date(sig_matrices[selected_list[0]])
    else:
        combined_mat = combine_signals(sig_matrices, selected_list, mode=combine_mode)
        if combined_mat is not None:
            combined_mat = _slice_by_date(combined_mat)

    selected_trades = (
        compute_signal_returns(combined_mat, matrices['close'], hold_days)
        if combined_mat is not None else pd.DataFrame()
    )

    return {
        "signal":            signal,
        "selected_signals":  selected_list,
        "combine_mode":      combine_mode,
        "hold_days":         hold_days,
        "trades":            selected_trades,
        "stats":             summarize(selected_trades),
        "all_signals_stats": all_signals_stats,
    }
