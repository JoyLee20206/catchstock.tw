"""訊號回測引擎(PRO升級版)

新增功能:
1. 動態停損與區間最大回檔 (MDD) 計算
2. 真實超額報酬 (Alpha) 計算 (相對於 TWII 大盤)
3. 支援多條件複合訊號 (Multi-Signal AND 邏輯)
"""
import pandas as pd
import numpy as np
import yfinance as yf

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
    "breakout": "量價齊揚突破(60 日新高)",
    "foreign":  "外資連 5 日買超",
    "kd_cross": "KD 低檔金叉",
}

# ══════════════════════════════════════════════════════════════════════
# 載入資料(矩陣化優化)
# ══════════════════════════════════════════════════════════════════════
def _load_daily_matrix(cache_dir):
    try:
        files = sorted(cache_dir.glob('daily_*.parquet'))
        if not files: return None
        df = pd.read_parquet(files[-1])
        if df.empty: return None
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

def _load_foreign_net_matrix(cache_dir):
    try:
        files = sorted(cache_dir.glob('institutional_*.parquet'))
        if not files: return None
        df = pd.read_parquet(files[-1])
        if df.empty: return None
        mask = (df['name'].str.contains('Foreign_Investor|外資', na=False) & ~df['name'].str.contains('Dealer', na=False))
        fi = df[mask].copy()
        if fi.empty: return None
        fi['date'] = pd.to_datetime(fi['date'])
        fi['stock_id'] = fi['stock_id'].astype(str)
        fi['net'] = fi['buy'] - fi['sell']
        fi = fi.groupby(['date', 'stock_id'], as_index=False)['net'].sum()
        return fi.pivot(index='date', columns='stock_id', values='net')
    except Exception as e:
        print(f"⚠ 讀取外資 matrix 失敗: {e}")
        return None

def _get_twii_series(start_date, end_date):
    """獲取大盤指數用來計算 Alpha"""
    if pd.isna(start_date) or pd.isna(end_date): return None
    try:
        # 多抓前後幾天避免時區與假日對齊問題
        twii = yf.download("^TWII", start=start_date - pd.Timedelta(days=10), end=end_date + pd.Timedelta(days=60), progress=False)
        if isinstance(twii.columns, pd.MultiIndex):
            twii.columns = twii.columns.get_level_values(0)
        if twii.empty: return None
        if twii.index.tz is not None:
            twii.index = twii.index.tz_localize(None)
        return twii['Close']
    except Exception as e:
        print(f"⚠ 抓取大盤失敗: {e}")
        return None

# ══════════════════════════════════════════════════════════════════════
# 訊號偵測(向量化)
# ══════════════════════════════════════════════════════════════════════
def detect_breakout_signals(matrices):
    close = matrices['close']
    vol   = matrices['volume']
    rolling_high = close.shift(1).rolling(window=HIGH_BREAK_DAYS, min_periods=HIGH_BREAK_DAYS).max()
    vol_ma20     = vol.rolling(window=20, min_periods=20).mean()
    return (close >= rolling_high * HIGH_TOLERANCE) & (vol >= vol_ma20 * BREAKOUT_VOL_RATIO)

def detect_foreign_signals(matrices, fi_matrix):
    close = matrices['close']
    if fi_matrix is None: return pd.DataFrame(False, index=close.index, columns=close.columns)
    fi_aligned = fi_matrix.reindex(index=close.index, columns=close.columns)
    rolling_sum  = fi_aligned.rolling(window=LOOKBACK_DAYS, min_periods=LOOKBACK_DAYS).sum()
    rolling_pos  = (fi_aligned > 0).rolling(window=LOOKBACK_DAYS, min_periods=LOOKBACK_DAYS).sum()
    return (rolling_sum > 0) & (rolling_pos >= FI_MIN_BUY_DAYS)

def _calc_kd_matrix(matrices, n_period=9):
    close, high, low = matrices['close'], matrices['high'], matrices['low']
    period_high = high.rolling(window=n_period, min_periods=n_period).max()
    period_low  = low.rolling(window=n_period, min_periods=n_period).min()
    rsv = (close - period_low) / (period_high - period_low) * 100
    rsv = rsv.where(period_high != period_low, 50.0)

    K = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    D = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    rsv_arr, K_arr, D_arr = rsv.values, np.full_like(rsv.values, np.nan), np.full_like(rsv.values, np.nan)
    n_rows, n_cols = rsv_arr.shape

    for j in range(n_cols):
        k_prev, d_prev, started = 50.0, 50.0, False
        for i in range(n_rows):
            if not np.isnan(rsv_arr[i, j]):
                k = (2/3) * k_prev + (1/3) * rsv_arr[i, j]
                d = (2/3) * d_prev + (1/3) * k
                K_arr[i, j], D_arr[i, j] = k, d
                k_prev, d_prev, started = k, d, True
            elif started:
                K_arr[i, j], D_arr[i, j] = np.nan, np.nan
    K[:], D[:] = K_arr, D_arr
    return K, D

def detect_kd_cross_signals(matrices):
    K, D = _calc_kd_matrix(matrices)
    cross_now = K > D
    not_overbought = K < KD_HIGH_CAP_NOW
    K_prev, D_prev = K.shift(1), D.shift(1)
    low_cross_event = (K_prev <= D_prev) & (K > D) & (K < KD_LOW_FROM)
    had_low_cross = low_cross_event.shift(1).rolling(window=KD_LOOKBACK, min_periods=1).sum() > 0
    return cross_now & not_overbought & had_low_cross & K.notna()

# ══════════════════════════════════════════════════════════════════════
# 動態停損與報酬計算
# ══════════════════════════════════════════════════════════════════════
def compute_signal_returns(signal_matrix, close_matrix, low_matrix, twii_series, hold_days: int, sl_pct: float) -> pd.DataFrame:
    if signal_matrix is None or signal_matrix.empty: return pd.DataFrame()

    # 運用 Forward Window 抓取進場後 N 日內的「最低價」以計算 MDD 與停損
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=hold_days)
    future_min_low = low_matrix.shift(-1).rolling(window=indexer, min_periods=1).min()
    exit_close = close_matrix.shift(-hold_days)

    sig_long = signal_matrix.stack()
    sig_long = sig_long[sig_long].reset_index()
    sig_long.columns = ['date', 'stock_id', '_flag']
    if sig_long.empty: return pd.DataFrame()

    entry_long = close_matrix.stack().reset_index(name='entry_close')
    exit_long = exit_close.stack().reset_index(name='exit_close')
    min_low_long = future_min_low.stack().reset_index(name='min_low')

    merged = sig_long[['date', 'stock_id']].merge(entry_long, on=['date', 'stock_id'], how='left')
    merged = merged.merge(exit_long, on=['date', 'stock_id'], how='left')
    merged = merged.merge(min_low_long, on=['date', 'stock_id'], how='left')

    # 加入大盤基準
    if twii_series is not None and not twii_series.empty:
        twii_df = pd.DataFrame({'date': twii_series.index, 'twii_entry': twii_series.values})
        twii_df['twii_exit'] = twii_df['twii_entry'].shift(-hold_days)
        merged = merged.merge(twii_df, on='date', how='left')
    else:
        merged['twii_entry'], merged['twii_exit'] = np.nan, np.nan

    merged = merged.dropna(subset=['entry_close', 'exit_close'])
    merged = merged[merged['entry_close'] > 0]

    # 結算與 MDD
    merged['raw_return'] = (merged['exit_close'] - merged['entry_close']) / merged['entry_close'] * 100
    merged['mdd'] = (merged['min_low'] - merged['entry_close']) / merged['entry_close'] * 100
    
    # 觸價停損判定
    merged['hit_sl'] = merged['mdd'] <= -sl_pct
    merged['return_pct'] = np.where(merged['hit_sl'], -sl_pct, merged['raw_return'])

    # Alpha 判定
    merged['market_return'] = (merged['twii_exit'] - merged['twii_entry']) / merged['twii_entry'] * 100
    merged['alpha'] = merged['return_pct'] - merged['market_return']

    return merged.sort_values('date').reset_index(drop=True)

def summarize(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty: return {"n": 0}
    rets = trades['return_pct']
    alphas = trades['alpha'].dropna()
    mdds = trades['mdd'].dropna()
    return {
        "n":            len(trades),
        "win_rate":     (rets > 0).sum() / len(trades),
        "avg_return":   rets.mean(),
        "median_return":rets.median(),
        "max_return":   rets.max(),
        "min_return":   rets.min(),
        "sl_rate":      trades['hit_sl'].sum() / len(trades) if 'hit_sl' in trades.columns else 0.0,
        "avg_mdd":      mdds.mean() if not mdds.empty else np.nan,
        "avg_alpha":    alphas.mean() if not alphas.empty else np.nan,
    }

# ══════════════════════════════════════════════════════════════════════
# 對外主介面
# ══════════════════════════════════════════════════════════════════════
def run_backtest(cache_dir, signals: list, hold_days: int = 10, sl_pct: float = 7.0, date_range: tuple = None) -> dict:
    matrices = _load_daily_matrix(cache_dir)
    if matrices is None: return {"error": "無法讀取 daily 快取", "trades": pd.DataFrame(), "stats": {"n": 0}}

    fi_matrix = _load_foreign_net_matrix(cache_dir)

    sig_matrices = {
        "breakout": detect_breakout_signals(matrices),
        "foreign":  detect_foreign_signals(matrices, fi_matrix),
        "kd_cross": detect_kd_cross_signals(matrices),
    }

    # 多條件組合 (AND 邏輯)
    if not signals:
        combined_sig = pd.DataFrame(False, index=matrices['close'].index, columns=matrices['close'].columns)
    else:
        combined_sig = sig_matrices[signals[0]].copy()
        for s in signals[1:]:
            combined_sig = combined_sig & sig_matrices[s]

    if date_range is not None:
        start, end = date_range
        mask = (combined_sig.index >= start) & (combined_sig.index <= end)
        combined_sig = combined_sig.loc[mask]

    twii_series = _get_twii_series(matrices['close'].index.min(), matrices['close'].index.max())

    trades = compute_signal_returns(combined_sig, matrices['close'], matrices['low'], twii_series, hold_days, sl_pct)
    combo_stats = summarize(trades)

    # 用於對照圖: 單一訊號比較
    all_signals_stats = {}
    for sig_name, sig_mat in sig_matrices.items():
        if date_range is not None: sig_mat = sig_mat.loc[mask]
        st_trades = compute_signal_returns(sig_mat, matrices['close'], matrices['low'], twii_series, hold_days, sl_pct)
        all_signals_stats[sig_name] = summarize(st_trades)

    return {
        "signals":  signals,
        "hold_days": hold_days,
        "sl_pct":   sl_pct,
        "trades":   trades,
        "stats":    combo_stats,
        "all_signals_stats": all_signals_stats,
    }