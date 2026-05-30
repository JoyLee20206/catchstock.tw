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

# 市價單滑價估計(%):以訊號日隔日開盤掛市價,實際成交通常略高於開盤報價
SLIPPAGE_PCT = 0.1

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
    # ── Tier S 🌟 台股神器(實證最強三檔) ──
    "resonance":         "籌碼共振(散戶↓)",
    "foreign":           "外資連 5 日買超",
    "revenue_breakout":  "月營收雙紅突破(YoY>10% & MoM>0)",
    # ── Tier A ✅ 有效(次強四檔) ──
    "margin_squeeze":    "資減券增(軋空力道)",
    "triple_buy":        "三大法人同步買超",
    "investment_trust":  "投信連 5 日買超",
    "breakout":          "量價齊揚突破(60 日新高)",
    # ── Tier B 🟡 效果一般(三檔) ──
    "momentum_top":      "20 日動能 Top 10%",
    "quality_breakout":  "品質突破(量縮整理後爆量)",
    "ma_golden_cross":   "MA 黃金交叉(MA20 上穿 MA60)",
    # ── Tier C ❌ 效果不顯著(待驗證) ──
    "kd_cross":          "KD 低檔金叉",
    # ── Tier R 🔄 反轉/抄底(實驗,需自行回測驗證 edge) ──
    # 與上方「追強勢」訊號方向相反:在價弱/跌深時找轉折,勝率天生較低、務必嚴設停損。
    # 另:資減券增(margin_squeeze)、籌碼共振散戶↓(resonance)本身也偏底部,可搭配。
    "washout_bounce":      "跌深量縮後帶量翻揚",
    "kd_bull_divergence":  "KD 低檔背離(價創低、KD 沒破)",
    "chip_accumulation":   "大戶逆勢增持(低檔吃貨)",
    # ── 已移除:ma20_pullback (葛蘭碧 2,實證無 alpha);detect_ma20_pullback_signals 函式保留 orphan ──
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
        open_col   = 'open' if 'open' in df.columns else None

        result = {
            'close':  df.pivot(index='date', columns='stock_id', values='close'),
            'high':   df.pivot(index='date', columns='stock_id', values=high_col),
            'low':    df.pivot(index='date', columns='stock_id', values=low_col),
            'volume': df.pivot(index='date', columns='stock_id', values=volume_col),
        }
        if open_col:
            result['open'] = df.pivot(index='date', columns='stock_id', values=open_col)
        return result
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


def _load_margin_matrices(cache_dir):
    """讀融資融券 parquet,組成 {'margin': 融資餘額 pivot, 'short': 融券餘額 pivot}。

    用於偵測「資減券增」軋空訊號 — 多頭認賠出場(融資↓) + 空頭加碼(融券↑)
    通常是股價即將反彈的反向訊號。
    """
    try:
        files = sorted(cache_dir.glob('margin_*.parquet'))
        if not files:
            return None
        df = pd.read_parquet(files[-1])
        if df.empty:
            return None
        # 必要欄位檢查
        need_cols = {'date', 'stock_id', 'MarginPurchaseTodayBalance', 'ShortSaleTodayBalance'}
        if not need_cols.issubset(df.columns):
            print(f"⚠ margin parquet 欄位不全,實際:{set(df.columns)}")
            return None
        df['date']     = pd.to_datetime(df['date'])
        df['stock_id'] = df['stock_id'].astype(str)
        df = df.drop_duplicates(subset=['date', 'stock_id'], keep='last')
        return {
            'margin': df.pivot(index='date', columns='stock_id', values='MarginPurchaseTodayBalance'),
            'short':  df.pivot(index='date', columns='stock_id', values='ShortSaleTodayBalance'),
        }
    except Exception as e:
        print(f"⚠ 讀取融資融券 matrix 失敗: {e}")
        return None


def _load_dealer_net_matrix(cache_dir):
    """自營商每日淨買超 matrix(股)。"""
    try:
        files = sorted(cache_dir.glob('institutional_*.parquet'))
        if not files:
            return None
        df = pd.read_parquet(files[-1])
        if df.empty:
            return None
        mask = df['name'].str.contains('Dealer|自營', na=False)
        d = df[mask].copy()
        if d.empty:
            return None
        d['date']     = pd.to_datetime(d['date'])
        d['stock_id'] = d['stock_id'].astype(str)
        d['net']      = d['buy'] - d['sell']
        d = d.groupby(['date', 'stock_id'], as_index=False)['net'].sum()
        return d.pivot(index='date', columns='stock_id', values='net')
    except Exception as e:
        print(f"⚠ 讀取自營商 matrix 失敗: {e}")
        return None


def _load_retail_pct_matrix(cache_dir):
    """讀 holders parquet,組成「每週各股散戶持股率 %」pivot。

    取每檔股票每週「最低 HoldingSharesLevel」(通常為 1~999 股 / 零股)當散戶代表。
    完整版「大戶↑+散戶↓」實作較繁瑣,這裡用「散戶↓」作為共振訊號的代理 ──
    實務上兩者高度相關(零和關係),已足夠抓 80% 的共振訊號。
    """
    try:
        files = sorted(cache_dir.glob('holders_*.parquet'))
        if not files:
            return None
        df = pd.read_parquet(files[-1])
        if df.empty or 'HoldingSharesLevel' not in df.columns:
            return None
        df['date']     = pd.to_datetime(df['date'])
        df['stock_id'] = df['stock_id'].astype(str)
        df['percent']  = pd.to_numeric(df['percent'], errors='coerce')
        df = df.dropna(subset=['percent', 'HoldingSharesLevel'])
        # 依 level 字串/數值排序,取每組第一筆 = 最小級距(假設「1~999 股」字串會排第一)
        df['_level_str'] = df['HoldingSharesLevel'].astype(str)
        df = df.sort_values(['stock_id', 'date', '_level_str'])
        retail = df.groupby(['stock_id', 'date'], as_index=False).first()
        return retail.pivot(index='date', columns='stock_id', values='percent')
    except Exception as e:
        print(f"⚠ 讀取散戶 matrix 失敗: {e}")
        return None


# TDCC HoldingSharesLevel:12=400,001~600,000 股;13/14=~100 萬;15=100 萬股以上(= 400 張以上大戶)
# 與 screening0515.LARGE_HOLDER_LEVELS 同步,確保「大戶」定義一致
LARGE_HOLDER_LEVELS = [12, 13, 14, 15]


def _load_large_holder_pct_matrix(cache_dir):
    """讀 holders parquet,組成「每週各股『大戶(400 張以上)』合計持股率 %」pivot。

    大戶% = HoldingSharesLevel ∈ [12,13,14,15] 的 percent 加總(每股每週)。
    用於「大戶逆勢增持」訊號:大戶% 週對週上升 = 主力在吃貨。
    """
    try:
        files = sorted(cache_dir.glob('holders_*.parquet'))
        if not files:
            return None
        df = pd.read_parquet(files[-1])
        if df.empty or 'HoldingSharesLevel' not in df.columns:
            return None
        df['date']     = pd.to_datetime(df['date'])
        df['stock_id'] = df['stock_id'].astype(str)
        df['HoldingSharesLevel'] = pd.to_numeric(df['HoldingSharesLevel'], errors='coerce')
        df['percent']  = pd.to_numeric(
            df['percent'].astype(str).str.replace('%', '', regex=False).str.strip(),
            errors='coerce'
        )
        df = df.dropna(subset=['percent', 'HoldingSharesLevel'])
        large = df[df['HoldingSharesLevel'].isin(LARGE_HOLDER_LEVELS)]
        if large.empty:
            return None
        agg = large.groupby(['date', 'stock_id'], as_index=False)['percent'].sum()
        return agg.pivot(index='date', columns='stock_id', values='percent')
    except Exception as e:
        print(f"⚠ 讀取大戶 matrix 失敗: {e}")
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


def detect_triple_buy_signals(matrices, fi_matrix, it_matrix, dealer_matrix):
    """三大法人同步買超:當日 外資 > 0 AND 投信 > 0 AND 自營 > 0
    最強籌碼共識訊號,但樣本少;適合用 OR 跟其他訊號搭配看「擇優」。
    """
    close = matrices['close']
    if any(m is None for m in (fi_matrix, it_matrix, dealer_matrix)):
        return pd.DataFrame(False, index=close.index, columns=close.columns)
    fi = fi_matrix.reindex(index=close.index, columns=close.columns)
    it = it_matrix.reindex(index=close.index, columns=close.columns)
    dl = dealer_matrix.reindex(index=close.index, columns=close.columns)
    return (fi > 0) & (it > 0) & (dl > 0)


def detect_resonance_signals(matrices, retail_matrix):
    """籌碼共振(簡化版):本週散戶比例 < 上週(散戶↓ → 籌碼集中)。
    holders 是週資料,daily 是日資料 → 用 reindex + ffill 把週訊號延伸到該週每個交易日。
    """
    close = matrices['close']
    if retail_matrix is None:
        return pd.DataFrame(False, index=close.index, columns=close.columns)
    delta = retail_matrix.diff()           # 週對週變化
    weekly_signal = delta < 0              # 散戶比例下降即觸發
    daily_signal = weekly_signal.reindex(
        index=close.index, columns=close.columns, method='ffill'
    )
    return daily_signal.fillna(False).astype(bool)


def detect_ma20_pullback_signals(matrices):
    """MA20 拉回守穩(葛蘭碧第二法則):
    1. 過去 10 日內至少 7 日 close > MA20(持續站上,確認多頭結構)
    2. 過去 3 日內至少 1 日 low <= MA20(有拉回觸碰 MA20)
    3. 今日 close > MA20 AND close > 昨日 close(重新站上 + 紅 K 守穩)
    """
    close = matrices['close']
    low   = matrices['low']
    ma20  = close.rolling(window=20, min_periods=20).mean()

    above_ma20  = close > ma20
    days_above  = above_ma20.rolling(window=10, min_periods=10).sum()
    cond1 = days_above >= 7

    touched_ma20 = low <= ma20
    cond2 = touched_ma20.rolling(window=3, min_periods=3).sum() >= 1

    cond3 = (close > ma20) & (close > close.shift(1))

    return cond1 & cond2 & cond3


def _load_double_red_revenue_matrix(cache_dir, daily_index, daily_columns):
    """讀 revenue parquet,組成「該股最新已公告月營收為雙紅(YoY>10% 且 MoM>0)」的 daily flag matrix。

    對齊邏輯:月營收約於次月 10~15 日公告(MOPS),這裡用「次月 15 日」為近似 effective_date,
    用 forward-fill 把月旗標延伸到該月之後的每個交易日(直到下一筆新營收出爐)。
    """
    try:
        files = sorted(cache_dir.glob('revenue_*.parquet'))
        if not files:
            return None
        df = pd.read_parquet(files[-1])
        if df.empty or 'revenue' not in df.columns:
            return None
        df = df.sort_values(['stock_id', 'revenue_year', 'revenue_month']).copy()

        # 算 MoM:同股上一筆營收
        df['rev_mom_base'] = df.groupby('stock_id')['revenue'].shift(1)
        df['mom'] = (df['revenue'] / df['rev_mom_base'] - 1) * 100

        # 算 YoY:找該股 12 個月前同月份
        df['_ym'] = df['revenue_year'] * 12 + df['revenue_month']
        df['_self_key']  = df['stock_id'] + '_' + df['_ym'].astype(str)
        df['_yoy_key']   = df['stock_id'] + '_' + (df['_ym'] - 12).astype(str)
        _lookup = df.drop_duplicates('_self_key').set_index('_self_key')['revenue']
        df['rev_yoy_base'] = df['_yoy_key'].map(_lookup)
        df['yoy'] = (df['revenue'] / df['rev_yoy_base'] - 1) * 100

        # 雙紅:YoY > 10% 且 MoM > 0
        df['double_red'] = (df['yoy'] > 10) & (df['mom'] > 0)

        # effective_date = 公告日近似(次月 15 號)
        df['effective_year']  = df['revenue_year']
        df['effective_month'] = df['revenue_month'] + 1
        _overflow = df['effective_month'] > 12
        df.loc[_overflow, 'effective_year']  = df.loc[_overflow, 'effective_year'] + 1
        df.loc[_overflow, 'effective_month'] = 1
        df['effective_date'] = pd.to_datetime(
            df['effective_year'].astype(str) + '-' +
            df['effective_month'].astype(str).str.zfill(2) + '-15',
            errors='coerce'
        )

        # 建 daily matrix:每個股每個交易日是否處於「雙紅生效中」
        result = pd.DataFrame(False, index=daily_index, columns=daily_columns)
        for sid, grp in df.groupby('stock_id'):
            if sid not in daily_columns:
                continue
            grp = grp.sort_values('effective_date').dropna(subset=['effective_date'])
            if grp.empty:
                continue
            stock_s = pd.Series(grp['double_red'].values, index=grp['effective_date'])
            # 同月若有多筆(理論上不會),保留最後一筆
            stock_s = stock_s[~stock_s.index.duplicated(keep='last')]
            stock_s = stock_s.reindex(daily_index, method='ffill').fillna(False)
            result[sid] = stock_s.astype(bool)
        return result
    except Exception as e:
        print(f"⚠ 讀取雙紅營收 matrix 失敗: {e}")
        return None


def detect_revenue_breakout_signals(matrices, breakout_matrix, double_red_matrix):
    """月營收雙紅突破:當日突破 60 日新高 AND 該股目前最新營收為雙紅(YoY>10% 且 MoM>0)。
    結合基本面 + 技術面,過濾掉「炒短線無業績」的假突破。
    """
    close = matrices['close']
    if double_red_matrix is None:
        return pd.DataFrame(False, index=close.index, columns=close.columns)
    dr = double_red_matrix.reindex(index=breakout_matrix.index,
                                   columns=breakout_matrix.columns).fillna(False).astype(bool)
    return breakout_matrix & dr


def detect_ma_golden_cross_signals(matrices):
    """MA 黃金交叉:MA20 上穿 MA60(中長線轉折),稀少但可靠。"""
    close = matrices['close']
    ma20 = close.rolling(window=20, min_periods=20).mean()
    ma60 = close.rolling(window=60, min_periods=60).mean()
    above       = ma20 > ma60
    above_prev  = ma20.shift(1) <= ma60.shift(1)
    not_na = ma20.notna() & ma60.notna() & ma20.shift(1).notna() & ma60.shift(1).notna()
    return above & above_prev & not_na


def detect_quality_breakout_signals(matrices, breakout_matrix):
    """品質突破:當日突破 AND 突破前 10 日內有 ≥ 6 日量 < MA20 × 0.7(明顯量縮整理過)。
    過濾「持續飆漲後勉強突破」,只保留「整理蓄勢後爆量突破」,品質更高。
    """
    close = matrices['close']
    vol   = matrices['volume']
    vol_ma20 = vol.rolling(window=20, min_periods=20).mean()
    is_low_vol = vol < vol_ma20 * 0.7
    quiet_count = is_low_vol.shift(1).rolling(window=10, min_periods=10).sum()
    return breakout_matrix & (quiet_count >= 6)


def detect_margin_short_squeeze_signals(matrices, margin_mats):
    """資減券增(軋空力道):
    - 近 5 日融資餘額累積跌幅 ≥ 3%(多頭認賠出場)
    - 近 5 日融券餘額累積增幅 ≥ 5%(空頭加碼)
    意義:籌碼結構轉為「散戶逃 + 空頭壓」→ 容易被軋空,股價往上反彈機率高
    """
    close = matrices['close']
    if margin_mats is None:
        return pd.DataFrame(False, index=close.index, columns=close.columns)
    margin = margin_mats['margin'].reindex(index=close.index, columns=close.columns)
    short  = margin_mats['short'].reindex(index=close.index, columns=close.columns)

    # 5 日累積變化率(%);融資餘額為 0 的股(沒掛融資)會出現 NaN/inf,自動被 mask 掉
    margin_5d_chg = (margin / margin.shift(5) - 1) * 100
    short_5d_chg  = (short  / short.shift(5)  - 1) * 100

    cond = (margin_5d_chg < -3) & (short_5d_chg > 5)
    return cond.fillna(False) & margin.shift(5).notna() & short.shift(5).notna()


def detect_momentum_top_signals(matrices, window: int = 20, top_pct: float = 0.10):
    """中期動能 Top 10%(Jegadeesh-Titman 1993 經典動能因子):
    個股近 window 日報酬,排名在全市場前 top_pct 分位。
    比 rs_market(贏中位數)更嚴格,只選真正的強勢股。
    """
    close = matrices['close']
    ret_n = close.pct_change(window)
    # 每日跨股票算分位數;若該日有效股數太少(< 100),threshold 可能 NaN,則該日無訊號
    threshold = ret_n.quantile(1 - top_pct, axis=1)
    return ret_n.gt(threshold, axis=0).fillna(False) & ret_n.notna()


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
# Tier R:反轉/抄底訊號(找「低檔即將起漲」,與追強勢方向相反)
# ══════════════════════════════════════════════════════════════════════
def detect_washout_bounce_signals(matrices,
                                  quiet_days: int = 5, quiet_ratio: float = 0.8,
                                  surge_ratio: float = 1.5):
    """跌深量縮後帶量翻揚(washout bounce):賣壓耗盡 → 第一根帶量紅K 起漲。

    三條件同時成立:
    1. 跌深/中期偏弱:今日 close < MA60(站在季線之下)
    2. 量縮築底:前 quiet_days 日(不含今日)平均量 < MA20量 × quiet_ratio(賣壓耗盡)
    3. 今日帶量翻揚:今日量 > MA20量 × surge_ratio,且今日紅K(close > open;無 open 則用 close > 昨收)

    意義:跌深後沒人想賣了(量縮),突然一根帶量紅K = 買方回頭,常是反轉起點。
    Returns: DataFrame index=date, columns=stock_id, values=bool
    """
    close = matrices['close']
    vol   = matrices['volume']
    open_ = matrices.get('open')

    ma60     = close.rolling(60, min_periods=60).mean()
    vol_ma20 = vol.rolling(20, min_periods=20).mean()

    below_ma60 = close < ma60
    # 前 quiet_days 日(shift(1) 排除今日)平均量偏低 = 量縮築底
    prior_avg_vol = vol.shift(1).rolling(quiet_days, min_periods=quiet_days).mean()
    quiet = prior_avg_vol < vol_ma20 * quiet_ratio
    # 今日帶量
    vol_surge = vol > vol_ma20 * surge_ratio
    # 今日紅K(優先用開盤;無開盤資料則退回「今日收 > 昨收」)
    if open_ is not None:
        red_k = (close > open_) & (close > close.shift(1))
    else:
        red_k = close > close.shift(1)

    sig = below_ma60 & quiet & vol_surge & red_k
    return sig.fillna(False) & ma60.notna() & vol_ma20.notna()


def detect_kd_bull_divergence_signals(matrices, window: int = 12,
                                      low_zone: float = 35.0, kd_gap: float = 3.0):
    """KD 低檔背離(bullish divergence):價格創新低,但 KD 沒跟著破 → 下跌動能衰竭。

    條件(收緊版,避免「只是比 N 日前低」就觸發,提高篩選力):
    1. 價格創 window 日**新低**:close ≤ 近 window 日最低收盤(真的破底)
    2. KD **沒**創新低:今日 K > 近 window 日 K 的最低 + kd_gap(明顯高於區間低點 = 背離)
    3. 低檔區:今日 K < low_zone(確保是「底部」背離,不是高檔)

    意義:價格新低但指標不再新低 = 賣壓邊際遞減,常領先價格落底翻揚。
    Returns: DataFrame index=date, columns=stock_id, values=bool
    """
    K, _ = _calc_kd_matrix(matrices)
    close = matrices['close']

    price_new_low = close <= close.rolling(window, min_periods=window).min()
    kd_higher_low = K > (K.rolling(window, min_periods=window).min() + kd_gap)
    in_low_zone   = K < low_zone

    sig = price_new_low & kd_higher_low & in_low_zone
    return sig.fillna(False) & K.notna()


def detect_chip_accumulation_signals(matrices, large_matrix, min_weekly_rise: float = 0.05):
    """大戶逆勢增持(低檔吃貨):大戶持股率上升,但股價還沒漲上去。

    條件:
    1. 大戶% 週對週上升 > min_weekly_rise(主力在吃貨)
    2. 今日 close < MA20(股價還在低檔/沒起漲 → 是「逆勢吃貨」而非追高)

    holders 是週資料 → 把週訊號 ffill 延伸到該週每個交易日(同 detect_resonance_signals)。
    意義:底部最可靠的領先訊號是「價平/價跌但籌碼在集中」,聰明錢常先price 一步動。
    Returns: DataFrame index=date, columns=stock_id, values=bool
    """
    close = matrices['close']
    if large_matrix is None:
        return pd.DataFrame(False, index=close.index, columns=close.columns)
    delta = large_matrix.diff()               # 週對週大戶%變化
    accumulating = delta > min_weekly_rise    # 大戶逆勢增持(週)
    daily = accumulating.reindex(
        index=close.index, columns=close.columns, method='ffill'
    ).fillna(False).astype(bool)
    ma20 = close.rolling(20, min_periods=20).mean()
    weak = (close < ma20).fillna(False)       # 股價還沒起漲(低檔)
    return daily & weak


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
def _dedup_overlap(trades_df: pd.DataFrame, hold_days: int, trading_dates) -> pd.DataFrame:
    """同檔股票 hold_days 個交易日內的第二次觸發 → 忽略(避免持倉重疊灌水)。
    保留每段「進場 → 出場」週期的第一筆訊號,後續同股觸發要等過了出場日才算下一筆。
    """
    if trades_df.empty:
        return trades_df
    # 建 trading date → ordinal index 對映(避免用日曆天差,週末會偏差)
    date_idx = {d: i for i, d in enumerate(trading_dates)}
    keep_rows = []
    for _, grp in trades_df.sort_values(['stock_id', 'date']).groupby('stock_id'):
        last_exit_idx = -1
        for _, row in grp.iterrows():
            entry_idx = date_idx.get(row['date'], -1)
            if entry_idx < 0:
                continue
            if entry_idx > last_exit_idx:           # 上一筆已出場才算新交易
                keep_rows.append(row)
                last_exit_idx = entry_idx + hold_days
    if not keep_rows:
        return pd.DataFrame(columns=trades_df.columns)
    return pd.DataFrame(keep_rows).reset_index(drop=True)


def compute_signal_returns(signal_matrix, close_matrix, hold_days: int,
                           dedup_within_hold: bool = True,
                           open_matrix=None) -> pd.DataFrame:
    """對訊號觸發點,算「進場後 hold_days 個交易日的報酬率」。

    Args:
        signal_matrix: bool DataFrame index=date, columns=stock_id (True 表觸發)
        close_matrix:  收盤價 DataFrame 同 shape
        hold_days:     持有交易日數
        dedup_within_hold: True = 同股票 hold_days 內第二次訊號會被忽略(避免持倉重疊
                       與強勢股權重灌水);False = 每天觸發都算一筆(原始逐日掃描)。
        open_matrix:   隔日開盤價 DataFrame;提供時進場價改為「訊號日隔日開盤 + 滑價」,
                       去除前視偏誤;None 時回退為訊號日收盤(舊行為)。

    Returns:
        DataFrame[date, stock_id, entry_price, exit_close, return_pct]
        return_pct = (exit_close - entry_price) / entry_price * 100
    """
    if signal_matrix is None or signal_matrix.empty:
        return pd.DataFrame()

    # exit = 持有後 N 個交易日的收盤
    exit_close = close_matrix.shift(-hold_days)

    # 進場價:隔日開盤(shift(-1)) + 滑價;無開盤資料時用當日收盤
    if open_matrix is not None:
        entry_open_next = open_matrix.shift(-1)   # 把 T+1 的開盤對齊到 T 行
        # 有效性遮罩:開盤必須為正數才採用
        use_open = entry_open_next.notna() & (entry_open_next > 0)
        entry_price_mat = entry_open_next.where(use_open, close_matrix) * (1 + SLIPPAGE_PCT / 100)
    else:
        # 無開盤資料(舊 daily 無 open 欄)→ 退回當日收盤進場,仍加滑價,與 performance.py 口徑一致
        entry_price_mat = close_matrix * (1 + SLIPPAGE_PCT / 100)

    # 把訊號矩陣 stack 成長表
    sig_long = signal_matrix.stack()
    sig_long = sig_long[sig_long].reset_index()
    sig_long.columns = ['date', 'stock_id', '_flag']

    if sig_long.empty:
        return pd.DataFrame()

    entry_long = entry_price_mat.stack().reset_index()
    entry_long.columns = ['date', 'stock_id', 'entry_price']

    exit_long = exit_close.stack().reset_index()
    exit_long.columns = ['date', 'stock_id', 'exit_close']

    merged = (
        sig_long[['date', 'stock_id']]
        .merge(entry_long, on=['date', 'stock_id'], how='left')
        .merge(exit_long,  on=['date', 'stock_id'], how='left')
    )
    merged = merged.dropna(subset=['entry_price', 'exit_close'])
    merged = merged[merged['entry_price'] > 0]

    merged['return_pct'] = (
        (merged['exit_close'] - merged['entry_price']) / merged['entry_price'] * 100
    )
    # 過濾離譜值(資料壞掉造成的 1000% 漲跌)
    merged = merged[merged['return_pct'].abs() < 100]
    merged = merged.sort_values('date').reset_index(drop=True)

    if dedup_within_hold:
        merged = _dedup_overlap(merged, hold_days, list(close_matrix.index))

    return merged


def summarize(trades: pd.DataFrame, hold_days: int = 10) -> dict:
    """對一組交易紀錄算統計摘要 + 風險指標。

    Args:
        trades: 含 'date', 'return_pct' 的交易紀錄
        hold_days: 持有天數,用於年化 Sharpe

    新增:
    - mdd:    最大回檔 (%),累積資金曲線從高點到低點的最大跌幅
    - sharpe: 年化夏普值 (avg/std × √(252/hold_days)),衡量風險調整後報酬
    """
    if trades is None or trades.empty:
        return {"n": 0}
    rets = trades.sort_values('date')['return_pct'].values
    wins = (rets > 0).sum()

    # ── 最大回檔(MDD):假設等權部位、序列交易、複利累積 ──
    equity = np.cumprod(1 + rets / 100.0)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    mdd = float(drawdowns.min() * 100) if len(drawdowns) > 0 else 0.0   # 負數,例 -15.3

    # ── 年化 Sharpe:用 hold_days 推算「一年大約幾次交易」做年化 ──
    std = float(rets.std()) if len(rets) > 1 else 0.0
    if std > 0 and hold_days > 0:
        trades_per_year = 252.0 / hold_days
        sharpe = (rets.mean() / std) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    return {
        "n":             len(trades),
        "win_rate":      wins / len(trades),
        "avg_return":    rets.mean(),
        "median_return": float(pd.Series(rets).median()),
        "max_return":    float(rets.max()),
        "min_return":    float(rets.min()),
        "std":           std,
        "mdd":           mdd,
        "sharpe":        sharpe,
    }


# ══════════════════════════════════════════════════════════════════════
# 對外主介面
# ══════════════════════════════════════════════════════════════════════
def build_signal_matrices(cache_dir) -> dict:
    """建構全部訊號矩陣(數量見 SIGNAL_LABELS)+ matrices + open_matrix 並回傳。

    耗時 ~5-7 秒,供 UI 獨立 cache 後重複使用。
    Returns: {"matrices": ..., "sig_matrices": {...}, "open_matrix": ...}
    """
    matrices = _load_daily_matrix(cache_dir)
    if matrices is None:
        return None

    # 排除 ETF(代號 00 開頭)欄位:個股訊號回測不含 ETF,且避免 ETF 拉動 momentum_top 的
    # 跨股分位數 / rs_market 的跨股中位數(兩者對 close 全欄位算)。ETF 價格仍保留在 daily 快取
    # 供「ETF 期貨」分頁帶價(_load_latest_close_map 另讀),這裡只清訊號掃描用的矩陣。
    _etf_cols = [c for c in matrices['close'].columns if str(c).startswith("00")]
    if _etf_cols:
        matrices = {k: v.drop(columns=[c for c in _etf_cols if c in v.columns])
                    for k, v in matrices.items() if v is not None}

    fi_matrix     = _load_foreign_net_matrix(cache_dir)
    it_matrix     = _load_it_net_matrix(cache_dir)
    dealer_matrix = _load_dealer_net_matrix(cache_dir)
    retail_matrix = _load_retail_pct_matrix(cache_dir)
    large_matrix  = _load_large_holder_pct_matrix(cache_dir)
    margin_mats   = _load_margin_matrices(cache_dir)
    double_red_matrix = _load_double_red_revenue_matrix(
        cache_dir, matrices['close'].index, matrices['close'].columns
    )

    _breakout = detect_breakout_signals(matrices)
    sig_matrices = {
        "resonance":        detect_resonance_signals(matrices, retail_matrix),
        "foreign":          detect_foreign_signals(matrices, fi_matrix),
        "revenue_breakout": detect_revenue_breakout_signals(matrices, _breakout, double_red_matrix),
        "margin_squeeze":   detect_margin_short_squeeze_signals(matrices, margin_mats),
        "triple_buy":       detect_triple_buy_signals(matrices, fi_matrix, it_matrix, dealer_matrix),
        "investment_trust": detect_investment_trust_signals(matrices, it_matrix),
        "breakout":         _breakout,
        "momentum_top":     detect_momentum_top_signals(matrices),
        "quality_breakout": detect_quality_breakout_signals(matrices, _breakout),
        "ma_golden_cross":  detect_ma_golden_cross_signals(matrices),
        "kd_cross":         detect_kd_cross_signals(matrices),
        # Tier R 反轉/抄底
        "washout_bounce":      detect_washout_bounce_signals(matrices),
        "kd_bull_divergence":  detect_kd_bull_divergence_signals(matrices),
        "chip_accumulation":   detect_chip_accumulation_signals(matrices, large_matrix),
    }
    return {
        "matrices":     matrices,
        "sig_matrices": sig_matrices,
        "open_matrix":  matrices.get('open'),
    }


def run_backtest(cache_dir, signal="breakout", hold_days: int = 10,
                 date_range: tuple = None, combine_mode: str = "and",
                 dedup_within_hold: bool = True,
                 stock_filter: str = None,
                 precomputed: dict = None) -> dict:
    """跑一次回測。

    Args:
        cache_dir:    pathlib.Path 指向 CACHE_DIR
        signal:       'breakout' / 'foreign' / 'kd_cross' 或 list
        hold_days:    持有交易日數(5/10/20)
        date_range:   (start_date, end_date) 兩個 pd.Timestamp;None = 全部
        precomputed:  build_signal_matrices() 的回傳值;提供時略過重建矩陣(加速)

    Returns:
        {
            "signal": 'breakout',
            "hold_days": 10,
            "trades": DataFrame[date, stock_id, entry_price, exit_close, return_pct],
            "stats": {"n": ..., "win_rate": ..., ...},
            "all_signals_stats": {...},
            "error": str (僅失敗時),
        }
    """
    # selected_list 提前算,供下方驗證 precomputed 是否包含所選訊號
    _selected_check = [signal] if isinstance(signal, str) else list(signal)

    # 防禦性:precomputed 必須是「含 matrices + sig_matrices 的新格式 dict」才採用。
    # 否則(None、或部署後 Streamlit 殘留的舊格式/舊訊號集 cache、或版本不同步)一律自行重建。
    # 第二道:即使結構對,但所選訊號不在 sig_matrices 裡(= 新增訊號後 cache 沒更新)也要重建,
    # 否則會「靜默回 0 觸發」(線上曾因 cache 殘留舊結構崩潰 / 漏新訊號回 0,兩次都是這類)。
    _valid_pre = (
        isinstance(precomputed, dict)
        and "matrices" in precomputed
        and isinstance(precomputed.get("sig_matrices"), dict)
        and all(s in precomputed["sig_matrices"] for s in _selected_check)
    )
    if _valid_pre:
        matrices     = precomputed["matrices"]
        sig_matrices = precomputed["sig_matrices"]
        open_matrix  = precomputed.get("open_matrix")
        if open_matrix is None and isinstance(matrices, dict):
            open_matrix = matrices.get("open")
    else:
        built = build_signal_matrices(cache_dir)
        if built is None:
            return {"error": "無法讀取 daily 快取", "trades": pd.DataFrame(), "stats": {"n": 0}}
        matrices     = built["matrices"]
        sig_matrices = built["sig_matrices"]
        open_matrix  = built.get("open_matrix")

    # ── 個股回測過濾:若指定 stock_filter,只保留該股票欄位 ──
    if stock_filter:
        _sid = str(stock_filter).strip()
        if _sid in matrices['close'].columns:
            for k in list(sig_matrices.keys()):
                # 保留該股一欄,其他欄位全設 False(等同過濾)
                _mat = sig_matrices[k]
                _filtered = pd.DataFrame(False, index=_mat.index, columns=_mat.columns)
                _filtered[_sid] = _mat[_sid]
                sig_matrices[k] = _filtered
        else:
            # 找不到該股 → 全部設空,讓上層 stats['n']=0 觸發提示
            for k in list(sig_matrices.keys()):
                sig_matrices[k] = pd.DataFrame(False,
                                              index=sig_matrices[k].index,
                                              columns=sig_matrices[k].columns)

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
        trades = compute_signal_returns(
            _slice_by_date(sig_mat), matrices['close'], hold_days,
            dedup_within_hold=dedup_within_hold, open_matrix=open_matrix,
        )
        all_signals_stats[sig_name] = summarize(trades, hold_days=hold_days)

    # 算「選定組合」的交易
    if len(selected_list) == 1 and selected_list[0] in sig_matrices:
        combined_mat = _slice_by_date(sig_matrices[selected_list[0]])
    else:
        combined_mat = combine_signals(sig_matrices, selected_list, mode=combine_mode)
        if combined_mat is not None:
            combined_mat = _slice_by_date(combined_mat)

    selected_trades = (
        compute_signal_returns(
            combined_mat, matrices['close'], hold_days,
            dedup_within_hold=dedup_within_hold, open_matrix=open_matrix,
        )
        if combined_mat is not None else pd.DataFrame()
    )

    return {
        "signal":            signal,
        "selected_signals":  selected_list,
        "combine_mode":      combine_mode,
        "hold_days":         hold_days,
        "trades":            selected_trades,
        "stats":             summarize(selected_trades, hold_days=hold_days),
        "all_signals_stats": all_signals_stats,
    }
