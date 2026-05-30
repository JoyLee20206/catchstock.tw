"""煙霧測試(smoke test)——快速驗證核心模組不崩 + 資料契約完整。

用法:
    python smoke_test.py

特性:
- 不需 Streamlit,只讀現有 cache;純邏輯模組(picks_history / performance / backtest)。
- 缺對應 cache 的檢查會標 SKIP(印出原因),不算失敗 —— 方便本機/CI 在資料不齊時仍能跑。
- 只有「真的崩」或「資料契約被違反」(缺新欄位、數值越界、ETF 混入個股…)才算 FAIL。
- 回傳碼:全過/僅 SKIP → 0;任一 FAIL → 1(可接 GitHub Actions / pre-push)。

設計目的:把「還有沒有 bug」從人工檢視,變成可重複執行的安全網。
日後改回測進場、保證金、ETF 等邏輯後,跑一次這支就能抓到「欄位改名/結構不同步/口徑跑掉」這類回歸。
"""
import sys
from pathlib import Path
import pandas as pd

CACHE = Path("cache")
results = []  # list of (status, name, detail)


class _Skip(Exception):
    """資料不齊 → 跳過(非失敗)。"""


def _need(glob_pat: str) -> Path:
    fs = sorted(CACHE.glob(glob_pat))
    if not fs:
        raise _Skip(f"無 {glob_pat}")
    return fs[-1]


def check(name, fn):
    try:
        detail = fn()
        results.append(("PASS", name, detail or ""))
    except _Skip as s:
        results.append(("SKIP", name, str(s)))
    except Exception as e:
        results.append(("FAIL", name, f"{type(e).__name__}: {e}"))


# ── 1. picks_history(純邏輯,不需 cache)──────────────────────────────
def t_history():
    from picks_history import load_history, get_picks, get_sids
    h = load_history()
    assert isinstance(h, list), "load_history 應回 list"
    picks = sum(len(get_picks(e)) for e in h if e.get("date") != "legacy")
    sids = sum(len(get_sids(e)) for e in h if e.get("date") != "legacy")
    return f"{len(h)} entries / {picks} picks / {sids} sids"


# ── 2. performance 全流程(隔日開盤進場 + 各回測)────────────────────
def t_performance():
    _need("daily_*.parquet")
    from picks_history import load_history
    from performance import (
        _load_price_matrices, compute_performance, compute_equity_curve,
        backtest_market_filter, attribute_signals, backtest_exit_rules,
        compute_per_stock_performance, check_system_health, format_performance_summary,
    )
    m = _load_price_matrices(CACHE)
    assert m is not None and "close" in m, "_load_price_matrices 應回含 close 的 dict"
    h = load_history()
    perf = compute_performance(h, CACHE, n_days_list=(5, 10))
    assert "overall" in perf and "samples" in perf and "by_score" in perf
    # 下列允許「資料不足」回 None/[]/error,但不可拋例外
    compute_equity_curve(h, CACHE, hold_days=5)
    backtest_market_filter(h, CACHE, hold_days=5)
    attribute_signals(h, CACHE, hold_days=5)
    backtest_exit_rules(h, CACHE, max_hold=10)
    ps = compute_per_stock_performance(h, CACHE, hold_days=5)
    assert isinstance(ps, list), "compute_per_stock_performance 應回 list"
    sysh = check_system_health(h, CACHE, hold_days=5)
    assert sysh.get("status") in ("ok", "warn", "fail", "insufficient"), "system_health 狀態異常"
    format_performance_summary(perf)
    return (f"overall={len(perf['overall'])} 指標 / samples={len(perf['samples'])} / "
            f"per_stock={len(ps)} 檔 / health={sysh.get('status')}")


# ── 3. backtest 引擎(build_signal_matrices / run_backtest / 防禦性)──
def t_backtest():
    _need("daily_*.parquet")
    from backtest import build_signal_matrices, run_backtest
    pre = build_signal_matrices(CACHE)
    assert pre is not None, "build_signal_matrices 回 None"
    assert {"matrices", "sig_matrices", "open_matrix"} <= set(pre), "precomputed 缺鍵"
    # 訊號矩陣不應含 ETF(00 開頭)欄位
    etf = [c for c in pre["matrices"]["close"].columns if str(c).startswith("00")]
    assert not etf, f"訊號矩陣不該含 ETF 欄: {etf[:5]}"
    r1 = run_backtest(CACHE, signal=["foreign"], hold_days=5, precomputed=pre)
    assert "stats" in r1 and "trades" in r1
    if not r1["trades"].empty:
        assert "entry_price" in r1["trades"].columns, "trades 應有 entry_price 欄(非 entry_close)"
    # 防禦性:壞掉的 precomputed 應自動重建、結果一致,不可崩
    r2 = run_backtest(CACHE, signal=["foreign"], hold_days=5, precomputed={"garbage": 1})
    assert r2["stats"].get("n") == r1["stats"].get("n"), "壞 precomputed 未正確 fallback"
    return f"foreign n={r1['stats'].get('n')} / ETF已排除 / 防禦性 fallback OK"


# ── 4. 資料契約(新欄位、數值範圍、ETF 不混入個股)────────────────────
def t_schema_daily():
    cols = set(pd.read_parquet(_need("daily_*.parquet")).columns)
    assert {"stock_id", "date", "close"} <= cols, "daily 缺基本欄位"
    assert "open" in cols, "daily 缺 open 欄(隔日開盤進場需要)"
    return "stock_id/date/close/open 齊全"


def t_schema_stock_futures():
    df = pd.read_parquet(_need("stock_futures_*.parquet"))
    assert {"stock_id", "multiplier"} <= set(df.columns)
    if "eng_code" not in df.columns:
        raise _Skip("無 eng_code 欄(尚未重抓新版 fetch_cache)")
    etf = df[df.stock_id.astype(str).str.startswith("00")]
    assert etf.empty, f"stock_futures 不該含 ETF(00 開頭): {list(etf.stock_id.unique())[:5]}"
    return f"{df.stock_id.nunique()} 檔 / 含 eng_code / 無 ETF"


def t_schema_stock_margin():
    df = pd.read_parquet(_need("stock_margin_*.parquet"))
    assert {"stock_id", "init_rate"} <= set(df.columns)
    bad = df[(df.init_rate <= 0) | (df.init_rate > 100)]
    assert bad.empty, f"init_rate 應在 1~100%,異常: {bad.init_rate.tolist()[:5]}"
    return f"{len(df)} 檔 / init_rate 範圍 OK"


def t_schema_etf():
    df = pd.read_parquet(_need("etf_futures_*.parquet"))
    assert {"stock_id", "multiplier", "init_amt", "maint_amt"} <= set(df.columns)
    assert (df.init_amt >= df.maint_amt).all(), "原始保證金應 >= 維持保證金"
    assert df.stock_id.astype(str).str.startswith("00").all(), "ETF 代號應全為 00 開頭"
    assert df.multiplier.isin([1000, 10000]).all(), "ETF 乘數應為 1000(小型)或 10000(標準)"
    return f"{df.stock_id.nunique()} 檔 ETF / {len(df)} 契約"


CHECKS = [
    ("picks_history",         t_history),
    ("performance 全流程",     t_performance),
    ("backtest 引擎",          t_backtest),
    ("schema: daily(open)",   t_schema_daily),
    ("schema: stock_futures", t_schema_stock_futures),
    ("schema: stock_margin",  t_schema_stock_margin),
    ("schema: etf_futures",   t_schema_etf),
]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 64)
    print(" 煙霧測試 smoke_test.py")
    print("=" * 64)
    for name, fn in CHECKS:
        check(name, fn)

    icon = {"PASS": "✅", "SKIP": "⏭️ ", "FAIL": "❌"}
    for status, name, detail in results:
        print(f" {icon[status]} {status:4} {name:22} {detail}")

    n_fail = sum(1 for s, _, _ in results if s == "FAIL")
    n_skip = sum(1 for s, _, _ in results if s == "SKIP")
    n_pass = sum(1 for s, _, _ in results if s == "PASS")
    print("-" * 64)
    print(f" 結果:{n_pass} PASS / {n_skip} SKIP / {n_fail} FAIL")
    if n_fail:
        print(" ❌ 有檢查失敗,請看上面 FAIL 列。")
        return 1
    print(" ✅ 全部通過(SKIP 為資料未齊,非錯誤)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
