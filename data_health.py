"""資料健康度監控

針對 daily / institutional / chips parquet 跑一系列檢查,
偵測 fetch_cache 過程可能發生的資料異常,在 TG header 與 UI 顯示警告。

檢查項目:
1. 今日總成交量 vs 20 日均量 → 低於 50% 視為異常(可能部分股票漏抓)
2. 個股 close = 0 或 NaN → 抓取失敗
3. 散戶持股率不在 [0, 100] → chips 資料壞掉
4. 法人資料筆數驟降 → 三大法人 fetch 異常
"""
import pandas as pd


def check_data_health(cache_dir) -> dict:
    """跑全套健康度檢查。

    Returns:
        {
            "level": "ok" | "warn" | "error",
            "issues": ["說明1", "說明2", ...],
            "summary": "一行摘要"
        }
    """
    issues = []
    level = "ok"

    # ── 1. daily volume 異常 ────────────────────────────────
    try:
        files = sorted(cache_dir.glob('daily_*.parquet'))
        if not files:
            return {"level": "error", "issues": ["找不到 daily parquet"], "summary": "daily 快取缺失"}

        df = pd.read_parquet(files[-1], columns=['stock_id', 'date', 'close', 'Trading_Volume'])
        df['date'] = pd.to_datetime(df['date'])
        dates = sorted(df['date'].unique())

        if len(dates) >= 21:
            latest = dates[-1]
            # 算「最新日總量」與「前 20 日均日總量」
            latest_total = df[df['date'] == latest]['Trading_Volume'].sum()
            prev20_dates = dates[-21:-1]
            prev20_avg = df[df['date'].isin(prev20_dates)].groupby('date')['Trading_Volume'].sum().mean()
            if prev20_avg > 0:
                vol_ratio = latest_total / prev20_avg
                if vol_ratio < 0.3:
                    issues.append(
                        f"最新日總量僅前 20 日均量的 {vol_ratio*100:.0f}%(可能大量股票資料缺失)"
                    )
                    level = "error"
                elif vol_ratio < 0.5:
                    issues.append(
                        f"最新日總量偏低(僅前 20 日均量 {vol_ratio*100:.0f}%,可能部分股票漏抓)"
                    )
                    level = "warn" if level == "ok" else level

        # ── 2. 個股 close 異常 ──────────────────────────────
        latest_df = df[df['date'] == dates[-1]]
        bad_close = latest_df[(latest_df['close'] <= 0) | (latest_df['close'].isna())]
        if len(bad_close) > 0:
            bad_sids = bad_close['stock_id'].astype(str).head(5).tolist()
            issues.append(
                f"{len(bad_close)} 檔個股 close 為 0 或 NaN(例:{', '.join(bad_sids)})"
            )
            level = "error" if len(bad_close) > 10 else ("warn" if level == "ok" else level)

        # ── 2b. 日期一致性:檢查多少股票卡在舊日期(部分 yfinance 漏抓的徵兆) ──
        # 每檔股票的最新一筆日期,理論上應該都 = dates[-1]
        # 若大量股票卡在舊日期,代表抓取過程有缺漏
        per_stock_latest = df.groupby('stock_id')['date'].max()
        global_latest = dates[-1]
        stuck = per_stock_latest[per_stock_latest < global_latest]
        total_stocks = len(per_stock_latest)
        if total_stocks > 0:
            stuck_ratio = len(stuck) / total_stocks
            if stuck_ratio > 0.5:
                stuck_sample = stuck.index.astype(str).tolist()[:5]
                issues.append(
                    f"{len(stuck)}/{total_stocks} 檔股票最新日期非 {global_latest.strftime('%Y-%m-%d')}"
                    f"(占 {stuck_ratio*100:.0f}%,例:{', '.join(stuck_sample)})"
                )
                level = "error"
            elif stuck_ratio > 0.2:
                stuck_sample = stuck.index.astype(str).tolist()[:5]
                issues.append(
                    f"{len(stuck)}/{total_stocks} 檔股票最新日期非 {global_latest.strftime('%Y-%m-%d')}"
                    f"(占 {stuck_ratio*100:.0f}%,例:{', '.join(stuck_sample)})"
                )
                level = "warn" if level == "ok" else level

    except Exception as e:
        issues.append(f"daily 健康度檢查失敗: {str(e)[:80]}")
        level = "warn" if level == "ok" else level

    # ── 3. chips(散戶持股率)異常 ────────────────────────
    try:
        chip_files = sorted(cache_dir.glob('chips_*.parquet'))
        if chip_files:
            chip_df = pd.read_parquet(chip_files[-1], columns=['stock_id', 'date', 'percent'])
            chip_df['date'] = pd.to_datetime(chip_df['date'])
            latest_chip_date = chip_df['date'].max()
            latest_chips = chip_df[chip_df['date'] == latest_chip_date]
            # percent 可能是字串(歷史 bug),用 to_numeric 安全轉
            percents = pd.to_numeric(latest_chips['percent'], errors='coerce').dropna()
            if not percents.empty:
                bad = ((percents < 0) | (percents > 100)).sum()
                if bad > 0:
                    issues.append(f"chips 有 {bad} 筆持股率不在 0~100%(資料異常)")
                    level = "warn" if level == "ok" else level
    except Exception as e:
        # chips 缺失只算 info 級,不阻擋
        pass

    # ── 4. 法人資料筆數驟降 ──────────────────────────────
    try:
        inst_files = sorted(cache_dir.glob('institutional_*.parquet'))
        if inst_files:
            inst_df = pd.read_parquet(inst_files[-1], columns=['stock_id', 'date'])
            inst_df['date'] = pd.to_datetime(inst_df['date'])
            inst_dates = sorted(inst_df['date'].unique())
            if len(inst_dates) >= 6:
                latest_n = len(inst_df[inst_df['date'] == inst_dates[-1]])
                prev5_avg = (
                    inst_df[inst_df['date'].isin(inst_dates[-6:-1])]
                    .groupby('date').size().mean()
                )
                if prev5_avg > 0 and latest_n < prev5_avg * 0.5:
                    issues.append(
                        f"法人資料最新日僅 {latest_n} 筆,前 5 日均 {prev5_avg:.0f} 筆(fetch 可能失敗)"
                    )
                    level = "warn" if level == "ok" else level
    except Exception:
        pass

    # ── 摘要 ─────────────────────────────────────────────
    if not issues:
        summary = "資料健康度正常"
    elif level == "error":
        summary = f"資料嚴重異常({len(issues)} 項問題,僅供參考)"
    else:
        summary = f"資料部分異常({len(issues)} 項警告)"

    return {"level": level, "issues": issues, "summary": summary}


def format_health_for_tg(health: dict) -> str:
    """組 TG header 用的健康度警示(僅在 warn/error 時返回非空字串)。"""
    if health.get("level") == "ok":
        return ""
    icon = "❗" if health["level"] == "error" else "⚠️"
    text = f"{icon} <b>{health['summary']}</b>"
    # 列出最多 3 項問題
    for issue in health.get("issues", [])[:3]:
        text += f"\n   ・{issue}"
    return text + "\n"
