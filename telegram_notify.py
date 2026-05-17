import os
import json
import requests
import pandas as pd
import tempfile
import html
from pathlib import Path
from screening0515 import run_screening, PASS_SCORE, HIGH_BREAK_DAYS, CACHE_DIR

# 共用模組
from ai_helper import call_openrouter_ai
from cache_status import cache_freshness
from picks_history import load_history, save_history, compute_streak

# ── 環境變數 ───────────────────────────────────────────────────────────
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# ── 路徑與設定 ─────────────────────────────────────────────────────────
TG_MAX_LEN = 4000          # Telegram 單訊息上限為 4096,留 96 字餘裕
TOP_N_DISPLAY = 15         # 訊息列出前 N 檔


# ══════════════════════════════════════════════════════════════════════
# Telegram 送訊息(自動分段)
# ══════════════════════════════════════════════════════════════════════
def _split_for_telegram(text: str, max_len: int) -> list:
    """在換行邊界切分長訊息;單段絕不超過 max_len。
    HTML 標籤都在單一行內成對出現,只在「行與行之間」切就不會弄壞 <b>/<i>/<a>。
    """
    if len(text) <= max_len:
        return [text]

    chunks = []
    current = ""
    for line in text.split("\n"):
        # 罕見情況:單行超長(例如 AI 失控吐巨型句子),強制按字元切
        if len(line) > max_len:
            if current:
                chunks.append(current.rstrip())
                current = ""
            for i in range(0, len(line), max_len):
                chunks.append(line[i:i + max_len])
            continue

        # 加上這行會爆量 → 收一段
        if len(current) + len(line) + 1 > max_len:
            if current.strip():     # 防呆:避免 append 空 chunk(Telegram 會 400)
                chunks.append(current.rstrip())
            current = line + "\n"
        else:
            current += line + "\n"

    if current.strip():
        chunks.append(current.rstrip())
    return chunks


def send_telegram_message(text: str) -> bool:
    """透過 Telegram Bot API 發送 HTML 格式訊息,超過 TG_MAX_LEN 自動分段。"""
    if not TOKEN or not CHAT_ID:
        print("未設定 Telegram Token 或 Chat ID")
        return False

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    chunks = _split_for_telegram(text, TG_MAX_LEN)
    all_ok = True

    for i, chunk in enumerate(chunks, 1):
        payload = {
            "chat_id": CHAT_ID,
            "text": chunk,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,  # 避免每個 Yahoo 連結都展開縮圖
        }
        try:
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"[Telegram 第 {i}/{len(chunks)} 段發送失敗] {e}")
            all_ok = False
    return all_ok


# ══════════════════════════════════════════════════════════════════════
# 漲跌幅:從 daily parquet 讀最新兩個交易日的 close
# ══════════════════════════════════════════════════════════════════════
def load_change_pct_map() -> dict:
    """讀最新 daily parquet,算各檔最新交易日 vs 前一交易日漲跌幅。

    Returns:
        {sid_str: change_pct_float}  例如 {"2330": 2.34, "2454": -0.55}
        讀檔失敗或資料不足回 {}
    """
    try:
        files = sorted(CACHE_DIR.glob('daily_*.parquet'))
        if not files:
            return {}
        df = pd.read_parquet(files[-1], columns=['stock_id', 'date', 'close'])
        if df.empty:
            return {}
        df['date'] = pd.to_datetime(df['date'])
        dates = sorted(df['date'].unique())
        if len(dates) < 2:
            return {}
        latest, prev = dates[-1], dates[-2]

        latest_close = (
            df[df['date'] == latest]
            .drop_duplicates(subset='stock_id', keep='last')
            .set_index('stock_id')['close']
        )
        prev_close = (
            df[df['date'] == prev]
            .drop_duplicates(subset='stock_id', keep='last')
            .set_index('stock_id')['close']
        )

        common = latest_close.index.intersection(prev_close.index)
        pct = (latest_close.loc[common] - prev_close.loc[common]) / prev_close.loc[common] * 100

        # 過濾 NaN 與分母異常造成的離譜值
        result = {}
        for k, v in pct.items():
            if pd.notna(v) and abs(v) < 100:
                result[str(k)] = float(v)
        return result
    except Exception as e:
        print(f"⚠ 讀取漲跌幅失敗(將以空白顯示): {e}")
        return {}


def fmt_change_pct(pct) -> str:
    """格式化漲跌幅(台股慣例:紅漲綠跌)

    2.34  → '🔴+2.3%'
    -0.55 → '🟢-0.6%'
    0.0   → '⚪0.0%'
    0.03  → '⚪0.0%'   (四捨五入到 .1f 仍是 0,顯示為平盤避免「🔴+0.0%」這種矛盾)
    None  → ''
    """
    if pct is None or pd.isna(pct):
        return ""
    # 平盤閾值對齊顯示精度(.1f);避免出現「🔴+0.0%」(0.03% 漲但顯示 0% 看起來矛盾)
    if abs(pct) < 0.05:
        return "⚪0.0%"
    icon = "🔴" if pct > 0 else "🟢"   # 台股:紅漲 / 綠跌
    return f"{icon}{pct:+.1f}%"


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════
def main():
    try:
        # ── 0. 讀取歷史 ─────────────────────────────────────
        history = load_history()

        # ── 1. 執行選股 ─────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            df, _, meta = run_screening(pass_score=PASS_SCORE, output_dir=Path(tmpdir))

        # ── 2. 日期與安全提取 meta ──────────────────────────
        now_tpe = pd.Timestamp.now(tz="Asia/Taipei")
        today_str = now_tpe.strftime("%Y-%m-%d")
        date_str = now_tpe.strftime("%m/%d")

        # 同日重跑保護:剔除 history 中「今日」的項目,避免污染 yesterday_sids/streak/退場偵測
        history = [h for h in history if h.get("date") != today_str]
        yesterday_sids = set(history[-1]["sids"]) if history else set()

        def _safe_num(v, default=0.0):
            return float(v) if pd.notna(v) else default

        market_data_ok = meta.get('market_data_ok', False)
        twii_now_raw   = meta.get('twii_now')
        twii_pct       = _safe_num(meta.get('twii_pct'))
        twii_bias      = _safe_num(meta.get('twii_bias'))
        score_note     = meta.get('score_note', "")
        market_status  = meta.get('market_status', '未知')
        cache_max_date = meta.get('cache_max_date')
        twii_now       = twii_now_raw if pd.notna(twii_now_raw) else None

        # ── 3. 動態圖示 ─────────────────────────────────────
        if not market_data_ok or twii_now is None:
            icon = "⚪"; bias_icon = "⚪"
            twii_now_str  = "N/A"
            twii_pct_str  = "N/A"
            twii_bias_str = "N/A"
        else:
            icon          = "🔴" if twii_pct > 0 else ("🟢" if twii_pct < 0 else "⚪")  # 台股:紅漲綠跌
            bias_icon     = "📈" if twii_bias >= 0 else "📉"
            twii_now_str  = f"{twii_now:,.0f}"
            twii_pct_str  = f"{twii_pct:+.2f}%"
            twii_bias_str = f"{twii_bias:+.2f}%"

        # ── 4. 大盤戰情摘要 + 快取新鮮度警告 ────────────────
        header = (
            f"🚀 <b>台股戰情摘要 ({date_str})</b>\n"
            f"━━━━━━━━━━━━━━\n"
            f"📈 指數:{twii_now_str} ({icon}{twii_pct_str})\n"
            f"📐 乖離:{bias_icon}{twii_bias_str} (位階)\n"
            f"🌡️ 盤勢:<b>{market_status}</b>\n"
        )
        if score_note:
            header += f"❗ <b>{score_note}</b>\n"

        # 快取新鮮度:共用 cache_status,只在 warn/error/missing 等級才示警
        freshness = cache_freshness(cache_max_date)
        if freshness["level"] in ("warn", "error", "missing"):
            header += f"⚠️ <b>{freshness['msg']},僅供參考</b>\n"

        header += "━━━━━━━━━━━━━━\n\n"

        # ── 5. AI 點評(模型輪替) ──────────────────────────
        ai_comment = ""
        if df is not None and not df.empty:
            top_stock = df.iloc[0]
            sid_top   = str(top_stock['代號'])
            name_top  = top_stock['名稱']
            score_top = top_stock['總分']

            breakout_col = f"·{HIGH_BREAK_DAYS}日量價齊揚突破"
            is_breakout  = top_stock.get(breakout_col) == 1
            is_sync      = top_stock.get("★籌碼共振(大戶↑散戶↓)") == 1

            prompt = (
                f"你是一位台灣股市的資深量化分析師。今日大盤狀態:{market_status} ({twii_pct_str}),"
                f"共 {len(df)} 檔達標。\n"
                f"冠軍標的:{sid_top} {name_top}(總分 {score_top}/10 分)"
                f"{'、帶量突破' if is_breakout else ''}"
                f"{'、籌碼共振(大戶增散戶減)' if is_sync else ''}。\n"
                f"請用繁體中文(台灣用語)寫一段 60~80 字的盤後精闢點評,語氣專業客觀。\n"
                f"直接輸出純文字,絕對不要使用任何 Markdown 語法(不要星號、井號、反引號)。"
            )

            model_name, ai_text = call_openrouter_ai(prompt, max_tokens=250)
            if ai_text:
                ai_comment = (
                    f"🧠 <b>AI 分析師({model_name}):</b>\n"
                    f"<i>「{html.escape(ai_text)}」</i>\n"
                    f"━━━━━━━━━━━━━━\n"
                )

        # ── 6. 個股清單 + tags(新進 / 連 N 日 / 突破)+ 漲跌幅 ─
        change_pct_map = load_change_pct_map()  # {sid: pct}

        if df is None or len(df) == 0:
            content = "💡 目前盤勢較嚴峻,沒有股票達標。"
            today_set = set()
        else:
            content = f"🔥 <b>今日達標個股 (共 {len(df)} 檔)</b>\n"
            breakout_col = f"·{HIGH_BREAK_DAYS}日量價齊揚突破"
            today_set = set(df['代號'].astype(str).tolist())

            for _, row in df.head(TOP_N_DISPLAY).iterrows():
                sid_str    = str(row['代號'])
                score_icon = "🔥" if row['總分'] >= 9 else "•"
                stock_name = html.escape(str(row['名稱']))   # 防禦 ETF 名稱含 &/< 等字元

                # 漲跌幅(緊接在分數後,訊息密度高)
                change_str = fmt_change_pct(change_pct_map.get(sid_str))
                change_part = f" {change_str}" if change_str else ""

                tags = ""
                # [新進] / [連 N 日]:第一次跑(history 空)時都不標,避免誤報
                if history:
                    streak = compute_streak(sid_str, history)
                    if streak == 1:
                        tags += " <code>[新進]</code>"
                    elif streak >= 3:
                        tags += f" <code>[連{streak}日]</code>"
                    # streak == 2 不標,避免雜訊

                if row.get(breakout_col) == 1:
                    tags += " <code>[突破]</code>"
                # 已移除 [共振] tag(改用 AI 點評描述,清單聚焦於連續性與突破)

                content += (
                    f"{score_icon} "
                    f"<a href='https://tw.stock.yahoo.com/quote/{sid_str}'>{sid_str}</a> "
                    f"{stock_name} ({row['總分']}分){change_part}{tags}\n"
                )

            # 主流群聚 + 集中度警告
            if '產業' in df.columns and len(df) > 0:
                valid_industries = df['產業'].dropna()
                valid_industries = valid_industries[valid_industries != ""]
                if not valid_industries.empty:
                    top_industry = valid_industries.value_counts().idxmax()
                    top_count = int(valid_industries.value_counts().max())
                    concentration = top_count / len(df)
                    content += f"\n📊 <b>今日主流群聚:</b> {top_industry} ({top_count}檔)"
                    if concentration > 0.5:
                        content += (
                            f" ⚠️ <i>高度集中 ({concentration*100:.0f}%),"
                            f"留意系統性風險</i>"
                        )
                    content += "\n"

            if len(df) > TOP_N_DISPLAY:
                content += f"\n<i>...等其餘 {len(df) - TOP_N_DISPLAY} 檔請至網頁查看完整分析</i>\n"

        # 退場通知:昨在、今天沒了(history 空時跳過,避免首跑誤報)
        if history and yesterday_sids:
            exited = yesterday_sids - today_set
            if exited:
                content += f"\n📤 <b>今日退場:</b> {len(exited)} 檔"
                if len(exited) <= 8:
                    content += f" ({', '.join(sorted(exited))})"
                content += "\n"

        content += (
            f"\n🌐 <b>"
            f"<a href='https://catchstocktw.streamlit.app/'>開啟我的全自動選股儀表板</a>"
            f"</b>"
        )

        # ── 7. 合體並發送(自動分段) ──────────────────────
        send_ok = send_telegram_message(header + ai_comment + content)
        n_hit = len(df) if df is not None else 0
        if send_ok:
            print(f"✅ 推播完成!今日共 {n_hit} 檔達標。")
        else:
            print(f"⚠️ 推播失敗或部分失敗(已寫入歷史),今日 {n_hit} 檔達標。")

        # ── 8. 更新歷史 ─────────────────────────────────────
        current_sids = df['代號'].astype(str).tolist() if (df is not None and not df.empty) else []
        # 同一天重跑會覆蓋(取最後一次),legacy 也順便清掉
        history = [
            h for h in history
            if h.get("date") not in (today_str, "legacy")
        ]
        history.append({"date": today_str, "sids": current_sids})
        save_history(history)

    except Exception as e:
        safe_error = html.escape(str(e))
        error_msg = (
            f"❌ <b>選股機器人罷工求救!</b>\n\n"
            f"系統發生致命錯誤,請至 GitHub 檢查:\n<code>{safe_error}</code>"
        )
        send_telegram_message(error_msg)
        print(f"❌ 選股推播發生致命錯誤:{e}")


if __name__ == "__main__":
    main()
