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
from picks_history import load_history, save_history, compute_streak, build_picks_from_df, get_sids, compute_hot_picks
from data_health import check_data_health, format_health_for_tg
from watchlist_alerts import check_watchlist, format_alerts_for_tg
from performance import compute_performance, format_performance_summary, check_system_health
from market_sentiment import compute_sentiment, format_sentiment_for_tg

WATCHLIST_FILE = str(CACHE_DIR / "watchlist.json")  # 由 UI 寫入,TG 讀取做警示

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
# 自選股 & 名稱對映
# ══════════════════════════════════════════════════════════════════════
def load_watchlist() -> list:
    """讀 UI 寫入的自選股清單。"""
    if not os.path.exists(WATCHLIST_FILE):
        return []
    try:
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [str(s) for s in data if str(s).strip()]
    except Exception as e:
        print(f"⚠ 讀取自選股失敗: {e}")
        return []


def load_name_map() -> dict:
    """從 daily parquet 拉 sid → 簡稱對映。"""
    try:
        files = sorted(CACHE_DIR.glob('daily_*.parquet'))
        if not files:
            return {}
        df = pd.read_parquet(files[-1], columns=['stock_id'])
        # daily 沒有 name 欄,改從 ID 對照(若有 name 欄會自動 fallback)
        # 實際上 screening0515 有 STOCK_NAMES 全域 dict,但為避免循環 import,這裡只做防護
        return {}
    except Exception:
        return {}


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
        # ✅ 修改為（透過 get_sids 函式安全提取，自動兼容新舊格式）：
        yesterday_sids = set(get_sids(history[-1])) if history else set()

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

        # 資料健康度檢查:總量、close、chips、法人筆數異常
        try:
            health = check_data_health(CACHE_DIR)
            health_text = format_health_for_tg(health)
            if health_text:
                header += health_text
        except Exception as e:
            print(f"⚠ 健康度檢查失敗(略過): {e}")

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

            # ── 補資料給 AI 看(讓點評更有層次,不要只重述冠軍訊息) ──
            today_sids = set(df['代號'].astype(str).tolist())

            # 1. 退場數(昨在今天沒了)
            n_exit = len(yesterday_sids - today_sids) if yesterday_sids else 0

            # 2. 主流產業(若有)
            industry_ctx = ""
            if '產業' in df.columns:
                ind_series = df['產業'].dropna()
                ind_series = ind_series[ind_series != ""]
                if not ind_series.empty:
                    top_ind = ind_series.value_counts().idxmax()
                    top_cnt = int(ind_series.value_counts().max())
                    pct_ind = top_cnt / len(df) * 100
                    industry_ctx = f"主流產業:{top_ind}({top_cnt}/{len(df)} 檔,占 {pct_ind:.0f}%)\n"

            # 3. 近 20 日熱度榜 TOP 3(最常上榜的強勢股)
            # 歷史保留已延長到一年(供績效回測),熱度榜需限定近期視窗,否則會混入舊資料
            hot_ctx = ""
            try:
                hot_picks = compute_hot_picks(history, top_n=3, window=20)
                if hot_picks:
                    hot_lines = []
                    for h in hot_picks:
                        sid_h = h['sid']
                        # 從 df 找名稱;找不到就只用代號
                        name_h = ""
                        row_match = df[df['代號'].astype(str) == sid_h]
                        if not row_match.empty:
                            name_h = str(row_match.iloc[0]['名稱'])
                        in_today = "★今日續入" if h['in_latest'] else "今日退場"
                        hot_lines.append(
                            f"{sid_h} {name_h}({h['hits']}/{h['total_days']}日,{in_today})"
                        )
                    hot_ctx = "近期熱門股:" + "、".join(hot_lines) + "\n"
            except Exception as _e:
                print(f"⚠ 熱度榜計算失敗,AI prompt 不含此項: {_e}")

            prompt = (
                f"你是台灣股市的盤後資深分析師,用 70~100 繁體中文字寫今日盤後重點。\n"
                f"\n"
                f"【今日盤勢】\n"
                f"大盤:{market_status}({twii_pct_str})\n"
                f"達標:{len(df)} 檔"
                f"{f',退場 {n_exit} 檔' if n_exit > 0 else ''}\n"
                f"{industry_ctx}"
                f"{hot_ctx}"
                f"\n"
                f"【冠軍標的】\n"
                f"{sid_top} {name_top}(總分 {score_top}/10)"
                f"{'、帶量突破' if is_breakout else ''}"
                f"{'、籌碼共振(大戶增散戶減)' if is_sync else ''}\n"
                f"\n"
                f"【分數系統說明】\n"
                f"本系統雖以 10 分為滿分,但實務最高僅見 8 分"
                f"(法人雙買+大戶散戶共振+RS強+月營收YoY同時成立極罕見)。\n"
                f"故 8 分視同冠軍級訊號,7 分為合格,6 分為邊緣"
                f"(僅在大盤資料缺失自動降標時出現)。\n"
                f"\n"
                f"【寫作規範】\n"
                f"1. 先點出今日「最值得注意的 1 個現象」"
                f"(例:產業集中度、退場潮、熱門股是否退場、冠軍特性)\n"
                f"2. 接一句具體的「明日該觀察什麼」提醒\n"
                f"3. 不要重複我給的數字。禁止使用以下廢話:"
                f"「持續觀察」「值得關注」「動能強勁」「投資者情緒」「淨流入」「穩健向上」\n"
                f"4. 簡單句,不堆疊形容詞,純文字無 Markdown\n"
            )

            model_name, ai_text = call_openrouter_ai(prompt, max_tokens=300)
            if ai_text:
                ai_comment = (
                    f"🧠 <b>AI 分析師({model_name}):</b>\n"
                    f"<i>「{html.escape(ai_text)}」</i>\n"
                    f"━━━━━━━━━━━━━━\n"
                )
            else:
                # 全部 AI 模型失敗 → 模板化 fallback 點評(不洗版、訊息密度仍夠)
                _fb_parts = [f"今日達標 {len(df)} 檔"]
                if n_exit > 0:
                    _fb_parts.append(f"退場 {n_exit} 檔")
                if name_top:
                    _fb_parts.append(f"冠軍 {sid_top} {name_top}(總分 {score_top}/10)")
                fallback_text = "、".join(_fb_parts) + "。AI 點評暫時無法產生(免費 API 過載),完整訊號見下方清單。"
                ai_comment = (
                    f"📊 <b>盤後重點:</b>\n"
                    f"<i>「{html.escape(fallback_text)}」</i>\n"
                    f"━━━━━━━━━━━━━━\n"
                )

        # ── 5b. 大盤情緒指標 ──────────────────────────────
        # 詳細區塊放在 AI 點評之後、個股清單之前(獨立區塊)
        sentiment_section = ""
        try:
            sentiment = compute_sentiment(CACHE_DIR)
            sentiment_text = format_sentiment_for_tg(sentiment)
            if sentiment_text:
                sentiment_section = sentiment_text + "━━━━━━━━━━━━━━\n"
        except Exception as e:
            print(f"⚠ 大盤情緒指標產生失敗(略過): {e}")

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
                # 系統實務最高分為 8(法人雙買+大戶散戶共振+RS強+營收YoY 同時成立極罕見),
                # 故 8 分視同冠軍級訊號
                score_icon = "🔥" if row['總分'] >= 8 else "•"
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

        # ── 6b. 自選股警示(MA20/MA60 跌破 / KD 死叉 / 外資連賣) ──
        try:
            watchlist = load_watchlist()
            if watchlist:
                # 用 df 的「代號→名稱」當作 name_map(自選股若不在今日 df 裡就只顯示代號)
                name_map = {}
                if df is not None and not df.empty and '名稱' in df.columns:
                    name_map = {str(r['代號']): str(r['名稱']) for _, r in df.iterrows()}
                alerts = check_watchlist(CACHE_DIR, watchlist)
                alerts_text = format_alerts_for_tg(alerts, name_map=name_map)
                if alerts_text:
                    content += alerts_text
        except Exception as e:
            print(f"⚠ 自選股警示產生失敗(略過): {e}")

        # ── 6c. 近 30 日策略績效摘要(在 history 累積足夠後才會有數字) ──
        try:
            # 用「不含今日」的 history 算,避免今日資料尚未有後續報酬干擾
            perf = compute_performance(history, CACHE_DIR, n_days_list=(5,))
            perf_text = format_performance_summary(perf)
            if perf_text:
                content += f"\n{perf_text}\n"
        except Exception as e:
            print(f"⚠ 績效摘要產生失敗(略過): {e}")

        # ── 6d. 系統失效警報(僅 🟡警戒/🔴失效 才推,避免每日洗版;🟢/累積中不推)──
        try:
            sysh = check_system_health(history, CACHE_DIR, hold_days=5, recent_window=20)
            if sysh.get("status") in ("warn", "fail"):
                content += f"\n🚨 <b>{sysh.get('label','')}</b>:{sysh.get('reason','')}\n"
        except Exception as e:
            print(f"⚠ 系統失效監控產生失敗(略過): {e}")

        content += (
            f"\n🌐 <b>"
            f"<a href='https://catchstocktw.streamlit.app/'>開啟我的全自動選股儀表板</a>"
            f"</b>"
        )

        # ── 7. 合體並發送(自動分段) ──────────────────────
        send_ok = send_telegram_message(header + ai_comment + sentiment_section + content)
        n_hit = len(df) if df is not None else 0
        if send_ok:
            print(f"✅ 推播完成!今日共 {n_hit} 檔達標。")
        else:
            print(f"⚠️ 推播失敗或部分失敗(已寫入歷史),今日 {n_hit} 檔達標。")

        # ── 8. 更新歷史(v2 schema:寫入完整 picks 含 score/close/industry) ─
        picks_today = build_picks_from_df(df)
        # 同一天重跑會覆蓋(取最後一次),legacy 也順便清掉
        history = [
            h for h in history
            if h.get("date") not in (today_str, "legacy")
        ]
        history.append({"date": today_str, "picks": picks_today})
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
