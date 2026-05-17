import os
import json
import requests
import pandas as pd
import tempfile
import html
from pathlib import Path
from screening0515 import run_screening, PASS_SCORE, HIGH_BREAK_DAYS

# ── 環境變數 ───────────────────────────────────────────────────────────
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# 可用環境變數指定優先 AI 模型(填關鍵字即可,例如 "deepseek" / "qwen" / "gemini")
# 不設定就照 AI_MODELS 預設順序跑
PREFERRED_AI = os.environ.get("PREFERRED_AI_MODEL", "").strip().lower()

# ── 路徑與設定 ─────────────────────────────────────────────────────────
HISTORY_FILE = "cache/previous_picks.json"   # 多日歷史(向下相容舊單日 flat list)
HISTORY_DAYS = 7                              # 保留最近 N 天,夠判斷「連 5 日入選」
TG_MAX_LEN = 4000                             # Telegram 單訊息上限為 4096,留 96 字餘裕
STALE_WARN_DAYS = 4                           # 快取超過 N 天才示警(週末自然延遲不算)
TOP_N_DISPLAY = 15                            # Telegram 訊息列出前 N 檔

# ── AI 模型輪替清單(依優先序排列,前面失敗就試下一個) ──────────────
# 注意:免費模型可用性會變動,部署前建議到 https://openrouter.ai/models?max_price=0 確認
AI_MODELS = [
    {"id": "deepseek/deepseek-chat-v3-0324:free",     "name": "DeepSeek V3"},
    {"id": "qwen/qwen-2.5-72b-instruct:free",         "name": "Qwen 2.5"},
    {"id": "google/gemini-2.0-flash-exp:free",        "name": "Gemini 2.0"},
    {"id": "meta-llama/llama-3.3-70b-instruct:free",  "name": "Llama 3.3"},
    {"id": "openai/gpt-oss-20b:free",                 "name": "GPT-OSS 20B"},
]


# ══════════════════════════════════════════════════════════════════════
# Telegram 送訊息(自動分段)
# ══════════════════════════════════════════════════════════════════════
def _split_for_telegram(text: str, max_len: int) -> list:
    """在換行邊界切分長訊息;單段絕不超過 max_len。
    所有 HTML 標籤都在單一行內成對出現,只在「行與行之間」切就不會弄壞 <b>/<i>/<a>。
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
            if current.strip():           # 防呆:極端情況下 current 可能是空字串,避免 append 空 chunk(Telegram 會 400)
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
# AI 呼叫(多模型輪替 + fallback)
# ══════════════════════════════════════════════════════════════════════
def call_openrouter_ai(prompt: str, timeout: int = 20):
    """依序嘗試 AI_MODELS,回傳 (model_name, ai_text);全部失敗回傳 (None, None)。
    若環境變數 PREFERRED_AI_MODEL 有設關鍵字,符合的模型會被排到最前面。
    """
    if not OPENROUTER_API_KEY:
        return None, None

    # 優先模型置頂(穩定排序,符合的拉到前面、其他維持原順序)
    models = sorted(
        AI_MODELS,
        key=lambda m: 0 if PREFERRED_AI and PREFERRED_AI in m["id"].lower() else 1
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    for m in models:
        try:
            payload = {
                "model": m["id"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 250,   # 限長,避免免費模型話癆把 TG 訊息撐爆
            }
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=timeout
            )
            resp.raise_for_status()
            j = resp.json()

            if "choices" in j and j["choices"]:
                text = j["choices"][0]["message"]["content"].strip()
                # 強制清掉模型偷渡的 Markdown(免費模型常常不聽 prompt 指令)
                # Bug 1 修正：長 token 優先,否則 `###` 會被 `##` 先吃掉只剩 `#` 殘渣
                for tok in ("###", "##", "**", "*", "`"):
                    text = text.replace(tok, "")
                text = text.strip()
                if text:
                    print(f"   ✅ AI 模型 {m['name']} 回應成功")
                    return m["name"], text
            elif "error" in j:
                print(f"   ⚠ {m['name']} 拒絕: {j['error'].get('message', '')[:120]}")
        except requests.exceptions.Timeout:
            print(f"   ⚠ {m['name']} 逾時,換下一個")
        except Exception as e:
            print(f"   ⚠ {m['name']} 失敗: {str(e)[:120]}")

    print("   ❌ 所有 AI 模型皆失敗")
    return None, None


# ══════════════════════════════════════════════════════════════════════
# 多日歷史([新進] / [連 N 日] / 退場 偵測)
# ══════════════════════════════════════════════════════════════════════
def load_history() -> list:
    """讀取多日歷史。
    新格式: [{"date": "YYYY-MM-DD", "sids": ["2330", ...]}, ...]  依日期遞增排序
    舊格式: ["2330", ...]                                       自動包成單筆 "legacy"
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data and isinstance(data[0], str):
            # 舊版 flat list → 包成單筆 legacy,後續會被今日結果取代
            return [{"date": "legacy", "sids": data}]
        if isinstance(data, list):
            return data
    except Exception as e:
        print(f"讀取歷史名單失敗,忽略比對: {e}")
    return []


def save_history(history: list) -> None:
    """寫回歷史,只保留最近 HISTORY_DAYS 天。"""
    os.makedirs("cache", exist_ok=True)
    trimmed = history[-HISTORY_DAYS:]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, ensure_ascii=False, indent=2)


def _safe_num(v, default: float = 0.0) -> float:
    """擋 None 與 NaN(NaN 是 truthy,`or default` 抓不到)。"""
    return float(v) if pd.notna(v) else default


def compute_streak(sid: str, history: list) -> int:
    """計算 sid 從今日往前推的連續入選天數(假設今日清單已含此 sid,故初值 = 1)。
    呼叫時機:history 尚未加入今日 → 從 history 尾巴往前掃,有就 +1,沒有就 break。
    """
    streak = 1  # 今天本來就含此 sid
    for entry in reversed(history):
        if sid in entry.get("sids", []):
            streak += 1
        else:
            break
    return streak


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

        # 同日重跑保護:把 history 中「今日」的項目剔除,避免污染 yesterday_sids、
        # 退場偵測與 compute_streak(否則早上跑過、下午重跑會把今早結果當「昨天」)
        history = [h for h in history if h.get("date") != today_str]
        yesterday_sids = set(history[-1]["sids"]) if history else set()

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
            icon          = "🔴" if twii_pct < 0 else "🟢"
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

        # 快取新鮮度:週末自然延遲(1~2 天)不算,真正異常(>= 4 天)才示警
        stale_days = None
        if cache_max_date is not None and not pd.isna(cache_max_date):
            try:
                # Suggestion 4 修正：tz_localize(None) 在 tz-naive Timestamp 上會丟 TypeError,
                # 對稱檢查避免日後 now_tpe 被改成 naive 後突然炸掉
                today_naive = now_tpe.normalize()
                if today_naive.tz is not None:
                    today_naive = today_naive.tz_localize(None)
                cache_dt = pd.Timestamp(cache_max_date).normalize()
                if cache_dt.tz is not None:
                    cache_dt = cache_dt.tz_localize(None)
                stale_days = (today_naive - cache_dt).days
            except Exception:
                stale_days = None

        if stale_days is not None and stale_days >= STALE_WARN_DAYS:
            header += (
                f"⚠️ <b>資料延遲 {stale_days} 天,可能是 fetch_cache 排程失敗,僅供參考</b>\n"
            )

        header += "━━━━━━━━━━━━━━\n\n"

        # ── 5. AI 點評(模型輪替) ──────────────────────────
        ai_comment = ""
        if df is not None and not df.empty:
            top_stock = df.iloc[0]
            sid_top   = str(top_stock['代號'])
            # Risk 2 修正：名稱缺失時用代號當替身,避免 AI prompt 出現「2330 nan」誤導點評
            _raw_name = top_stock['名稱']
            name_top  = str(_raw_name) if pd.notna(_raw_name) else sid_top
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

            model_name, ai_text = call_openrouter_ai(prompt)
            if ai_text:
                ai_comment = (
                    f"🧠 <b>AI 分析師({model_name}):</b>\n"
                    f"<i>「{html.escape(ai_text)}」</i>\n"
                    f"━━━━━━━━━━━━━━\n"
                )

        # ── 6. 個股清單 + tags(新進 / 連 N 日 / 突破 / 共振) ─
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

                tags = ""
                # [新進] / [連 N 日]:第一次跑(history 空)時都不標,避免誤報
                if history:
                    streak = compute_streak(sid_str, history)
                    if streak == 1:
                        tags += " <code>[新進]</code>"
                    elif streak >= 3:
                        tags += f" <code>[連{streak}日]</code>"
                    # streak == 2 不標,避免過度雜訊;3 日才算有持續性

                if row.get(breakout_col) == 1:
                    tags += " <code>[突破]</code>"

                # 防禦 ETF 名稱含 &/< 等字元;名稱缺失時用代號當替身,避免顯示「nan」
                _raw = row['名稱']
                stock_name = html.escape(str(_raw) if pd.notna(_raw) else sid_str)
                content += (
                    f"{score_icon} "
                    f"<a href='https://tw.stock.yahoo.com/quote/{sid_str}'>{sid_str}</a> "
                    f"{stock_name} ({row['總分']}分){tags}\n"
                )

            # 主流群聚 + 集中度警告
            if '產業' in df.columns and len(df) > 0:
                valid_industries = df['產業'].dropna()
                valid_industries = valid_industries[valid_industries != ""]
                if not valid_industries.empty:
                    counts = valid_industries.value_counts()
                    top_industry = counts.index[0]
                    top_count = int(counts.iloc[0])
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

        # 退場通知:昨在、今天沒了(history 空時跳過,避免首跑誤報「全部退場」)
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
        # 同一天重跑會覆蓋(取最後一次),不會疊出兩筆;legacy 也順便清掉
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
