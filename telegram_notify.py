import os
import json
import requests
import pandas as pd
import tempfile
import html
from pathlib import Path
from screening0515 import run_screening, PASS_SCORE, HIGH_BREAK_DAYS, CACHE_DIR
from picks_history import load_history, save_history, compute_streak, build_picks_from_df

# 共用模組
from ai_helper import call_openrouter_ai
from cache_status import cache_freshness
from picks_history import load_history, save_history, compute_streak

# 🚀 升級：引入自選股檢查模組
from watchlist_alerts import check_watchlist, format_alerts_for_tg

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TG_MAX_LEN = 4000
TOP_N_DISPLAY = 15

def _load_watchlist() -> list:
    """🚀 升級：讀取使用者自選股清單"""
    wl_path = Path("cache/watchlist.json")
    if wl_path.exists():
        with open(wl_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def _split_for_telegram(text: str, max_len: int) -> list:
    if len(text) <= max_len: return [text]
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(line) > max_len:
            if current: chunks.append(current.rstrip()); current = ""
            for i in range(0, len(line), max_len): chunks.append(line[i:i + max_len])
            continue
        if len(current) + len(line) + 1 > max_len:
            if current.strip(): chunks.append(current.rstrip())
            current = line + "\n"
        else: current += line + "\n"
    if current.strip(): chunks.append(current.rstrip())
    return chunks

def send_telegram_message(text: str) -> bool:
    if not TOKEN or not CHAT_ID: return False
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    chunks = _split_for_telegram(text, TG_MAX_LEN)
    all_ok = True
    for i, chunk in enumerate(chunks, 1):
        payload = {"chat_id": CHAT_ID, "text": chunk, "parse_mode": "HTML", "disable_web_page_preview": True}
        try: requests.post(url, json=payload, timeout=15).raise_for_status()
        except Exception as e: print(f"[Telegram 第 {i} 段發送失敗] {e}"); all_ok = False
    return all_ok

def load_change_pct_map() -> dict:
    try:
        files = sorted(CACHE_DIR.glob('daily_*.parquet'))
        if not files: return {}
        df = pd.read_parquet(files[-1], columns=['stock_id', 'date', 'close'])
        df['date'] = pd.to_datetime(df['date'])
        dates = sorted(df['date'].unique())
        if len(dates) < 2: return {}
        latest_close = df[df['date'] == dates[-1]].drop_duplicates(subset='stock_id', keep='last').set_index('stock_id')['close']
        prev_close = df[df['date'] == dates[-2]].drop_duplicates(subset='stock_id', keep='last').set_index('stock_id')['close']
        common = latest_close.index.intersection(prev_close.index)
        pct = (latest_close.loc[common] - prev_close.loc[common]) / prev_close.loc[common] * 100
        return {str(k): float(v) for k, v in pct.items() if pd.notna(v) and abs(v) < 100}
    except Exception: return {}

def fmt_change_pct(pct) -> str:
    if pct is None or pd.isna(pct): return ""
    if abs(pct) < 0.05: return "⚪0.0%"
    return f"{'🔴' if pct > 0 else '🟢'}{pct:+.1f}%"

def main():
    try:
        history = load_history()

        # 🚀 升級：0.5 資料健康度檢查 (Self-Check)
        health_warnings = []
        try:
            d_files = sorted(CACHE_DIR.glob('daily_*.parquet'))
            if d_files:
                df_chk = pd.read_parquet(d_files[-1], columns=['stock_id', 'close'])
                if (df_chk['close'] == 0).any(): health_warnings.append("發現個股收盤價為 0 的異常 bug")
                if len(df_chk['stock_id'].unique()) < 1500: health_warnings.append("今日股票檔數異常減少 (<1500檔)")
            
            i_files = sorted(CACHE_DIR.glob('institutional_*.parquet'))
            if i_files:
                df_inst = pd.read_parquet(i_files[-1])
                if df_inst['buy'].isna().all(): health_warnings.append("法人數據完全空缺")
        except Exception as e:
            health_warnings.append(f"健康度自檢失敗: {e}")

        with tempfile.TemporaryDirectory() as tmpdir:
            df, _, meta = run_screening(pass_score=PASS_SCORE, output_dir=Path(tmpdir))

        now_tpe = pd.Timestamp.now(tz="Asia/Taipei")
        today_str = now_tpe.strftime("%Y-%m-%d")
        date_str = now_tpe.strftime("%m/%d")

        history = [h for h in history if h.get("date") != today_str]
        yesterday_sids = set(history[-1]["sids"]) if history else set()

        market_data_ok = meta.get('market_data_ok', False)
        twii_now       = meta.get('twii_now') if pd.notna(meta.get('twii_now')) else None
        twii_pct       = float(meta.get('twii_pct', 0.0))
        twii_bias      = float(meta.get('twii_bias', 0.0))
        market_status  = meta.get('market_status', '未知')

        icon = "🔴" if twii_pct > 0 else ("🟢" if twii_pct < 0 else "⚪")
        header = (
            f"🚀 <b>台股戰情摘要 ({date_str})</b>\n"
            f"━━━━━━━━━━━━━━\n"
            f"📈 指數:{f'{twii_now:,.0f}' if twii_now else 'N/A'} ({icon}{twii_pct:+.2f}%)\n"
            f"📐 乖離:{'📈' if twii_bias >= 0 else '📉'}{twii_bias:+.2f}% (位階)\n"
            f"🌡️ 盤勢:<b>{market_status}</b>\n"
        )
        if meta.get('score_note'): header += f"❗ <b>{meta.get('score_note')}</b>\n"

        # 加入健康度警告紅字
        if health_warnings:
            for w in health_warnings: header += f"🚨 <b>資料異常警告: {w}</b>\n"

        freshness = cache_freshness(meta.get('cache_max_date'))
        if freshness["level"] in ("warn", "error", "missing"): header += f"⚠️ <b>{freshness['msg']}</b>\n"
        header += "━━━━━━━━━━━━━━\n\n"

        ai_comment = ""
        if df is not None and not df.empty:
            top_stock = df.iloc[0]
            prompt = f"今日大盤狀態:{market_status} ({twii_pct:+.2f}%)。冠軍標的:{top_stock['代號']} {top_stock['名稱']}。請用繁體中文寫一段 60~80 字盤後客觀點評，無Markdown。"
            m_name, ai_text = call_openrouter_ai(prompt, max_tokens=250)
            if ai_text: ai_comment = f"🧠 <b>AI 分析師({m_name}):</b>\n<i>「{html.escape(ai_text)}」</i>\n━━━━━━━━━━━━━━\n"

        change_pct_map = load_change_pct_map()
        if df is None or len(df) == 0: content = "💡 目前盤勢較嚴峻,沒有股票達標。"
        else:
            content = f"🔥 <b>今日達標個股 (共 {len(df)} 檔)</b>\n"
            today_set = set(df['代號'].astype(str).tolist())
            for _, row in df.head(TOP_N_DISPLAY).iterrows():
                sid_str = str(row['代號'])
                change_str = fmt_change_pct(change_pct_map.get(sid_str))
                tags = ""
                if history:
                    streak = compute_streak(sid_str, history)
                    if streak == 1: tags += " <code>[新進]</code>"
                    elif streak >= 3: tags += f" <code>[連{streak}日]</code>"
                if row.get(f"·{HIGH_BREAK_DAYS}日量價齊揚突破") == 1: tags += " <code>[突破]</code>"
                
                content += f"{'🔥' if row['總分'] >= 9 else '•'} <a href='https://tw.stock.yahoo.com/quote/{sid_str}'>{sid_str}</a> {html.escape(str(row['名稱']))} ({row['總分']}分) {change_str}{tags}\n"

            valid_ind = df['產業'].dropna()
            if not valid_ind.empty:
                top_ind = valid_ind.value_counts().idxmax()
                content += f"\n📊 <b>今日主流群聚:</b> {top_ind} ({int(valid_ind.value_counts().max())}檔)\n"

        if history and yesterday_sids:
            exited = yesterday_sids - (today_set if df is not None else set())
            if exited: content += f"\n📤 <b>今日退場:</b> {len(exited)} 檔 ({', '.join(sorted(exited)[:8])})\n"

        content += "\n🌐 <b><a href='https://catchstocktw.streamlit.app/'>開啟我的全自動選股儀表板</a></b>\n"

        # 🚀 升級：自選股主動告警
        watchlist = _load_watchlist()
        watchlist_msg = ""
        if watchlist:
            alerts = check_watchlist(CACHE_DIR, watchlist)
            name_map = {}
            info_files = sorted(CACHE_DIR.glob('info_*.parquet'))
            if info_files:
                df_info = pd.read_parquet(info_files[-1])
                name_map = df_info.set_index('stock_id')['stock_name'].astype(str).to_dict()
            watchlist_msg = format_alerts_for_tg(alerts, name_map)

        send_telegram_message(header + ai_comment + content + watchlist_msg)

        # ✅ 替換成以下這段新邏輯：
        cur_picks = build_picks_from_df(df) if (df is not None and not df.empty) else []       
        
        # 移除舊的今天紀錄與 legacy
        history = [h for h in history if h.get("date") not in (today_str, "legacy")]
        
        # 以 "picks" 的新格式存入，保留當日 close / score / industry
        history.append({"date": today_str, "picks": cur_picks})
        save_history(history)

    except Exception as e:
        send_telegram_message(f"❌ <b>選股機器人罷工求救!</b>\n\n<code>{html.escape(str(e))}</code>")

if __name__ == "__main__":
    main()