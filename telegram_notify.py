import os
import json
import requests
import pandas as pd
import tempfile
import html
from pathlib import Path
from screening0515 import run_screening, PASS_SCORE, HIGH_BREAK_DAYS

# 從環境變數讀取安全金鑰
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") # 🎯 改用 OpenRouter 金鑰

# 設定記憶檔案的路徑
PREV_PICKS_FILE = "cache/previous_picks.json"

def send_telegram_message(text):
    """透過 Telegram Bot API 發送 HTML 格式訊息"""
    if not TOKEN or not CHAT_ID:
        print("未設定 Telegram Token 或 Chat ID")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print(f"[Telegram 發送失敗] {e}")

if __name__ == "__main__":
    try:
        # 0. 讀取昨天的過關名單
        previous_sids = set()
        if os.path.exists(PREV_PICKS_FILE):
            try:
                with open(PREV_PICKS_FILE, "r", encoding="utf-8") as f:
                    previous_sids = set(json.load(f))
            except Exception as e:
                print(f"讀取歷史名單失敗，忽略比對：{e}")

        # 1. 執行選股
        with tempfile.TemporaryDirectory() as tmpdir:
            df, _, meta = run_screening(pass_score=PASS_SCORE, output_dir=Path(tmpdir))

        # 2. 準備台北時區當前日期
        date_str = pd.Timestamp.now(tz="Asia/Taipei").strftime("%m/%d")
        
        # 3. 安全提取 meta 變數
        # Bug A 修正：用 pd.notna 同時擋 None 與 NaN（NaN 在 Python 是 truthy，`or 0.0` 抓不到）
        def _safe_num(v, default=0.0):
            return float(v) if pd.notna(v) else default

        market_data_ok = meta.get('market_data_ok', False)
        twii_now_raw  = meta.get('twii_now')
        twii_pct      = _safe_num(meta.get('twii_pct'))
        twii_bias     = _safe_num(meta.get('twii_bias'))
        score_note    = meta.get('score_note', "")
        market_status = meta.get('market_status', '未知')
        twii_now      = twii_now_raw if pd.notna(twii_now_raw) else None

        # 4. 根據數據動態決定漲跌與位階圖示
        # 市場資料失效時使用灰色圖示，避免 0 值誤導為真實報價
        if not market_data_ok or twii_now is None:
            icon      = "⚪"
            bias_icon = "⚪"
            twii_now_str  = "N/A"
            twii_pct_str  = "N/A"
            twii_bias_str = "N/A"
        else:
            icon          = "🔴" if twii_pct < 0 else "🟢"
            bias_icon     = "📈" if twii_bias >= 0 else "📉"
            twii_now_str  = f"{twii_now:,.0f}"
            twii_pct_str  = f"{twii_pct:+.2f}%"
            twii_bias_str = f"{twii_bias:+.2f}%"

        # 5. 組裝【大盤戰情摘要儀表板】
        header = (
            f"🚀 <b>台股戰情摘要 ({date_str})</b>\n"
            f"━━━━━━━━━━━━━━\n"
            f"📈 指數：{twii_now_str} ({icon}{twii_pct_str})\n"
            f"📐 乖離：{bias_icon}{twii_bias_str} (位階)\n"
            f"🌡️ 盤勢：<b>{market_status}</b>\n"
        )
        if score_note:
            header += f"❗ <b>{score_note}</b>\n"
        header += "━━━━━━━━━━━━━━\n\n"

        # 🧠 6. 呼叫 OpenRouter 免費 AI 產生盤後點評 (安全防禦+不塞車模型版)
        ai_comment = ""
        if df is not None and not df.empty and OPENROUTER_API_KEY:
            try:
                top_stock = df.iloc[0]
                sid_top = str(top_stock['代號'])
                name_top = top_stock['名稱']
                score_top = top_stock['總分']
                
                breakout_col = f"·{HIGH_BREAK_DAYS}日量價齊揚突破"
                is_breakout = top_stock.get(breakout_col) == 1
                is_sync = top_stock.get("★籌碼共振(大戶↑散戶↓)") == 1
                
                prompt = f"""
                你是一位台灣股市的資深量化分析師。本日系統選股冠軍是 {sid_top} {name_top} (總分 {score_top}/10分)。
                該股亮點包含：{"帶量突破," if is_breakout else ""}{"大戶增散戶減的籌碼共振," if is_sync else ""}動能強勁。
                請用繁體中文寫一段約 60~80 字的盤後精闢點評，語氣要專業、客觀。
                注意：請直接輸出純文字，絕對不要使用 Markdown 語法 (不要有星號或井號)。
                """
                
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "openai/gpt-oss-20b:free",  # 🎯 OpenAI 官方最新下放的免費版
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3 
                }
                
                resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=15)
                resp.raise_for_status()   # ← 加這一行
                resp_json = resp.json()
                
                # 安全解包
                if 'choices' in resp_json:
                    ai_text = resp_json['choices'][0]['message']['content'].strip()
                    ai_comment = (
                        f"🧠 <b>AI 虛擬分析師評語：</b>\n"
                        f"<i>「{html.escape(ai_text)}」</i>\n"
                        f"━━━━━━━━━━━━━━\n"
                    )
                elif 'error' in resp_json:
                    print(f"❌ OpenRouter 拒絕請求原因: {resp_json['error'].get('message')}")
                else:
                    print(f"❌ 異常回應結構: {resp_json}")
                    
            except Exception as e:
                print(f"AI 生成失敗: {e}")

        # 7. 組裝個股清單與自動標籤
        if df is None or len(df) == 0:
            content = "💡 目前盤勢較嚴峻，沒有股票達標。"
        else:
            content = f"🔥 <b>今日達標個股 (共 {len(df)} 檔)：</b>\n"
            breakout_col = f"·{HIGH_BREAK_DAYS}日量價齊揚突破"
            for _, row in df.head(15).iterrows():
                sid_str = str(row['代號'])
                score_icon = "🔥" if row['總分'] >= 9 else "•"
                
                tags = ""
                if sid_str not in previous_sids and len(previous_sids) > 0:
                    tags += " <code>[新進]</code>"
                if row.get(breakout_col) == 1:
                    tags += " <code>[突破]</code>"
                
                content += f"{score_icon} <a href='https://tw.stock.yahoo.com/quote/{sid_str}'>{sid_str}</a> {row['名稱']} ({row['總分']}分){tags}\n"
            
            if '產業' in df.columns and len(df) > 0:
                valid_industries = df['產業'].dropna()
                valid_industries = valid_industries[valid_industries != ""] 
                if not valid_industries.empty:
                    top_industry = valid_industries.value_counts().idxmax()
                    top_count = valid_industries.value_counts().max()
                    content += f"\n📊 <b>今日主流群聚：</b> {top_industry} ({top_count}檔)\n"
                
            if len(df) > 15:
                content += f"\n<i>...等其餘 {len(df)-15} 檔請至網頁查看完整分析</i>"
                
        content += f"\n\n🌐 <b><a href='https://catchstocktw.streamlit.app/'>開啟我的全自動選股儀表板</a></b>"
                
        # 8. 合體並發送最終戰報
        send_telegram_message(header + ai_comment + content)
        print(f"推播成功！今日共 {len(df) if df is not None else 0} 檔達標。")

        # 9. 儲存今天的名單
        current_sids = df['代號'].astype(str).tolist() if (df is not None and not df.empty) else []
        os.makedirs("cache", exist_ok=True)
        with open(PREV_PICKS_FILE, "w", encoding="utf-8") as f:
            json.dump(current_sids, f, ensure_ascii=False)

    except Exception as e:
        safe_error = html.escape(str(e))
        error_msg = f"❌ <b>選股機器人罷工求救！</b>\n\n系統發生致命錯誤，請至 GitHub 檢查：\n<code>{safe_error}</code>"
        send_telegram_message(error_msg)
        print(f"❌ 選股推播發生致命錯誤：{e}")