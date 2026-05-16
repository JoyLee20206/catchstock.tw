import os
import json
import requests
import pandas as pd
import tempfile
from pathlib import Path
from screening0515 import run_screening, PASS_SCORE, HIGH_BREAK_DAYS

# 從環境變數讀取安全金鑰
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
# 設定記憶檔案的路徑 (存放在 cache 資料夾，GitHub Actions 會自動備份)
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

        # 1. 執行選股 ── 改前後只差這一段 ──────────────────────────
        # 改前（會在 repo 根目錄產生廢檔）:
        # df, _, meta = run_screening(pass_score=PASS_SCORE)

        # 改後（輸出到暫存目錄，用完自動清除）:
        with tempfile.TemporaryDirectory() as tmpdir:
            df, _, meta = run_screening(pass_score=PASS_SCORE, output_dir=Path(tmpdir))
        # ────────────────────────────────────────────────────────────

        # 2. 準備台北時區當前日期
        date_str = pd.Timestamp.now(tz="Asia/Taipei").strftime("%m/%d")
        
        
        
        # 3. 安全提取 meta 變數
        twii_now      = meta.get('twii_now', 0) if meta.get('twii_now') is not None else 0
        twii_pct      = meta.get('twii_pct', 0.0) if meta.get('twii_pct') is not None else 0.0
        twii_bias     = meta.get('twii_bias', 0.0) if meta.get('twii_bias') is not None else 0.0
        score_note    = meta.get('score_note', "")
        market_status = meta.get('market_status', '未知')
        
        # 4. 根據數據動態決定漲跌與位階圖示
        icon = "🔴" if twii_pct < 0 else "🟢"
        bias_icon = "📈" if twii_bias >= 0 else "📉"
        
        # 5. 組裝【大盤戰情摘要儀表板】
        header = (
            f"🚀 <b>台股戰情摘要 ({date_str})</b>\n"
            f"━━━━━━━━━━━━━━\n"
            f"📈 指數：{twii_now:,.0f} ({icon}{twii_pct:+.2f}%)\n"
            f"📐 乖離：{bias_icon}{twii_bias:+.2f}% (位階)\n"
            f"🌡️ 盤勢：<b>{market_status}</b>\n"
        )

        if score_note:
            header += f"❗ <b>{score_note}</b>\n"
        
        header += "━━━━━━━━━━━━━━\n\n"

        # 6. 組裝個股清單與自動標籤
        if df is None or len(df) == 0:
            content = "💡 目前盤勢較嚴峻，沒有股票達標。"
        else:
            content = f"🔥 <b>今日達標個股 (共 {len(df)} 檔)：</b>\n"

        # 問題 2 的修改位置：把硬寫的欄位名換成動態的
            breakout_col = f"·{HIGH_BREAK_DAYS}日量價齊揚突破"   # ← 新增這行            
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
                
        # 💡 修正 1：完全退到 if/else 外面，確保每天都顯示網址！
        content += f"\n\n🌐 <b><a href='https://catchstocktw.streamlit.app/'>開啟我的全自動選股儀表板</a></b>"
                
        # 7. 合體並發送最終戰報
        send_telegram_message(header + content)
        print(f"推播成功！今日共 {len(df) if df is not None else 0} 檔達標。")

        # 8. 儲存今天的名單，供明天比對使用
        current_sids = df['代號'].astype(str).tolist() if (df is not None and not df.empty) else []
        os.makedirs("cache", exist_ok=True)
        with open(PREV_PICKS_FILE, "w", encoding="utf-8") as f:
            json.dump(current_sids, f, ensure_ascii=False)

    except Exception as e:
        import html # 引入 html 模組處理跳脫字元
        # 💡 修正 2：使用 html.escape 把危險符號安全轉換，避免推播被 Telegram 擋掉
        safe_error = html.escape(str(e))
        error_msg = f"❌ <b>選股機器人罷工求救！</b>\n\n系統發生致命錯誤，請至 GitHub 檢查：\n<code>{safe_error}</code>"
        send_telegram_message(error_msg)
        print(f"❌ 選股推播發生致命錯誤：{e}")