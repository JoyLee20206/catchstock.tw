import os
import requests
from screening0515 import run_screening, PASS_SCORE

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_message(text):
    if not TOKEN or not CHAT_ID:
        print("未設定 Telegram Token 或 Chat ID")
        return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

if __name__ == "__main__":
    try:
        # 1. 執行選股 (使用預設過關門檻)
        df, _, meta = run_screening(pass_score=PASS_SCORE)

        # 2. 組裝訊息文字
        if df is None or len(df) == 0:
            msg = "📊 *今日台股選股報告*\n\n沒有股票達到過關門檻。"
        else:
            msg = f"📊 *今日台股選股報告*\n\n🔥 共 {len(df)} 檔達標：\n\n"
            # 依總分排序，只取前 10 檔避免訊息過長
            for idx, row in df.head(10).iterrows():
                msg += f"• `{row['代號']}` {row['名稱']} ({row['總分']}分)\n"
            
            if len(df) > 10:
                msg += f"\n*(僅顯示前 10 檔，請至網頁版查看完整圖表)*"

        # 3. 發送訊息
        send_telegram_message(msg)
        print("推播成功！")

    except Exception as e:
        send_telegram_message(f"❌ 選股推播發生錯誤：\n{e}")
        print(f"Error: {e}")
