import os
import requests
import pandas as pd
from screening0515 import run_screening, PASS_SCORE

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram_message(text):
    if not TOKEN or not CHAT_ID:
        print("未設定 Telegram Token 或 Chat ID"); return
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()   # HTTP 4xx/5xx 自動拋出
    except Exception as e:
        print(f"[Telegram 發送失敗] {e}")   # 只 print，不再遞迴呼叫

if __name__ == "__main__":
    try:
        # 1. 執行選股，同時取得大盤 meta 數據
        df, _, meta = run_screening(pass_score=PASS_SCORE)

        # 2. 準備日期
        date_str = pd.Timestamp.now(tz="Asia/Taipei").strftime("%m/%d")
        
        # 3. 組裝【大盤戰情摘要】
        # 假設 meta 包含：twii_close(收盤), twii_pct(漲跌幅), market_status(強弱勢)
        market_status = meta.get('market_status', '未知')
        twii_close = meta.get('twii_close', 0)
        twii_pct = meta.get('twii_pct', 0)
        
        # 根據漲跌決定表情符號
        icon = "🔴" if twii_pct < 0 else "🟢"
        
        header = (
            f"🚀 <b>台股戰情摘要 ({date_str})</b>\n"
            f"━━━━━━━━━━━━━━\n"
            f"📈 指數：{twii_close:,.0f} ({icon}{twii_pct:+.2f}%)\n"
            f"🌡️ 盤勢：<b>{market_status}</b>\n"
            f"🎯 門檻：{PASS_SCORE} 分過關\n"
            f"━━━━━━━━━━━━━━\n\n"
        )

        # 4. 組裝個股清單
        if df is None or len(df) == 0:
            content = "💡 目前盤勢較嚴峻，沒有股票達標。"
        else:
            content = f"🔥 <b>今日達標個股 (共 {len(df)} 檔)：</b>\n"
            for _, row in df.head(15).iterrows():
                # 加上小標籤區分：例如總分特別高的加個火苗
                score_icon = "🔥" if row['總分'] >= 9 else "•"
                content += f"{score_icon} <code>{row['代號']}</code> {row['名稱']} ({row['總分']}分)\n"
            
            if len(df) > 15:
                content += f"\n<i>...等其餘 {len(df)-15} 檔請至網頁查看</i>"

        # 發送最終訊息
        send_telegram_message(header + content)
        print(f"推播成功！({date_str})")

    except Exception as e:
        print(f"❌ 推播發生錯誤：{e}")