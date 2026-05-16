import os
import json
import requests
import pandas as pd
from screening0515 import run_screening, PASS_SCORE

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
        # 0. 讀取昨天的過關名單 (賦予程式記憶力)
        previous_sids = set()
        if os.path.exists(PREV_PICKS_FILE):
            try:
                with open(PREV_PICKS_FILE, "r", encoding="utf-8") as f:
                    previous_sids = set(json.load(f))
            except Exception as e:
                print(f"讀取歷史名單失敗，忽略比對：{e}")

        # 1. 執行選股，同時取得大盤 meta 數據
        df, _, meta = run_screening(pass_score=PASS_SCORE)

        # 2. 準備台北時區當前日期 (顯示於標題)
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
            
            for _, row in df.head(15).iterrows():
                sid_str = str(row['代號'])
                score_icon = "🔥" if row['總分'] >= 9 else "•"
                
                # 🏷️ 自動組裝亮點標籤
                tags = ""
                # 判斷是否為新進榜 (前提是 previous_sids 不是空的，避免第一次執行全部都標新進)
                if sid_str not in previous_sids and len(previous_sids) > 0:
                    tags += " <code>[新進]</code>"
                
                # 動能突破標籤
                if row.get('·60日量價齊揚突破') == 1:
                    tags += " <code>[突破]</code>"
                
                # 把單純的 <code> 換成帶有 Yahoo 股市連結的 <a> 標籤
                content += f"{score_icon} <a href='https://tw.stock.yahoo.com/quote/{sid_str}'>{sid_str}</a> {row['名稱']} ({row['總分']}分){tags}\n"
            
            # 📊 自動統計並顯示今日主流產業
            if '產業' in df.columns and len(df) > 0:
                valid_industries = df['產業'].dropna()
                valid_industries = valid_industries[valid_industries != ""] 
                if not valid_industries.empty:
                    top_industry = valid_industries.value_counts().idxmax()
                    top_count = valid_industries.value_counts().max()
                    content += f"\n📊 <b>今日主流群聚：</b> {top_industry} ({top_count}檔)\n"
                
            if len(df) > 15:
                content += f"\n<i>...等其餘 {len(df)-15} 檔請至網頁查看完整分析</i>"
                content += f"\n🌐 <b><a href='YOUR_STREAMLIT_URL'>開啟我的全自動選股儀表板</a></b>"
                
        # 7. 合體並發送最終戰報
        send_telegram_message(header + content)
        print(f"推播成功！今日共 {len(df) if df is not None else 0} 檔達標。")

        # 8. 儲存今天的名單，供明天比對使用
        current_sids = df['代號'].astype(str).tolist() if (df is not None and not df.empty) else []
        os.makedirs("cache", exist_ok=True)
        with open(PREV_PICKS_FILE, "w", encoding="utf-8") as f:
            json.dump(current_sids, f, ensure_ascii=False)

    except Exception as e:
        print(f"❌ 選股推播發生致命錯誤：{e}")