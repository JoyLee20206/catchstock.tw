# -*- coding: utf-8 -*-
"""止跌判讀 排程腳本(GitHub Actions / 本機皆可跑)

流程:抓資料 → 判定 → 寫歷史(cache/bottom_signal_history.json)→ Telegram 推播。
搭配 bottom_push.yaml 一天跑兩次:
  16:10 台灣時間 — 盤後初判(現貨資料齊;期貨未平倉/P/C 可能還是昨天的)
  21:30 台灣時間 — 完整版(期交所盤後資料齊,覆蓋同日歷史後重推)

設計:
- 部分資料失敗照樣推播(缺項顯示 ❓),只有「程式本身爆掉」才 exit 1
- VIXTWN 雙來源都掛 → 訊息帶 🚨 告警,絕不靜默
- 沒設 TELEGRAM_BOT_TOKEN 時轉為 dry-run(只印不發),方便本機測試
"""
import sys
import io
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

from bottom_signal import (
    run_all_checks, persist_bottom_history, persist_bottom_latest,
    format_bottom_for_tg,
)
from market_sentiment import persist_fi_history
from tg_common import send_telegram_message, get_credentials

CACHE_DIR = Path(__file__).parent / "cache"


def _tw_now() -> datetime:
    """台灣時間(GHA 主機是 UTC)。"""
    return datetime.now(timezone(timedelta(hours=8)))


def _pass_label() -> str:
    """16 點多跑的是初判,晚上跑的是完整版(期交所盤後資料已齊)。"""
    return "盤後初判" if _tw_now().hour < 19 else "完整版"


def send_telegram(text: str) -> bool:
    """發 Telegram;沒 token 就 dry-run 印出來。
    止跌訊息是純文字(parse_mode=None):內容可能含 <、& 等字元,套 HTML 模式會被 TG 退 400。
    """
    ok = send_telegram_message(text, parse_mode=None,
                               dry_run_when_no_token=True, timeout=20)
    token, chat_id = get_credentials()
    if token and chat_id:  # dry-run 時上面已印過內容,不再重複報「已發送」
        print("📲 Telegram", "已發送" if ok else "發送失敗(詳見上方逐段錯誤)")
    return ok


def main() -> int:
    label = _pass_label()
    print(f"🛑 止跌判讀排程({label})— {_tw_now():%Y-%m-%d %H:%M} 台灣時間")

    try:
        result = run_all_checks(cache_dir=CACHE_DIR)
    except Exception:
        # 程式本身爆掉:推告警 + 非零退出(GHA 顯示紅燈)
        err = traceback.format_exc()
        print(err)
        send_telegram(f"🚨 止跌判讀程式執行失敗({label}),請檢查 GitHub Actions log\n"
                      f"{err.splitlines()[-1][:200]}")
        return 1

    # 寫入今日歷史(同日重跑覆蓋 → 21:30 完整版會蓋掉 16:10 初判)
    persist_bottom_history(CACHE_DIR, result)

    # 存完整結果 → UI 直接讀檔秒開,不用現抓
    persist_bottom_latest(CACHE_DIR, result)

    # 外資期貨淨額逐日累積(「空單回補」隔天才有比較基準)
    if result.get("fi_net_today") is not None:
        persist_fi_history(CACHE_DIR, result["fi_net_today"])

    send_telegram(format_bottom_for_tg(result, pass_label=label))

    print(f"✅ 完成:{result['level_icon']} {result['level_label']}"
          f"(成立 {result['n_ok']}/{len(result['items'])})")
    return 0


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace")
    sys.exit(main())
