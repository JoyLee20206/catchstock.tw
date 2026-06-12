# -*- coding: utf-8 -*-
"""Telegram 發送共用模組 — telegram_notify.py(選股推播)與 bottom_push.py(止跌判讀)共用。

設計約束:這個檔案只准依賴 os / requests,**絕不 import 選股系統的任何模組**。
理由:bottom_push 是輕量排程,若透過 telegram_notify 取得發送函式,會連帶載入
screening0515 / performance / market_sentiment 整套堆疊——任何一個壞掉,
止跌推播就跟著掛。發送邏輯抽到這裡,兩邊的排程才能真正互不牽連。

兩種使用情境(差異用參數表達,不要再各自複製一份):
- 選股推播:HTML 格式、超長自動分段、沒 token 視為設定錯誤(回 False)
- 止跌推播:純文字、訊息短不分段也夠、沒 token 轉 dry-run 印出(方便本機測試)
"""
import os

import requests

TG_MAX_LEN = 4000   # Telegram 單訊息上限 4096,留 96 字餘裕


def get_credentials():
    """執行時才讀環境變數(而非模組載入時),讓測試/排程設定變更立即生效。"""
    return os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")


def split_for_telegram(text: str, max_len: int = TG_MAX_LEN) -> list:
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


def send_telegram_message(
    text: str,
    parse_mode: str = "HTML",
    dry_run_when_no_token: bool = False,
    timeout: int = 15,
) -> bool:
    """發送 Telegram 訊息,超過 TG_MAX_LEN 自動分段、逐段獨立發送(一段失敗不拖累其他段)。

    parse_mode:"HTML" 或 None(純文字;含 <、& 等字元的非 HTML 訊息務必用 None,
                否則 Telegram 會回 400)。
    dry_run_when_no_token:True 時沒設 token 改印到 stdout 並回 True(本機測試友善);
                False 時視為設定錯誤,印警告回 False。
    """
    token, chat_id = get_credentials()
    if not token or not chat_id:
        if dry_run_when_no_token:
            print("ℹ️ 未設定 TELEGRAM_BOT_TOKEN/CHAT_ID,dry-run 模式:")
            print("-" * 40)
            print(text)
            print("-" * 40)
            return True
        print("未設定 Telegram Token 或 Chat ID")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = split_for_telegram(text, TG_MAX_LEN)
    all_ok = True

    for i, chunk in enumerate(chunks, 1):
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "disable_web_page_preview": True,  # 避免每個 Yahoo 連結都展開縮圖
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
        except Exception as e:
            print(f"[Telegram 第 {i}/{len(chunks)} 段發送失敗] {e}")
            all_ok = False
    return all_ok
