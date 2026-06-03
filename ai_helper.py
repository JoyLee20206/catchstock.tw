"""OpenRouter AI 共用模組(多模型 fallback)

供 telegram_notify.py 與 screening_ui16.py 共用,
避免兩邊各維護一份模型清單與呼叫邏輯。

⚠️ OpenRouter 免費帳號限制 (2026):
- 每日 50 次請求 (帳號內儲值 ≥$10 升級為 1000 次/日)
- 全帳號每分鐘 20 RPM (requests per minute)
策略:
1. 第一順位用 `openrouter/auto:free` 自動路由,讓 OpenRouter 自己挑可用免費模型
2. 收到 429 (rate limit) 時 retry 同一支等 RPM 恢復,不立刻換下一支耗額度
3. 累計 N 次 429 後直接放棄,避免一次燒光當日配額
"""
import os
import time
import requests


# ── 可調設定 ───────────────────────────────────────────────────────────
# 環境變數 PREFERRED_AI_MODEL 填關鍵字即可指定優先模型,例如 "deepseek" / "qwen" / "gemini"
# 不設定就照 AI_MODELS 預設順序跑。
PREFERRED_AI = os.environ.get("PREFERRED_AI_MODEL", "").strip().lower()

# 429 重試設定
RETRY_429_WAIT  = 15   # 收到 429 等幾秒再 retry 同一支
RETRY_429_TIMES = 1    # 同一支 429 最多 retry 次數 (1 = 等一次後再試一次)
MAX_TOTAL_429   = 3    # 跨模型累計 429 上限,超過直接放棄(保護當日配額)

# 503 重試設定(Service Unavailable 通常是平台短暫過載,等 3 秒再試大多會通)
RETRY_503_WAIT  = 3    # 收到 503 等幾秒再 retry 同一支
RETRY_503_TIMES = 1    # 同一支 503 最多 retry 次數

# 模型清單(依優先序排列,前面失敗就試下一個)
# ⚠️ 免費模型可用性會變動,部署前建議到 https://openrouter.ai/models?max_price=0 確認
# ⚠️ 維護要點:
#   1. 每 1~2 個月回 https://openrouter.ai/models?max_price=0 確認此清單
#   2. 中文語意能力概略順序:DeepSeek ≈ Qwen ≈ GLM > Llama > Mistral
#   3. 第一位放 openrouter/auto:free 讓平台自選當下可用免費模型
AI_MODELS = [
    {"id": "openrouter/auto:free",                       "name": "Auto Free Router"},
    {"id": "deepseek/deepseek-chat-v3-0324:free",        "name": "DeepSeek V3"},
    {"id": "qwen/qwen-2.5-72b-instruct:free",            "name": "Qwen 2.5"},
    {"id": "google/gemini-2.0-flash-exp:free",           "name": "Gemini 2.0"},
    {"id": "openai/gpt-oss-20b:free",                    "name": "GPT-OSS 20B"},
    {"id": "meta-llama/llama-3.3-70b-instruct:free",     "name": "Llama 3.3"},
]


def get_api_key():
    """讀 OpenRouter API Key:
    優先順序 = 環境變數 > Streamlit secrets(只有在 streamlit 環境才會讀)。

    寫成函式而非常數,是為了讓非 streamlit 環境(例如 GitHub Actions 跑 telegram)
    匯入時不會 crash。
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    try:
        import streamlit as st  # 延遲匯入,避免非 streamlit 環境噴 ModuleNotFoundError
        return st.secrets.get("OPENROUTER_API_KEY")
    except Exception:
        return None


def _is_rate_limit_error(exc) -> bool:
    """判斷 exception 是不是 429 rate limit。"""
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


def _is_service_unavailable(exc) -> bool:
    """判斷 exception 是不是 503 service unavailable(暫時性過載,值得 retry)。"""
    msg = str(exc).lower()
    return "503" in msg or "service unavailable" in msg


def call_openrouter_ai(prompt: str, timeout: int = 20, max_tokens: int = 250, models: list = None):
    """依序嘗試模型清單,回傳 (model_name, ai_text);全部失敗回傳 (None, None)。

    Args:
        prompt: 餵給 AI 的內容
        timeout: 單次 API 呼叫的逾時(秒)
        max_tokens: 限制輸出長度。Telegram 簡短點評建議 250,個股深度分析建議 400
        models: 自訂模型清單(預設使用 AI_MODELS 並套用 PREFERRED_AI 排序)。
                呼叫端可傳入單一模型 [{"id":..., "name":...}] 達成「強制指定」效果。

    Notes:
        - 函式內會把 Markdown 殘留(**、##、###、*、`)清掉,避免破壞 HTML/UI 顯示
        - 失敗會吞掉例外、改用 print log,呼叫端不必再包 try
        - 收到 429 時會等 RETRY_429_WAIT 秒後 retry 同一支(等 RPM 恢復)
    """
    api_key = get_api_key()
    if not api_key:
        return None, None

    # 預設清單套用 PREFERRED_AI 排序;呼叫端傳入的自訂清單則保持順序不動
    if models is None:
        models = sorted(
            AI_MODELS,
            key=lambda m: 0 if PREFERRED_AI and PREFERRED_AI in m["id"].lower() else 1
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    total_429 = 0   # 跨模型累計 429 數,達 MAX_TOTAL_429 直接放棄
    # 同一支內最多嘗試次數 = max(429 retry, 503 retry) + 1(首次);實際每次失敗會依錯誤類型決定要不要再 retry
    max_attempts = max(RETRY_429_TIMES, RETRY_503_TIMES) + 1

    for m in models:
        payload = {
            "model": m["id"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }
        attempt_429 = 0
        attempt_503 = 0
        for _attempt in range(max_attempts):
            try:
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers, json=payload, timeout=timeout
                )
                resp.raise_for_status()
                j = resp.json()

                if "choices" in j and j["choices"]:
                    text = j["choices"][0]["message"]["content"].strip()
                    for tok in ("###", "##", "**", "*", "`"):
                        text = text.replace(tok, "")
                    text = text.strip()
                    if text:
                        print(f"   ✅ AI 模型 {m['name']} 回應成功")
                        return m["name"], text
                elif "error" in j:
                    print(f"   ⚠ {m['name']} 拒絕: {j['error'].get('message', '')[:120]}")
                break   # 非 429/503 失敗 → 換下一支
            except requests.exceptions.Timeout:
                print(f"   ⚠ {m['name']} 逾時,換下一個")
                break
            except Exception as e:
                if _is_rate_limit_error(e):
                    total_429 += 1
                    if total_429 >= MAX_TOTAL_429:
                        print(f"   ⛔ 累計 {total_429} 次 429,放棄以保護當日配額")
                        return None, None
                    if attempt_429 < RETRY_429_TIMES:
                        attempt_429 += 1
                        print(f"   ⚠ {m['name']} 429 限速,等 {RETRY_429_WAIT}s 再 retry...")
                        time.sleep(RETRY_429_WAIT)
                        continue
                    else:
                        print(f"   ⚠ {m['name']} 429 retry 後仍失敗,換下一支")
                        break
                elif _is_service_unavailable(e):
                    if attempt_503 < RETRY_503_TIMES:
                        attempt_503 += 1
                        print(f"   ⚠ {m['name']} 503 服務暫時不可用,等 {RETRY_503_WAIT}s 再 retry...")
                        time.sleep(RETRY_503_WAIT)
                        continue
                    else:
                        print(f"   ⚠ {m['name']} 503 retry 後仍失敗,換下一支")
                        break
                else:
                    print(f"   ⚠ {m['name']} 失敗: {str(e)[:120]}")
                    break

    print("   ❌ 所有 AI 模型皆失敗")
    return None, None
