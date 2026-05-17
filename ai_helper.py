"""OpenRouter AI 共用模組(多模型 fallback)

供 telegram_notify.py 與 screening_ui16.py 共用,
避免兩邊各維護一份模型清單與呼叫邏輯。
"""
import os
import requests


# ── 可調設定 ───────────────────────────────────────────────────────────
# 環境變數 PREFERRED_AI_MODEL 填關鍵字即可指定優先模型,例如 "deepseek" / "qwen" / "gemini"
# 不設定就照 AI_MODELS 預設順序跑。
PREFERRED_AI = os.environ.get("PREFERRED_AI_MODEL", "").strip().lower()

# 模型清單(依優先序排列,前面失敗就試下一個)
# ⚠️ 免費模型可用性會變動,部署前建議到 https://openrouter.ai/models?max_price=0 確認
AI_MODELS = [
    {"id": "deepseek/deepseek-chat-v3-0324:free",     "name": "DeepSeek V3"},
    {"id": "qwen/qwen-2.5-72b-instruct:free",         "name": "Qwen 2.5"},
    {"id": "google/gemini-2.0-flash-exp:free",        "name": "Gemini 2.0"},
    {"id": "meta-llama/llama-3.3-70b-instruct:free",  "name": "Llama 3.3"},
    {"id": "openai/gpt-oss-20b:free",                 "name": "GPT-OSS 20B"},
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

    for m in models:
        try:
            payload = {
                "model": m["id"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": max_tokens,
            }
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=timeout
            )
            resp.raise_for_status()
            j = resp.json()

            if "choices" in j and j["choices"]:
                text = j["choices"][0]["message"]["content"].strip()
                # 強制清掉殘留 Markdown(免費模型常常不聽 prompt 指令)
                for tok in ("**", "##", "###", "*", "`"):
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
