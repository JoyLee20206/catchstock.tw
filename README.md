# 📊 台股全自動選股系統

> 整合 **法人籌碼 / 大戶持股 / 月營收 / 技術面 / 大盤情緒** 的 10 分制台股選股引擎,
> 搭配 Streamlit Web UI、Telegram 每日推播、GitHub Actions 排程,構成一套**完全免費 + 100% 開源** 的個人投資決策系統。

```
GitHub Actions(每日排程)
    ↓ 抓取 TWSE/TPEx/TAIFEX/TDCC/yfinance 資料
cache/*.parquet
    ↓ git commit + push
GitHub repo
    ↓ 同步
┌─────────────────────────┬──────────────────────────┐
│  Streamlit Cloud UI     │  Telegram Bot Daily Push │
│  (互動式選股 + 個股分析)  │  (盤後自動推播達標股)     │
└─────────────────────────┴──────────────────────────┘
```

---

## 🚀 快速開始

### 本機跑

```bash
# 1. clone repo
git clone https://github.com/<你的帳號>/<你的 repo>.git
cd <你的 repo>

# 2. 裝依賴
pip install -r requirements.txt

# 3. 抓資料(第一次 ~ 5 分鐘)
python fetch_cache.py

# 4. 開 UI
streamlit run screening_ui16.py
```

打開 http://localhost:8501 就能用。

### 雲端部署

| 元件 | 平台 | 用途 |
|---|---|---|
| **Streamlit UI** | [Streamlit Community Cloud](https://streamlit.io/cloud)(免費) | 互動式網頁版選股工具 |
| **資料抓取排程** | GitHub Actions(免費) | 每日盤後自動抓資料 + commit 回 repo |
| **每日 Telegram 推播** | GitHub Actions(免費) | 盤後送訊息含 AI 點評 |
| **AI 點評** | [OpenRouter](https://openrouter.ai/) 免費模型 | DeepSeek / Llama / GPT-OSS 輪替 |

**整套系統成本:$0**(每月 OpenRouter 免費額度 50 次足夠日推播)

---

## 🎯 核心功能

### 🏆 1. 10 分制計分選股([screening0515.py](screening0515.py))

| 計分項目 | 滿分 | 訊號 |
|---|---|---|
| 法人籌碼 | 3 | 外資/投信/自營商買賣超 |
| 大戶 vs 散戶 | 3 | TDCC 大戶持股增加 + 散戶減少(籌碼共振) |
| 技術面 | 2 | KD 低檔金叉 + 量價齊揚突破 60 日新高 |
| 基本面 | 1 | 月營收 YoY > 10% 且 MoM > 0 |
| 大盤相對強度 | 1 | 該股近 20 日報酬優於大盤 |

**動態 PASS_SCORE**:
- 大盤跌破季線 → 門檻 +1(空頭從嚴)
- 大盤資料失效 → 門檻 -1(RS 訊號失效補償)

### 🌡️ 2. 大盤情緒指標(6 訊號溫度計 0~100)

| 指標 | 權重 | 資料源 | 評分方式 |
|---|---|---|---|
| 🇺🇸 美股 VIX | 22% | yfinance ^VIX | 絕對門檻(< 15 樂觀 / > 30 恐慌) |
| 🇹🇼 台指波動率 | 11% | ^TWII 20 日年化實現波動率 | 90 日歷史百分位 |
| 📐 大盤位階 | 22% | ^TWII vs MA60 乖離率 | 絕對門檻(±3% / ±8%) |
| 💰 融資水位 | 11% | margin parquet | 90 日歷史百分位 |
| 🏦 外資期貨 | 22% | TAIFEX 大台未平倉 | 90 日歷史百分位(累積 20 日後啟用) |
| 👥 散戶估算 | 11% | TAIFEX 微台反推 | 90 日歷史百分位(累積 20 日後啟用) |

**輸出**:
- 總溫度 0~100(綜合加權)
- 標籤:☀️ 偏熱 / 🌤️ 略偏多 / 🌥️ 中性 / 🌦️ 略偏空 / ❄️ 偏冷
- **歷史趨勢圖**(近 30/90 日溫度走勢)
- **AI 操作建議**(可選,按鈕觸發)

### 🔬 3. 訊號回測引擎([backtest.py](backtest.py))

掃描 11 個訊號的歷史觸發點,算進場後 N 日報酬:

| Tier | 訊號 | 觸發條件 |
|---|---|---|
| 🌟 S | 籌碼共振(散戶↓) | 大戶持股增加 + 散戶減少 |
| 🌟 S | 外資連 5 日買超 | 近 5 日外資總淨額 > 0 |
| 🌟 S | 月營收雙紅突破 | YoY > 10% + MoM > 0 + 突破 60 日新高 |
| ✅ A | 資減券增 | 融資↓3% + 融券↑5% |
| ✅ A | 三大法人同步買超 | 同日 外資+投信+自營 全買 |
| ✅ A | 投信連 5 日買超 | 近 5 日投信總淨額 > 0 |
| ✅ A | 量價齊揚突破 | close ≥ 60 日新高 + 量 ≥ MA20 × 1.5 |
| 🟡 B | 20 日動能 Top 10% | 近 20 日報酬全市場前 10% |
| 🟡 B | 品質突破 | 突破前 10 日量縮 ≥ 6 日 |
| 🟡 B | MA 黃金交叉 | MA20 上穿 MA60 |
| ❌ C | KD 低檔金叉 | K 上穿 D + K < 30 |

支援 **AND/OR 組合測試**、**個股回測**、**11 訊號對照圖**。

### 🎨 4. UI 體驗

| 功能 | 描述 |
|---|---|
| **首頁速覽** | 4 卡 metric(達標數/市場溫度/主流產業/5 日勝率) |
| **狀態總覽** | 3 欄(資料新鮮度 / 大盤多空 / 市場溫度) |
| **快速搜尋** | 輸入代號或名稱關鍵字直跳個股分析 |
| **5 主 tab** | 入選熱度榜 / 產業輪動 / 策略績效 / 訊號回測 / 大盤情緒 |
| **個股 5 sub-tab** | 技術分析 K 線 / 籌碼基本面 / AI 虛擬點評 / 交易筆記 / 資金管理 |
| **自選股警示** | UI 加入 → Telegram 自動監控(MA 跌破/KD 死叉/外資連賣) |
| **下載匯出** | xlsx / 嘉實 dsl / 精誠 xls |

### 📲 5. Telegram 每日推播([telegram_notify.py](telegram_notify.py))

盤後自動推送:
- 大盤戰情摘要(指數/乖離/盤勢)
- AI 冠軍個股點評(模型輪替)
- 大盤情緒溫度
- 達標個股 Top 15(含新進/連續上榜/突破標籤 + 漲跌幅)
- 主流產業群聚警告
- 退場通知(昨天有今天沒)
- 自選股警示(MA 跌破/KD 死叉/外資連賣)
- 近 30 日策略績效摘要

---

## 📁 核心檔案說明

```
0518選股程式/
├── README.md              ← 你正在看的
├── requirements.txt       ← 依賴清單
│
├── fetch_cache.py         ← 資料抓取主腳本(GitHub Actions 排程)
├── screening0515.py       ← 選股計分核心引擎
├── screening_ui16.py      ← Streamlit Web UI 主入口
├── telegram_notify.py     ← Telegram 每日推播
│
├── market_sentiment.py    ← 大盤情緒 6 指標
├── backtest.py            ← 訊號回測引擎
├── ai_helper.py           ← OpenRouter AI 共用呼叫
│
├── cache_status.py        ← 快取新鮮度檢查
├── data_health.py         ← 資料健康度檢查
├── picks_history.py       ← 歷史選股紀錄(v2 schema)
├── performance.py         ← 策略績效追蹤
├── industry_rotation.py   ← 產業輪動分析
├── watchlist_alerts.py    ← 自選股警示
│
├── cache/                 ← 資料快取(由 fetch_cache.py 寫入,git tracked)
│   ├── daily_*.parquet              ← 全市場 180 天 K 線
│   ├── info_*.parquet               ← 股票清單 + 產業別
│   ├── institutional_*.parquet      ← 三大法人買賣超(近 15 日)
│   ├── margin_*.parquet             ← 融資融券(近 15 日)
│   ├── holders_*.parquet            ← TDCC 大戶持股(近 12 週)
│   ├── revenue_*.parquet            ← 月營收(累積)
│   ├── twii_*.parquet               ← ^TWII 2 年指數 K 線
│   ├── vix_*.parquet                ← ^VIX 60 日恐慌指數
│   ├── retail_futures_history.json  ← 散戶期貨歷史(算百分位用)
│   ├── fi_futures_history.json      ← 外資期貨歷史(算百分位用)
│   ├── sentiment_history.json       ← 大盤溫度歷史(畫趨勢圖用)
│   ├── previous_picks.json          ← 每日達標股(算入選熱度/績效用)
│   ├── watchlist.json               ← 自選股清單(UI/TG 共用)
│   ├── notes.json                   ← 交易筆記
│   ├── last_fetch_daily.txt         ← daily 抓取時戳
│   └── ...
│
└── .github/workflows/     ← GitHub Actions(本地不一定 sync,雲端 repo 上)
    ├── fetch.yml          ← 排程抓資料 + commit
    └── notify.yml         ← 盤後 Telegram 推播
```

---

## 🗄 資料抓取 / Cache 系統

### 各資源更新頻率

| 資源 | 更新頻率 | Gate 邏輯 | 來源 |
|---|---|---|---|
| `info` | 每日全量 | 每日重抓(反映新上市/下市) | TWSE OpenAPI(主)+ ISIN HTML(備) |
| `institutional` | 每日增量 | 補近 15 個工作日缺漏 | TWSE/TPEx HTML |
| `margin` | 每日增量 | 同上 | TWSE/TPEx HTML |
| `daily` | 每日重抓 | --force-daily 強制重抓 | yfinance |
| `holders` | **週六~一**才有新 | Weekly gate(cache 有上週五就略過) | TDCC OpenData |
| `revenue` | **每月 10 日**前後 | Day-13 gate(13 日後若有上月就略過) | TWSE/TPEx OpenAPI |
| `twii` | 每日重抓 | 同 daily | yfinance |
| `vix` | 每日重抓 | 同 daily | yfinance |

### 三層降級策略(以股票清單為例)

```
[1] TWSE OpenAPI (JSON)     ← 主來源,< 10 秒
    ↓ 失敗
[2] ISIN HTML               ← 備援,30~60 秒
    ↓ 失敗
[3] 昨日 parquet           ← 最後一道保險,瞬時
```

### Cache 路徑統一

所有 cache 檔案路徑都以 `CACHE_DIR = Path("cache")` 為基準,不要 hardcode `"cache/xxx.json"`。

---

## ⚙️ 環境變數

| 變數 | 必填 | 說明 |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | TG 推播需要 | BotFather 拿到的 token |
| `TELEGRAM_CHAT_ID` | TG 推播需要 | 你的 chat ID |
| `OPENROUTER_API_KEY` | AI 功能需要 | https://openrouter.ai 免費註冊 |
| `PREFERRED_AI_MODEL` | 可選 | 關鍵字,例 `"deepseek"` 優先用 DeepSeek |

**Streamlit Cloud 設法**:Manage app → Secrets → 貼上 TOML 格式
**GitHub Actions 設法**:Settings → Secrets and variables → Actions → New secret

---

## 🧠 設計決策紀錄(本 session 主要踩雷與教訓)

### 1. yfinance 永遠不要相信會穩定運作

**踩雷**:
- 富邦 VIX ETF (00677U) 2024 下市
- yfinance 0.2.50+ 把 daily 抓取的 Date 欄改成 Datetime → KeyError
- Yahoo 對 GitHub Actions 雲端 IP 嚴格 anti-bot,偶爾整批 fail
- `_load_twii_cached` TTL 1 小時 → yfinance 失敗一次卡 60 分鐘

**對策**:
- 所有 yfinance 呼叫都先讀本地 parquet,失敗才打 yfinance
- `fetch_cache.py` 把 `^TWII` / `^VIX` 落地到 parquet
- 多日期欄名相容(Date / Datetime / index / level_0)
- TTL 縮短 + session-level 自動重試

### 2. Streamlit `@st.cache_data` 會擋住副作用

**踩雷**:
```python
@st.cache_data
def compute_sentiment(cache_dir):
    temp = 算溫度()
    write_history_file(cache_dir, temp)  # ← 副作用!
    return temp
```
第二次呼叫 → 直接回 cache → **write_history_file 永遠不跑** → 檔案永遠空。

**對策**:讀寫分離。把寫檔的 side effect 抽到 cache 外:
```python
@st.cache_data
def _load_sentiment_cached():
    return compute_sentiment(CACHE_DIR)   # 純函式

def _get_sentiment_and_persist():
    s = _load_sentiment_cached()          # 拿快取結果
    persist_history(CACHE_DIR, s)         # caller 端做寫檔(每次都跑)
    return s
```

### 3. 情緒指標的「絕對門檻」會過時、要持續校準

**踩雷**:外資期貨淨口數門檻 2018 時 ±15k 是極值,2026 時 ±50k 才算極值。半年要校準一次。

**對策**:**百分位制**。把當前值放進「過去 90 日的分布」算百分位,**永遠不需手動調**。等於把校準工作自動化。

### 4. yfinance threads=True 會出現 SQLite database locked

**現象**:多執行緒下載時,yfinance 內部 SQLite cache 偶爾 race condition,1~3 個 ticker 失敗。

**對策**:接受(失敗率 < 0.1%,下次跑會補)。或改 threads=False(慢 3~5 倍,不值得)。

### 5. TWSE OpenAPI 的「產業別」是數字代碼,ISIN 是中文名

**踩雷**:升級到 OpenAPI 主來源後,UI 上篩選器出現「05」「22」「24」這種數字 chip。

**對策**:加 `TWSE_INDUSTRY_CODE_MAP` + `_normalize_industry()` 統一轉中文。

---

## 🛠 維運常見問題

### Q1: 大盤資料抓取失敗,本次無 RS 計分

**原因**:Yahoo 對 GA 雲端 IP 限速,yfinance 抓 ^TWII 失敗。
**修法**:已改成讀 `cache/twii_*.parquet` 優先,fallback 才打 yfinance。確認 GA 排程有 push twii parquet 回 repo。

### Q2: GitHub Actions 跑 daily 出現「Failed to get ticker XXXX.TW」

**原因**:Yahoo IP 限速 / yfinance 內部 SQLite race。
**對策**:
- 1~3 個 ticker 失敗 → 正常,不用管
- 全部 400 ticker 失敗 → 等下次 GA 跑(間隔幾小時),或本機跑 `fetch_cache.py --force-daily` 後手動 commit cache

### Q3: 訊號回測點下去要等很久

**原因**:第一次跑要建 11 個訊號矩陣(掃描 180 天)~ 5-7 秒。
**現況**:已加 `_build_signal_matrices_cached`,**改參數後秒切**。隔天 cache_date 變才會重建。

### Q4: 月營收顯示「已有今日快取,略過」但其實昨天才更新

**原因**:Day-13 gate 邏輯:每月 13 日起若 cache 已包含上月資料就略過,避免浪費 API。
**強制重抓**:`python fetch_cache.py --force`(`--force-daily` 不會碰 revenue)。

### Q5: Streamlit Cloud 體感很慢

**檢查**:
1. 是否第一次部署 / 容器剛重啟 → 冷啟動本來就慢(parquet 沒 sync 完)
2. GA 排程有沒有把 cache push 回 repo → 看 GA log 最後一行 `將快取存回 GitHub`
3. Streamlit Cloud Reboot 試試 → Manage app → Reboot

### Q6: ImportError after deploy

**原因**:Streamlit Cloud module cache 沒同步。
**對策**:Manage app → Reboot,強制 Python process 重啟。

---

## 🚦 系統健康度自我檢查

打開 UI 後,觀察以下幾個指標,**全綠表示系統正常**:

| 指標 | 健康 | 異常 |
|---|---|---|
| **狀態總覽:資料日期** | 今日 / 上交易日 ✓ | 早於 2 個交易日 ⚠️ |
| **狀態總覽:大盤** | 📈 多頭 / 📉 空頭(有數字) | N/A ⚠️ |
| **狀態總覽:市場溫度** | 0~100 數字 | N/A ⚠️ |
| **資料健康度警示** | 不出現 | 紅色 banner 警告 |
| **首頁 4 卡 metric** | 都有值 | "—" 表示資料缺 |

UI 上有「💡 策略邏輯導覽 (FAQ)」expander,點開有完整 FAQ。

---

## 🤝 貢獻

這是個人專案,但歡迎開 issue 討論。送 PR 前請:
1. 確認 syntax check 通過(`python -c "import ast; [ast.parse(open(f).read()) for f in ('screening_ui16.py', 'market_sentiment.py', ...)]"`)
2. 跑過 fetch_cache.py 一次確認沒踩雷
3. 重大改動同步更新 README 對應段落

---

## 📜 授權

MIT License — 拿去用、改、商用都可以,**僅限個人投資決策參考**,不構成投資建議。

---

## 🙏 致謝

- [Streamlit](https://streamlit.io) — 整套 UI 免費 hosting
- [yfinance](https://github.com/ranaroussi/yfinance) — yahoo finance python wrapper
- [TWSE OpenAPI](https://openapi.twse.com.tw/) / [TPEx OpenAPI](https://www.tpex.org.tw/openapi/) — 台股官方資料
- [TDCC OpenData](https://opendata.tdcc.com.tw/) — 大戶持股
- [TAIFEX](https://www.taifex.com.tw/) — 三大法人期貨
- [OpenRouter](https://openrouter.ai/) — 免費 AI 模型輪替

---

**Made with ❤️ for individual investors who want institutional-grade data without paying institutional prices.**
