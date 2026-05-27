import os, sys, warnings, time, requests, io, random
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 固定用台北時區,避免雲端 (UTC) 跨日時檔名與 cache_status / telegram_notify (TPE) 不一致
TPE_TZ = timezone(timedelta(hours=8))
import pandas as pd
import yfinance as yf
from io import StringIO

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# 參數設定區 (100% 零 API 金鑰)
# ==========================================
CACHE_DIR = Path("cache")

# 旗標:
#   --force        無視所有 cache,全部重抓(info/法人/融資券/daily/大戶/營收)
#   --force-daily  只強制重抓 daily K 線,其他資源若有當日 cache 仍略過
#                  典型用途:盤中跑過後,收盤再按一次重抓「真實收盤價」
FORCE       = "--force" in sys.argv
FORCE_DAILY = "--force-daily" in sys.argv

# 爬蟲延遲設定 (秒)
TWSE_DELAY = 3.0
TPEX_DELAY = 3.0
TDCC_DELAY = 2.0
# (舊 MOPS_DELAY 已隨 MOPS HTML 爬蟲一併移除;OpenAPI 走 session_get 內建退讓)

# TPEx 結構守門員 (防止 TPEx 改版後硬編碼 index 默默抓錯欄位)
# 設計:每次抓取後抽樣 N 筆做「合計欄 = 細項加總」的橫向驗證
#       失敗率 > 50% 即視為改版,拒絕寫入該日資料以保護 cache
TPEX_VALIDATE_SAMPLE = 5    # 每次抓取後抽樣前 N 筆做結構驗證
TPEX_VALIDATE_TOL    = 2    # 加總驗證容差(張),容許四捨五入/整數轉換的微小誤差

# 增量類 cache 的歷史保留天數 (防止無限增長拖累 I/O)
# 套用對象: institutional / margin / holders
# 不套用: revenue (YoY 需累積 ≥15 個月) / info (每日全量重抓) / daily (yfinance 全量)
# screening1.py 對 institutional/margin 只看 tail(LOOKBACK_DAYS=5),holders 看 tail(LARGE_HOLDER_WEEKS=4 週)
# → 90 天 (~64 交易日 / ~12 週) 對所有訊號都遠遠夠用
CACHE_RETAIN_DAYS = 90

# UA 池 (隨機輪替,降低行為指紋)
HTTP_HEADERS_POOL = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"},
]

# Session 重用 (keep-alive) + 失敗熔斷
# 注意:改為「本次執行累計失敗總數」,成功不 reset,
# 避免「TWSE 200 / TPEx 503 交錯」使熔斷永不觸發而導致 IP 被封。
SESSION = requests.Session()
SESSION.verify = False
MAX_TOTAL_FAILS    = 8     # 本次執行累計失敗上限 (>= 則中止)
MAX_DAY_FAILS      = 10     # 單一 batch 內連續工作日失敗上限
COOKIE_CLEAR_EVERY = 50    # 每 N 個請求清一次 cookie,避免累積被追蹤
_fail_counter    = {"total": 0}
_request_counter = {"n": 0}

CACHE_DIR.mkdir(exist_ok=True)

# 全部用台北時區計算,確保檔名與 TPE 推播/UI 日期一致
_now_tpe = datetime.now(TPE_TZ)
today = _now_tpe.strftime("%Y-%m-%d")
d20   = (_now_tpe - timedelta(days=20)).strftime("%Y-%m-%d")
# 180 calendar days ≈ 128 trading days,給 MA60 / 60日新高約 68 天緩衝
# (原 d120 ≈ 86 trading days,連假月份會緊到不足 60 → 部分股票指標漏判)
d180  = (_now_tpe - timedelta(days=180)).strftime("%Y-%m-%d")

def path_for(name):   return CACHE_DIR / f"{name}_{today}.parquet"

def need_fetch(name):
    """判斷是否需要重抓。
    - FORCE: 全部資源強制重抓
    - FORCE_DAILY: 只有 'daily' 強制重抓,其他資源若已有當日檔仍略過
    - 預設: 沒當日檔就抓
    """
    if FORCE:
        return True
    if FORCE_DAILY and name == "daily":
        return True
    return not path_for(name).exists()

def cleanup_old_cache(name):
    """刪除同類型的舊快取,只保留今日這份。
    安全前提:所有累積型快取(institutional/margin/daily/holders/revenue)在寫入今日檔時,
    已將舊檔資料完整 merge 進來,舊檔是新檔的真子集,刪除不會遺失任何資料。
    info 為每日全量重抓,舊檔本即無用。
    本函式必須在 path_for(name) 確認寫入成功後才呼叫。"""
    today_file = path_for(name)
    deleted = 0
    for old_file in sorted(CACHE_DIR.glob(f"{name}_*.parquet")):
        if old_file != today_file and old_file.exists():
            try:
                old_file.unlink()
                deleted += 1
            except Exception as e:
                print(f"   [清理警告] 無法刪除 {old_file.name}: {e}")
    if deleted:
        print(f"   [清理] 已刪除 {name} 舊快取 {deleted} 份")

def to_int(v):
    if pd.isna(v) or v == "" or v == "--": return 0
    try: return int(str(v).replace(",", "").replace(" ", ""))
    except (ValueError, TypeError): return 0

def _fallback_prev_to_today(name):
    """失敗墊檔:把最新的舊 cache 拷貝為今日檔名。
    用途:抓取失敗時讓今日檔存在,避免:
      (a) 隔日執行時 need_fetch 仍 True 而重複嘗試已知失敗源
      (b) screening1.py 用 latest() 抓到很舊的 cache 但日期沒提示 (今日檔名讓使用者感受到「資料是今天的」是錯覺,應自行注意)
    安全條件:
      - 今日檔已存在則不動 (避免覆蓋本次部分成功的資料)
      - 沒有任何舊檔則不動 (初次執行就失敗,保留空狀態讓使用者察覺)
    回傳: True=有墊檔, False=未墊檔
    """
    today_file = path_for(name)
    if today_file.exists():
        return False
    prev_files = sorted(CACHE_DIR.glob(f"{name}_*.parquet"))
    if not prev_files:
        return False
    try:
        pd.read_parquet(prev_files[-1]).to_parquet(today_file)
        cleanup_old_cache(name)
        print(f"   [墊檔] {name} 抓取失敗,已將前次成功的 {prev_files[-1].name} 拷貝為今日檔")
        return True
    except Exception as e:
        print(f"   [墊檔失敗] {name}: {e}")
        return False

def _trim_by_retention(df, label=""):
    """依 CACHE_RETAIN_DAYS 截斷 DataFrame,只保留 cutoff 日期之後的 row。
    df 必須有 'date' 欄 (字串格式 'YYYY-MM-DD')。
    回傳截斷後的 df,並在實際截斷時印出統計。"""
    if df is None or df.empty or "date" not in df.columns:
        return df
    cutoff = (datetime.now(TPE_TZ) - timedelta(days=CACHE_RETAIN_DAYS)).strftime("%Y-%m-%d")
    before = len(df)
    df_trim = df[df["date"] >= cutoff].reset_index(drop=True)
    after = len(df_trim)
    if before > after:
        print(f"   [{label}] 保留期截斷:{before:,} → {after:,} 筆 (僅留 {CACHE_RETAIN_DAYS} 天內)")
    return df_trim

# ==========================================
# TPEx 結構守門員
# ==========================================
def _tpex_validate_inst(rows):
    """檢查 TPEx 三大法人 row 的 index 對應是否仍與 fetch_tpex_institutional 假設一致。
    驗證規則 (硬性,任何 row 全部通過才算 passed):
      - 各法人「買賣超 = 買 - 賣」(共 6 組:外資不含自營/外資自營商/外資合計/投信/自營自行/自營避險)
      - 外資合計買進 = 外資(不含自營) + 外資自營商
      - 外資合計賣出 = 外資(不含自營) + 外資自營商
    這 8 條規則只要 TPEx 欄位順序改變就一定有人會失準,是強守門。
    回傳 (passed, failed) — 失敗多於成功即視為改版。
    """
    passed = failed = 0
    for r in rows[:TPEX_VALIDATE_SAMPLE]:
        if len(r) < 20:
            failed += 1; continue
        try:
            checks = [
                # 規則 1: 各法人 買賣超 = 買 - 賣
                (to_int(r[4]),  to_int(r[2])  - to_int(r[3])),    # 外資(不含自營商)
                (to_int(r[7]),  to_int(r[5])  - to_int(r[6])),    # 外資自營商
                (to_int(r[10]), to_int(r[8])  - to_int(r[9])),    # 外資及陸資合計
                (to_int(r[13]), to_int(r[11]) - to_int(r[12])),   # 投信
                (to_int(r[16]), to_int(r[14]) - to_int(r[15])),   # 自營(自行買賣)
                (to_int(r[19]), to_int(r[17]) - to_int(r[18])),   # 自營(避險)
                # 規則 2: 外資合計 = 不含自營商 + 自營商
                (to_int(r[8]), to_int(r[2]) + to_int(r[5])),      # 合計買進
                (to_int(r[9]), to_int(r[3]) + to_int(r[6])),      # 合計賣出
            ]
            if all(abs(a - e) <= TPEX_VALIDATE_TOL for a, e in checks):
                passed += 1
            else:
                failed += 1
        except (ValueError, IndexError, TypeError):
            failed += 1
    return passed, failed

def _tpex_validate_margin(rows):
    """檢查 TPEx 融資融券 row 的 index 對應是否仍與 fetch_tpex_margin 假設一致。
    驗證規則 (弱性,因融資/融券「今日 = 前日 + 買進 - 賣出 - 現金償還」的中間欄位無法窮舉驗證):
      - stock_id 為 4 位數字 (r[0])
      - r[2] (融資前日)、r[6] (融資今日)、r[10] (融券前日)、r[14] (融券今日) 皆 >= 0
      - row 至少 15 欄
    若 r[14] 在改版後變成「融券限額」之類,絕大多數股票仍 >= 0,此驗證可能漏判。
    但若改成負值欄位 (例如「買賣超」) 則可有效抓到。
    回傳 (passed, failed)
    """
    passed = failed = 0
    for r in rows[:TPEX_VALIDATE_SAMPLE]:
        if len(r) < 15:
            failed += 1; continue
        try:
            sid = str(r[0]).strip()
            checks = [
                len(sid) >= 4,         # <--- 改成這樣：只要代號長度 >= 4 即可，放行 ETF
                to_int(r[2])  >= 0,    # 融資前日餘額
                to_int(r[6])  >= 0,    # 融資今日餘額
                to_int(r[10]) >= 0,    # 融券前日餘額
                to_int(r[14]) >= 0,    # 融券今日餘額
            ]
            if all(checks):
                passed += 1
            else:
                failed += 1
        except (ValueError, IndexError, TypeError):
            failed += 1
    return passed, failed

def trading_days(start_date, end_date):
    days = []
    d = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while d <= end:
        if d.weekday() < 5: days.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return days

# ==========================================
# 穩健 HTTP 取得 (Session + 指數退讓 + UA 輪替 + 失敗熔斷)
# ==========================================
def _bump_fail(reason=""):
    """失敗 +1;達門檻即中止以保護 IP。"""
    _fail_counter["total"] += 1
    if _fail_counter["total"] >= MAX_TOTAL_FAILS:
        print(f"!!! 本次執行累計失敗 {_fail_counter['total']} 次 ({reason}),中止以保護 IP。")
        sys.exit(1)

# TWSE/TPEx/MOPS 頻繁限速時,實務 body 特徵 (非嚴格 JSON 或含「頻繁」等字眼)
_THROTTLE_KEYWORDS = ("查詢過於頻繁", "Too Many Requests", "Access Denied", "rate limit", "Blocked")

def _wait_with_jitter(attempt):
    # 加上 0~2 秒隨機 jitter,避免退讓時間形成固定指紋
    return 3 * (3 ** attempt) + random.uniform(0, 2)

def _safe_text_head(resp, n=512):
    # TWSE/TPEx 偶爾吐壞編碼,避免 resp.text 直接 raise UnicodeDecodeError
    try:
        return resp.text[:n] if resp.text else ""
    except (UnicodeDecodeError, Exception):
        try:
            return resp.content[:n].decode("utf-8", errors="ignore")
        except Exception:
            return ""

def session_get(url, params=None, method="GET", data=None, timeout=15, max_retries=5,
                extra_headers=None):
    """
    強化版 HTTP 取得:
    - 5 次退讓 (~3s → 9s → 27s → 81s → 243s) + jitter,降低行為指紋
    - 檢查 200 body 是否為限速 HTML (避免 stat 200 但被 ban 的假象)
    - 任何失敗(含限速 200)都計入 _fail_counter,成功不 reset
    - 總失敗達 MAX_TOTAL_FAILS 即 sys.exit
    - 每 COOKIE_CLEAR_EVERY 個請求清一次 SESSION cookie,避免累積指紋
    """
    _request_counter["n"] += 1
    if _request_counter["n"] % COOKIE_CLEAR_EVERY == 0:
        SESSION.cookies.clear()

    for attempt in range(max_retries):
        try:
            headers = random.choice(HTTP_HEADERS_POOL).copy()
            if extra_headers:
                headers.update(extra_headers)
            if method == "POST":
                resp = SESSION.post(url, data=data, headers=headers, timeout=timeout, verify=False)
            else:
                resp = SESSION.get(url, params=params, headers=headers, timeout=timeout, verify=False)

            if resp.status_code == 200:
                # 驗證 body 不是限速警示頁 (TWSE 會回 200 + HTML 警告文字)
                snippet = _safe_text_head(resp, 512)
                if any(kw in snippet for kw in _THROTTLE_KEYWORDS):
                    wait = _wait_with_jitter(attempt)
                    print(f"      [警告] HTTP 200 但 body 疑似限速警告 第 {attempt+1} 次,等待 {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                return resp
            if resp.status_code in (429, 503):
                wait = _wait_with_jitter(attempt)
                print(f"      [警告] HTTP {resp.status_code} (限速) 第 {attempt+1} 次,等待 {wait:.1f}s 退讓...")
                time.sleep(wait)
                continue
            # 其他 4xx/5xx → 放棄本次且計入失敗
            _bump_fail(f"HTTP {resp.status_code}")
            return None
        except Exception as e:  # 含 SSLError / ConnectionError / Timeout
            wait = _wait_with_jitter(attempt)
            print(f"      [警告] 連線錯誤 {type(e).__name__} 第 {attempt+1} 次,等待 {wait:.1f}s...")
            time.sleep(wait)
    # 所有 retry 耗盡 → 計入總失敗
    _bump_fail("retry 耗盡")
    return None

# ==========================================
# [1] 官方股票清單爬蟲 (OpenAPI 主要 + ISIN 備用)
# ──────────────────────────────────────────
# 原本以 isin.twse.com.tw HTML 為主來源,但對雲端 IP 限速嚴重,
# 加上回應為 ~1MB HTML 表格 + pd.read_html 解析,單次 30-90 秒,
# GitHub Actions 上常看起來像「卡住」。OpenAPI 走 JSON,
# 體積小、速度快(< 10 秒)且無被封鎖紀錄,因此改為主要來源。
# ==========================================
def _filter_stock_info(df):
    """共用過濾邏輯：金融類 / ETF / -KY / 特別股"""
    ex_kw = ["銀行", "保險", "證券", "金融", "票券", "期貨", "金控", "存託憑證", "受益", "債"]
    mask = (
        (~df["industry_category"].fillna("").str.contains("|".join(ex_kw), na=False)) &
        (~df["stock_id"].str.startswith(("00", "91"))) &
        (~df["stock_name"].fillna("").str.contains("-KY", na=False)) &
        (~df["stock_name"].fillna("").str.endswith("特"))
    )
    return df[mask].drop_duplicates(subset=["stock_id"]).reset_index(drop=True)

# TWSE 產業分類代碼 → 中文名稱對映
# TWSE OpenAPI 的「產業別」欄位有時回 2 位數字代碼(05/22/24/...)、有時回中文名稱,
# 不一致很煩,所以一律 normalize 成中文名。資料來源:TWSE 官方產業分類表。
TWSE_INDUSTRY_CODE_MAP = {
    "01": "水泥工業",      "02": "食品工業",      "03": "塑膠工業",
    "04": "紡織纖維",      "05": "電機機械",      "06": "電器電纜",
    "08": "玻璃陶瓷",      "09": "造紙工業",      "10": "鋼鐵工業",
    "11": "橡膠工業",      "12": "汽車工業",      "14": "建材營造",
    "15": "航運業",        "16": "觀光事業",      "17": "金融保險",
    "18": "貿易百貨",      "20": "其他業",        "21": "化學工業",
    "22": "生技醫療業",    "23": "油電燃氣業",    "24": "半導體業",
    "25": "電腦及週邊設備業","26": "光電業",      "27": "通信網路業",
    "28": "電子零組件業",  "29": "電子通路業",    "30": "資訊服務業",
    "31": "其他電子業",    "32": "文化創意業",    "33": "農業科技業",
    "34": "電子商務業",    "38": "觀光餐旅",      "39": "居家生活",
    "40": "數位雲端",      "41": "運動休閒",      "80": "管理股票",
}


def _normalize_industry(raw: str) -> str:
    """把 TWSE OpenAPI 回的「產業別」標準化為中文名稱。

    輸入可能是:
      - 純數字代碼 "05"  → 對映到 "電機機械"
      - 中文名稱 "半導體業" → 原樣回傳
      - 不在 mapping 內的代碼 → 原樣回傳(讓使用者至少看到原始值)
      - 空字串 / None → 回傳 ""
    """
    s = (raw or "").strip()
    if not s:
        return ""
    # 1-2 位數字代碼 → 查表;查不到原樣回傳
    if s.isdigit() and len(s) <= 3:
        return TWSE_INDUSTRY_CODE_MAP.get(s.zfill(2), s)
    return s


def _fetch_stock_info_openapi():
    """主來源:從 TWSE OpenAPI 抓股票清單(JSON,< 10 秒)。"""
    print("   主來源:TWSE OpenAPI(JSON,通常 < 10 秒)")
    api_urls = {
        "twse": "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",  # 上市
        "tpex": "https://openapi.twse.com.tw/v1/opendata/t187ap03_O",  # 上櫃
    }
    all_stocks = []
    for mode, url in api_urls.items():
        try:
            import requests as _req
            t0 = time.time()
            print(f"   ↓ 下載 OpenAPI {mode} ...", flush=True)
            resp = _req.get(url, timeout=20, verify=False)
            elapsed = time.time() - t0
            if not resp.content.strip() or resp.text.lstrip().startswith("<"):
                print(f"   !!! OpenAPI {mode} 無效回應({elapsed:.1f}s),略過")
                continue
            data = resp.json()
            rows = []
            for item in data:
                sid = str(item.get("公司代號", "")).strip()
                if not sid or not sid.isdigit() or len(sid) != 4:
                    continue
                rows.append({
                    "stock_id":          sid,
                    "stock_name":        str(item.get("公司簡稱", item.get("公司名稱", ""))).strip(),
                    "industry_category": _normalize_industry(
                        str(item.get("產業別", item.get("產業類別", "")))
                    ),
                    "type":              mode,
                })
            if rows:
                all_stocks.append(pd.DataFrame(rows))
                print(f"   OpenAPI {mode}: {len(rows)} 檔({elapsed:.1f}s)", flush=True)
        except Exception as e:
            print(f"   !!! OpenAPI {mode} 失敗: {e}")
    if not all_stocks:
        return pd.DataFrame()
    return _filter_stock_info(pd.concat(all_stocks, ignore_index=True))

def _fetch_stock_info_isin(modes: list = None):
    """備援:從 isin.twse.com.tw 抓 HTML 股票清單(慢,雲端 IP 限速重)。

    modes: 指定要抓的子集(['twse']/['tpex']/None=全抓)。
           OpenAPI 部分失敗時,只補抓缺的市場可省一半時間。
    """
    all_urls = {
        "twse": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",  # 上市
        "tpex": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4",  # 上櫃
    }
    urls = {k: v for k, v in all_urls.items() if (modes is None or k in modes)}
    if not urls:
        return pd.DataFrame()
    print(f"   [備援] 改用 isin.twse.com.tw HTML 來源 modes={list(urls.keys())} "
          f"(可能 1-3 分鐘)...", flush=True)
    import ssl, urllib.request
    _ssl_ctx = ssl.create_default_context()
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = ssl.CERT_NONE

    all_stocks = []
    for mode, url in urls.items():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            t0 = time.time()
            print(f"   ↓ 下載 ISIN {mode} (timeout=30s,逾時改用昨日清單降級)...", flush=True)
            with urllib.request.urlopen(req, context=_ssl_ctx, timeout=30) as r:
                raw = r.read()
            print(f"      下載完成 {len(raw)/1024:.0f} KB ({time.time()-t0:.1f}s),解析 HTML 表格中...",
                  flush=True)
            text = raw.decode("ms950", errors="replace")
            df = pd.read_html(StringIO(text))[0]
        except Exception as e:
            print(f"   !!! 抓取 {mode} 清單 (下載) 失敗: {type(e).__name__}: {e}")
            continue
        try:
            df.columns = df.iloc[0]
            df = df.iloc[1:]
            # 避免 PyArrow str.match 問題，改用 apply 純 Python 比對
            def _is_stock_row(row):
                code_name = str(row.get("有價證券代號及名稱", ""))
                parts = code_name.replace("　", " ").split()
                # 備註欄在真實股票行是 NaN，分類標題行才是「股票」
                # 直接用代號是否為 4 位純數字來判斷
                return len(parts) >= 2 and len(parts[0]) == 4 and parts[0].isdigit()
            mask = df.apply(_is_stock_row, axis=1)
            df = df[mask].copy()
            def _split_code(x):
                return str(x).replace("　", " ").split()
            df["stock_id"]          = df["有價證券代號及名稱"].apply(lambda x: _split_code(x)[0])
            df["stock_name"]        = df["有價證券代號及名稱"].apply(lambda x: _split_code(x)[1] if len(_split_code(x)) > 1 else "")
            df["industry_category"] = df["產業別"]
            df["type"]              = mode
            all_stocks.append(df[["stock_id", "stock_name", "industry_category", "type"]])
            print(f"   ISIN {mode}: {len(df)} 檔解析完成", flush=True)
        except Exception as e:
            print(f"   !!! {mode} 清單解析失敗: {type(e).__name__}: {e}")
            continue
        time.sleep(2)

    if not all_stocks:
        return pd.DataFrame()

    final_info = pd.concat(all_stocks, ignore_index=True)
    return _filter_stock_info(final_info)


def _load_yesterday_info_subset(modes):
    """讀最近一份 info_*.parquet,只回需要 modes 的部分(降級用)。
    用昨日清單頂替缺失市場——每日上下市異動極少,延一天可接受。
    """
    try:
        files = sorted(CACHE_DIR.glob("info_*.parquet"))
        if not files:
            return pd.DataFrame()
        df = pd.read_parquet(files[-1])
        if "type" not in df.columns:
            return pd.DataFrame()
        sub = df[df["type"].isin(modes)].copy()
        if sub.empty:
            return pd.DataFrame()
        print(f"   ⤵ 降級:用昨日 {files[-1].name} 補抓 {modes} {len(sub)} 檔", flush=True)
        return sub
    except Exception as e:
        print(f"   !!! 昨日 info parquet 讀取失敗: {e}")
        return pd.DataFrame()


def fetch_public_stock_info():
    """股票清單主入口:三層降級。

    Tier 1 — OpenAPI(快,< 10 秒)
    Tier 2 — ISIN HTML(慢,1–3 分鐘,只補缺的市場)
    Tier 3 — 昨日 info parquet(瞬時,只補仍缺的市場)
    任一 tier 補齊就停。
    """
    df_api = _fetch_stock_info_openapi()
    api_modes = set(df_api["type"].unique()) if (not df_api.empty and "type" in df_api.columns) else set()
    missing = {"twse", "tpex"} - api_modes
    if not missing:
        return df_api

    print(f"   !!! OpenAPI 缺 {sorted(missing)} 市場,試 ISIN HTML 補抓...", flush=True)
    df_isin = _fetch_stock_info_isin(modes=sorted(missing))
    isin_modes = set(df_isin["type"].unique()) if (not df_isin.empty and "type" in df_isin.columns) else set()
    still_missing = missing - isin_modes

    parts = [df for df in (df_api, df_isin) if not df.empty]

    if still_missing:
        print(f"   !!! ISIN 也未補齊,缺 {sorted(still_missing)},改用昨日 parquet 降級...", flush=True)
        df_yest = _load_yesterday_info_subset(sorted(still_missing))
        if not df_yest.empty:
            parts.append(df_yest)

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=["stock_id"], keep="first").reset_index(drop=True)
    return combined

# ==========================================
# [2-3] TWSE / TPEx 法人與資券
# ==========================================
def fetch_twse_institutional(date_str):
    url = "https://www.twse.com.tw/rwd/zh/fund/T86"
    params = {"date": date_str.replace("-", ""), "selectType": "ALL", "response": "json"}
    resp = session_get(url, params=params, timeout=15)
    if resp is None: return pd.DataFrame()
    try: data = resp.json()
    except ValueError: return pd.DataFrame()
    if data.get("stat") != "OK": return pd.DataFrame()

    df = pd.DataFrame(data.get("data", []), columns=data.get("fields", []))
    def find_col(kw_list):
        for kw in kw_list:
            for c in df.columns:
                if all(k in c for k in kw): return c
        return None

    col_id           = find_col([["證券代號"]])
    col_fb1, col_fs1 = find_col([["外陸資買進", "不含"]]),     find_col([["外陸資賣出", "不含"]])
    col_fb2, col_fs2 = find_col([["外資自營商買進"]]),          find_col([["外資自營商賣出"]])
    col_tb,  col_ts  = find_col([["投信買進"]]),                find_col([["投信賣出"]])
    col_db1, col_ds1 = find_col([["自營商買進", "自行"]]),      find_col([["自營商賣出", "自行"]])
    col_db2, col_ds2 = find_col([["自營商買進", "避險"]]),      find_col([["自營商賣出", "避險"]])

    records = []
    for _, r in df.iterrows():
        sid = str(r[col_id]).strip('="')
        if not sid.isdigit() or len(sid) != 4: continue
        records.extend([
            {"date": date_str, "stock_id": sid, "name": "Foreign_Investor",
             "buy":  to_int(r.get(col_fb1)) + to_int(r.get(col_fb2)),
             "sell": to_int(r.get(col_fs1)) + to_int(r.get(col_fs2))},
            {"date": date_str, "stock_id": sid, "name": "Investment_Trust",
             "buy":  to_int(r.get(col_tb)),
             "sell": to_int(r.get(col_ts))},
            {"date": date_str, "stock_id": sid, "name": "Dealer",
             "buy":  to_int(r.get(col_db1)) + to_int(r.get(col_db2)),
             "sell": to_int(r.get(col_ds1)) + to_int(r.get(col_ds2))},
        ])
    return pd.DataFrame(records)

def fetch_tpex_institutional(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    url = "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php"
    params = {"l": "zh-tw", "se": "EW", "t": "D",
              "d": f"{dt.year-1911}/{dt.month:02d}/{dt.day:02d}", "o": "json"}
    resp = session_get(url, params=params, timeout=15)
    if resp is None: return pd.DataFrame()
    try: data = resp.json()
    except ValueError: return pd.DataFrame()
    if data.get("stat", "").lower() != "ok": return pd.DataFrame()
    rows = data.get("tables", [])[0].get("data", []) if data.get("tables") else []

    # --- TPEx 結構守門員: 抽樣驗證硬編碼 index 是否仍對應正確欄位 ---
    if rows:
        passed, failed = _tpex_validate_inst(rows)
        if failed > passed:
            print(f"      !!! [TPEx 法人結構警告] {date_str}: 抽樣 {passed+failed} 筆,"
                  f"通過 {passed}、失敗 {failed}")
            sample = list(rows[0])[:20]
            print(f"         首筆樣本(前 20 欄): {sample}")
            print(f"         → 疑似 TPEx 改版/欄位 index 錯位,**拒絕寫入本日法人資料**")
            print(f"         → 請手動比對 https://www.tpex.org.tw 三大法人買賣表,確認後更新 fetch_tpex_institutional()")
            return pd.DataFrame()

    # TPEx 3itrade_hedge 欄位結構 (已修正):
    # [0]代號 [1]名稱
    # [2]外資(不含自營)買  [3]賣
    # [5]外資自營商買      [6]賣
    # [8]外資及陸資合計買  [9]賣   (略過不抓)
    # [11]投信買           [12]賣   ← 真正的投信在這裡！
    # [14]自營(自行)買     [15]賣
    # [17]自營(避險)買     [18]賣

    records = []
    for r in rows:
        if len(r) < 19: continue  # 至少需要到自營(避險)的欄位
        sid = str(r[0]).strip()
        if not sid.isdigit() or len(sid) != 4: continue

        records.extend([
            {"date": date_str, "stock_id": sid, "name": "Foreign_Investor",
             "buy":  to_int(r[2])  + to_int(r[5]),
             "sell": to_int(r[3])  + to_int(r[6])},

            {"date": date_str, "stock_id": sid, "name": "Investment_Trust",
             "buy":  to_int(r[11]),   # 投信買
             "sell": to_int(r[12])},  # 投信賣

            {"date": date_str, "stock_id": sid, "name": "Dealer",
             "buy":  to_int(r[14]) + to_int(r[17]),  # 自行買 + 避險買
             "sell": to_int(r[15]) + to_int(r[18])}, # 自行賣 + 避險賣
        ])
    return pd.DataFrame(records)

def fetch_twse_margin(date_str):
    url = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
    params = {"date": date_str.replace("-", ""), "selectType": "ALL", "response": "json"}
    resp = session_get(url, params=params, timeout=15)
    if resp is None: return pd.DataFrame()
    try: data = resp.json()
    except ValueError: return pd.DataFrame()
    rows = next((tbl["data"] for tbl in data.get("tables", [])
                 if tbl.get("data") and len(tbl["data"][0]) >= 15),
                data.get("data", []))
    records = []
    for r in rows:
        sid = str(r[0]).strip('="')
        if not sid.isdigit() or len(sid) != 4: continue
        records.append({
            "date": date_str, "stock_id": sid,
            "MarginPurchaseTodayBalance":     to_int(r[6]),
            "MarginPurchaseYesterdayBalance": to_int(r[5]),
            "ShortSaleTodayBalance":          to_int(r[12]),
            "ShortSaleYesterdayBalance":      to_int(r[11]),
        })
    return pd.DataFrame(records)

def fetch_tpex_margin(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    url = "https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php"
    params = {"l": "zh-tw", "d": f"{dt.year-1911}/{dt.month:02d}/{dt.day:02d}", "o": "json"}
    resp = session_get(url, params=params, timeout=15)
    if resp is None: return pd.DataFrame()
    try: data = resp.json()
    except ValueError: return pd.DataFrame()
    rows = data.get("tables", [])[0].get("data", []) if data.get("tables") else []

    # --- TPEx 結構守門員: 弱驗證 (融資/融券無「合計=細項加總」可硬驗) ---
    if rows:
        passed, failed = _tpex_validate_margin(rows)
        if failed > passed:
            print(f"      !!! [TPEx 融資結構警告] {date_str}: 抽樣 {passed+failed} 筆,"
                  f"通過 {passed}、失敗 {failed}")
            sample = list(rows[0])[:15]
            print(f"         首筆樣本(前 15 欄): {sample}")
            print(f"         → 疑似 TPEx 改版/欄位 index 錯位,**拒絕寫入本日融資資料**")
            print(f"         → 請手動比對 https://www.tpex.org.tw 融資融券餘額表,確認後更新 fetch_tpex_margin()")
            return pd.DataFrame()

    records = []
    for r in rows:
        if len(r) < 15: continue
        sid = str(r[0]).strip()
        if not sid.isdigit() or len(sid) != 4: continue
        records.append({
            "date": date_str, "stock_id": sid,
            "MarginPurchaseTodayBalance":     to_int(r[6]),
            "MarginPurchaseYesterdayBalance": to_int(r[2]),
            "ShortSaleTodayBalance":          to_int(r[14]),
            "ShortSaleYesterdayBalance":      to_int(r[10]),
        })
    return pd.DataFrame(records)

def batch_daily_public(name, fn_twse, fn_tpex, start_date, end_date, desc):
    dates_needed = trading_days(start_date, end_date)

    # 讀取最新舊快取，找出已有哪些日期（增量模式）
    prev_files = sorted(CACHE_DIR.glob(f"{name}_*.parquet"))
    existing_dates = set()
    prev_df = pd.DataFrame()
    # 修正：拿掉外層的 and not FORCE，確保 prev_df 永遠能被載入
    if prev_files:
        try:
            prev_df = pd.read_parquet(prev_files[-1])
            # 保留期截斷:只保留 CACHE_RETAIN_DAYS 內的歷史
            prev_df = _trim_by_retention(prev_df, label=name)
            if "date" in prev_df.columns and not FORCE:
                # 只有非 FORCE 模式才排除已抓過的日期；FORCE 模式則全部重抓
                existing_dates = set(prev_df["date"].unique())
        except Exception:
            pass

    dates_to_fetch = [d for d in dates_needed if d not in existing_dates]

    if not dates_to_fetch:
        print(f"[{name}] {desc}: 所有日期已有快取，略過")
        # 若今日檔名不存在，複製一份避免明天重觸發，再清理舊檔
        if not path_for(name).exists() and not prev_df.empty:
            prev_df.to_parquet(path_for(name))
            cleanup_old_cache(name)
        return

    print(f"[{name}] {desc}: 需補抓 {len(dates_to_fetch)} 天（共 {len(dates_needed)} 天，已有 {len(existing_dates)} 天）...")
    all_dfs = [prev_df] if not prev_df.empty else []
    consecutive_empty = 0

    for i, d in enumerate(dates_to_fetch, 1):
        df_twse = fn_twse(d)
        if df_twse.empty:
            consecutive_empty += 1
            print(f"   [{i}/{len(dates_to_fetch)}] {d} - (無資料,可能為假日或被限速) 連續空 {consecutive_empty}")
            if consecutive_empty >= MAX_DAY_FAILS:
                print(f"   !!! 連續 {MAX_DAY_FAILS} 個工作日 TWSE 無資料,中止 {desc} 抓取以保護 IP。")
                break
            time.sleep(TWSE_DELAY * 2)
            continue
        consecutive_empty = 0
        time.sleep(TWSE_DELAY)
        df_tpex = fn_tpex(d)
        time.sleep(TPEX_DELAY)
        # 先檢查是否部分失敗
        if not df_twse.empty and df_tpex.empty:
            print(f"   [警告] {d} 上市成功但上櫃失敗，判定為部分異常。拒絕寫入以觸發下次重抓。")
            continue
        elif df_twse.empty and not df_tpex.empty:
            print(f"   [警告] {d} 上櫃成功但上市失敗，判定為部分異常。拒絕寫入以觸發下次重抓。")
            continue
            
        # 都沒問題，才把資料塞進去
        all_dfs.extend([df for df in [df_twse, df_tpex] if not df.empty])
        print(f"   [{i}/{len(dates_to_fetch)}] {d} ✓")

    if all_dfs:
        dedup_cols = ["date", "stock_id", "name"] if "name" in all_dfs[0].columns else ["date", "stock_id"]
        combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=dedup_cols, keep="last")
        combined.to_parquet(path_for(name))
        print(f"   -> {desc} 快取更新完成（總 {len(combined):,} 筆）")
        cleanup_old_cache(name)   # 新檔已包含全部歷史,舊檔安全刪除
    else:
        print(f"   !!! {desc} 無任何資料")

# ==========================================
# [4] yfinance 日K線 (改回全量覆蓋，確保還原日線正確)
# ==========================================
def fetch_yfinance_daily(info_df, default_start_date, chunk_size=400):
    if not need_fetch("daily"):
        print("[daily] 已有今日快取,略過"); return
        
    print(f"[daily] 啟動 yfinance 分批下載 (全市場重新抓取 {default_start_date} 起的還原日線)...")
    
    ticker_map = {
        (f"{r.stock_id}.TWO" if str(r.type).lower() in ('tpex', 'otc') else f"{r.stock_id}.TW"): r.stock_id
        for _, r in info_df.iterrows()
    }
    tickers = list(ticker_map.keys())
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    
    all_frames = []
    consecutive_fail = 0
    MAX_YF_FAIL = 2
    
    for ci, chunk in enumerate(chunks, 1):
        print(f"   [{ci}/{len(chunks)}] 下載 {len(chunk)} 檔...")
        try:
            # 每天都抓完整的 180 天，確保 auto_adjust 的歷史還原價是最新的
            data = yf.download(chunk, start=default_start_date, auto_adjust=True,
                               threads=True, progress=False)
        except Exception as e:
            print(f"   !!! 第 {ci} 批失敗: {e}")
            consecutive_fail += 1
            if consecutive_fail >= MAX_YF_FAIL: break
            time.sleep(5); continue
            
        if data.empty:
            consecutive_fail += 1
            if consecutive_fail >= MAX_YF_FAIL: break
            time.sleep(5); continue
            
        consecutive_fail = 0
        if len(chunk) == 1:
            df_flat = data.reset_index()
            df_flat['Ticker'] = chunk[0]
        else:
            try:
                df_flat = data.stack(level=1, future_stack=True).reset_index()
            except TypeError:
                df_flat = data.stack(level=1).reset_index()

        # 標準化日期欄名:yfinance 不同版本可能產出 Date / Datetime / index / level_0
        # 若 reset_index 拿到無名 index,Pandas 會給它 "index" 這個欄名
        _date_aliases = ('Date', 'Datetime', 'index', 'level_0')
        _found_date_col = next((c for c in _date_aliases if c in df_flat.columns), None)
        if _found_date_col is None:
            print(f"   !!! 第 {ci} 批找不到日期欄,columns = {list(df_flat.columns)[:8]},略過")
            consecutive_fail += 1
            if consecutive_fail >= MAX_YF_FAIL: break
            continue
        if _found_date_col != 'Date':
            df_flat = df_flat.rename(columns={_found_date_col: 'Date'})

        all_frames.append(df_flat)
        time.sleep(2)

    if not all_frames:
        print("   !!! yfinance 本次抓取無資料。")
        # 失敗時才拿舊檔來墊檔
        prev_files = sorted(CACHE_DIR.glob("daily_*.parquet"))
        if prev_files and not path_for("daily").exists():
            pd.read_parquet(prev_files[-1]).to_parquet(path_for("daily"))
            cleanup_old_cache("daily")
        # 即使墊檔也寫時戳,讓 UI 知道「有人按過按鈕」,並用 FALLBACK 標記
        try:
            _now_str = datetime.now(TPE_TZ).strftime("%Y-%m-%d %H:%M:%S")
            (CACHE_DIR / "last_fetch_daily.txt").write_text(f"{_now_str} FALLBACK", encoding="utf-8")
            print(f"   -> 寫入抓取時戳(墊檔模式): {_now_str}")
        except Exception as e:
            print(f"   ⚠ 寫入時戳失敗(略過): {e}")
        return
        
    df_new = pd.concat(all_frames, ignore_index=True)
    df_new = df_new.rename(columns={
        "Date": "date", "Ticker": "yf_ticker",
        "Open": "open", "High": "max", "Low": "min", "Close": "close", "Volume": "Trading_Volume",
    })
    df_new['date']     = pd.to_datetime(df_new['date']).dt.strftime('%Y-%m-%d')
    df_new['stock_id'] = df_new['yf_ticker'].map(ticker_map)
    df_new = df_new.dropna(subset=['close', 'stock_id'])
    df_new = df_new.sort_values(by=['stock_id', 'date']).reset_index(drop=True)
    
    # 直接存檔，無需與舊檔合併
    df_new.to_parquet(path_for("daily"))
    print(f"   -> 價量快取全量建置完成! (總庫存: {len(df_new):,} 筆)")
    cleanup_old_cache("daily")  # <--- 補上這行

    # 寫入抓取時戳(TPE),供 UI 顯示「資料更新到幾點」── 用內容而非 mtime,避免 git pull 重置
    try:
        _now_str = datetime.now(TPE_TZ).strftime("%Y-%m-%d %H:%M:%S")
        (CACHE_DIR / "last_fetch_daily.txt").write_text(_now_str, encoding="utf-8")
        print(f"   -> 寫入抓取時戳: {_now_str} (台北時間)")
    except Exception as e:
        print(f"   ⚠ 寫入時戳失敗(略過): {e}")

# ==========================================
# [5] TDCC 千張大戶 (增量更新)
# ==========================================
def fetch_tdcc_holders_latest():
    if not need_fetch("holders"):
        print("[holders] 已有今日快取,略過"); return
    print("[holders] 抓取 TDCC 最新大戶比例...")
    resp = session_get("https://opendata.tdcc.com.tw/getOD.ashx", params={"id": "1-5"}, timeout=30)
    time.sleep(TDCC_DELAY)
    if resp is None:
        print("   !!! TDCC 請求失敗")
        _fallback_prev_to_today("holders")
        return
    try:
        df = pd.read_csv(io.StringIO(resp.content.decode("utf-8-sig", errors="ignore")))
    except Exception as e:
        print(f"   !!! TDCC CSV 解析失敗: {e}")
        _fallback_prev_to_today("holders")
        return

    col_map = {}
    for c in df.columns:
        cs = str(c)
        if   "日期" in cs: col_map[c] = "date_raw"
        elif "代號" in cs: col_map[c] = "stock_id"
        elif "分級" in cs: col_map[c] = "HoldingSharesLevel"
        elif "比例" in cs or "百分比" in cs: col_map[c] = "percent"
    df = df.rename(columns=col_map)
    # 防護：確認必要欄位都已對應
    required = {"date_raw", "stock_id", "HoldingSharesLevel", "percent"}
    missing = required - set(df.columns)
    if missing:
        print(f"   !!! TDCC 欄位對應失敗，缺少: {missing}，略過本次更新")
        _fallback_prev_to_today("holders")
        return

    df["date"]     = pd.to_datetime(df["date_raw"].astype(str), format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    df["stock_id"] = df["stock_id"].astype(str).str.strip()
    df = df[df["stock_id"].str.match(r"^\d{4}$", na=False)]
    df_tdcc = df[["date", "stock_id", "HoldingSharesLevel", "percent"]].copy()

    prev_files = sorted(CACHE_DIR.glob("holders_*.parquet"))
    if prev_files:
        try:
            df_prev = pd.read_parquet(prev_files[-1])
            # 保留期截斷:只留 CACHE_RETAIN_DAYS 內的週期 (TDCC 為週資料,90 天 ~12 週,
            # screening1 LARGE_HOLDER_WEEKS=4 已遠遠夠用)
            df_prev = _trim_by_retention(df_prev, label="holders")
            combined = pd.concat([df_prev, df_tdcc], ignore_index=True) \
                         .drop_duplicates(subset=["date", "stock_id", "HoldingSharesLevel"], keep="last")
        except Exception as e:
            print(f"   (讀取舊快取失敗,只存本次: {e})")
            combined = df_tdcc
    else:
        combined = df_tdcc

    combined.to_parquet(path_for("holders"))
    uniq_dates = sorted(combined["date"].dropna().unique())
    print(f"   -> 大戶比例快取建置完成 ({len(combined):,} 筆,{len(uniq_dates)} 期)")
    cleanup_old_cache("holders")   # 新檔已包含全部歷史週期,舊檔安全刪除

# ==========================================
# [6] TWSE / TPEx OpenAPI 月營收 (取代舊的 MOPS HTML 爬蟲)
# 端點:
#   - 上市: https://openapi.twse.com.tw/v1/opendata/t187ap05_L
#   - 上櫃: https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O
# 特性:
#   - 純 JSON,免 cookie 暖身/Referer,IP 風險遠低於 MOPS
#   - 每次只回「最新一期」(各上市櫃公司剛公告的當月營收)
#   - 主程式靠 drop_duplicates 自然累積成歷史資料庫
# ==========================================
def fetch_twse_openapi_revenue():
    """抓取 TWSE/TPEx OpenAPI 最新月營收,回 DataFrame[stock_id, revenue_year, revenue_month, revenue]"""
    endpoints = [
        ("TWSE", "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"),
        ("TPEx", "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O"),
    ]
    frames = []
    for name, url in endpoints:
        print(f"    [OpenAPI] 請求 {name} ...")
        resp = session_get(url, timeout=30)
        if resp is None:
            print(f"      !!! {name} 請求失敗,略過"); continue
        try:
            data = resp.json()
        except ValueError:
            print(f"      !!! {name} JSON 解析失敗,略過"); continue
        if not isinstance(data, list) or not data:
            print(f"      !!! {name} 回傳空或格式不符,略過"); continue

        df = pd.DataFrame(data)
        # 模糊匹配欄位 (應對未來改名)
        col_id  = next((c for c in df.columns if "公司代號" in str(c)), None)
        col_ym  = next((c for c in df.columns if "資料年月" in str(c)), None)
        col_rev = next((c for c in df.columns if "當月營收" in str(c)), None)
        if not (col_id and col_ym and col_rev):
            print(f"      !!! {name} 欄位不齊 (id={col_id} ym={col_ym} rev={col_rev}),略過"); continue

        # 4 碼股票代號過濾 (排除權證/ETF/特別股等非 4 碼)
        df = df[df[col_id].astype(str).str.match(r"^\d{4}$", na=False)].copy()
        # 解析「資料年月」民國格式 "11403" → 2025/3 (yyymm,前 3 碼=民國年,後 2 碼=月)
        ym       = df[col_ym].astype(str).str.strip()
        roc_year = pd.to_numeric(ym.str[:-2], errors='coerce')
        month    = pd.to_numeric(ym.str[-2:], errors='coerce')
        df['revenue_year']  = (roc_year + 1911)
        df['revenue_month'] = month
        df['stock_id']      = df[col_id].astype(str)
        # 營收清洗:去逗號、轉數值
        df['revenue'] = pd.to_numeric(
            df[col_rev].astype(str).str.replace(',', '', regex=False).str.strip(),
            errors='coerce'
        )
        df = df.dropna(subset=['revenue_year', 'revenue_month', 'revenue'])
        df['revenue_year']  = df['revenue_year'].astype(int)
        df['revenue_month'] = df['revenue_month'].astype(int)

        out = df[['stock_id', 'revenue_year', 'revenue_month', 'revenue']].copy()
        if not out.empty:
            ym_set = sorted(set(zip(out['revenue_year'], out['revenue_month'])))
            ym_str = ", ".join(f"{y}-{m:02d}" for y, m in ym_set)
            print(f"      ✓ {name} 取得 {len(out):,} 筆,涵蓋月份: {ym_str}")
            frames.append(out)
        time.sleep(2)  # 端點之間禮貌延遲

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ==========================================
# 啟動主程式
# ==========================================
print("=" * 65)
print(f">>> 官方直連版快取系統 v2 (含 IP 防護強化) | {today}")
if FORCE:
    print(">>> ⚠️  --force 模式:無視所有當日 cache,全部重抓")
elif FORCE_DAILY:
    print(">>> 🔄 --force-daily 模式:只強制重抓 daily K 線,其他資源沿用當日 cache")
print("=" * 65)

# [1] 股票清單
# info 為每日全量重抓（需反映最新上市/下市/更名），不做增量累積
if need_fetch("info"):
    print("[1] 抓取官方股票清單 (OpenAPI 優先,ISIN 備援)...", flush=True)
    info = fetch_public_stock_info()
    if info.empty:
        print("!!! 清單取得失敗"); sys.exit(1)
    info.to_parquet(path_for("info"))
    print(f"   -> 精煉出 {len(info)} 檔產業個股")
    cleanup_old_cache("info")   # 每日全量重建，舊清單直接刪除
else:
    info = pd.read_parquet(path_for("info"))
    print(f"[1] 股票清單已有快取: {len(info)} 檔")

# [2] 三大法人
batch_daily_public("institutional", fetch_twse_institutional, fetch_tpex_institutional, d20, today, "三大法人")

# [3] 融資券
batch_daily_public("margin", fetch_twse_margin, fetch_tpex_margin, d20, today, "融資融券")

# [4] 價量
fetch_yfinance_daily(info, d180)

# [5] 大戶
fetch_tdcc_holders_latest()

# [6] TWSE/TPEx OpenAPI 月營收 (每月檢查 + 自動累積至快取)
# 邏輯:
#   1. 讀取最新一份 revenue_*.parquet 作為歷史基底 (累積過去抓過的所有月份)
#   2. 呼叫 OpenAPI 取「目前最新一期」(MOPS 一般每月 10 日左右公告上月營收)
#   3. 用 (stock_id, revenue_year, revenue_month) 去重後合併,keep='last' 讓新值覆蓋舊值
#   4. 即使每天執行也只會新增/更新最新月份,不會重複抓歷史 → 純加法累積
if need_fetch("revenue"):
    print("[6] 月營收 (TWSE/TPEx OpenAPI): 抓取最新一期並累積至快取...")

    # --- (a) 讀歷史快取 ---
    prev_files = sorted(CACHE_DIR.glob("revenue_*.parquet"))
    df_prev = pd.DataFrame()
    if prev_files:
        try:
            df_prev = pd.read_parquet(prev_files[-1])
            if {'revenue_year', 'revenue_month'}.issubset(df_prev.columns) and not df_prev.empty:
                ym_set = sorted(set(zip(df_prev['revenue_year'], df_prev['revenue_month'])))
                print(f"    歷史快取: {len(df_prev):,} 筆,共 {len(ym_set)} 個月 "
                      f"({ym_set[0][0]}-{ym_set[0][1]:02d} ~ {ym_set[-1][0]}-{ym_set[-1][1]:02d})")
            else:
                print(f"    歷史快取: {len(df_prev):,} 筆 (欄位不完整,將以本次資料覆蓋)")
        except Exception as e:
            print(f"    (歷史快取讀取失敗: {e},本次將從空累積)")
            df_prev = pd.DataFrame()
    else:
        print("    無歷史快取,初次建檔 (本次只會有 OpenAPI 最新一期;後續每月執行自動累積)")

    # --- (b) 抓 OpenAPI 最新一期 ---
    df_latest = fetch_twse_openapi_revenue()

    # --- (c) 合併去重 + 寫回 ---
    rev_frames = [df for df in (df_prev, df_latest) if not df.empty]
    if rev_frames:
        combined = pd.concat(rev_frames, ignore_index=True) \
                     .drop_duplicates(subset=['stock_id', 'revenue_year', 'revenue_month'], keep='last')
        combined.to_parquet(path_for("revenue"))
        ym_set = sorted(set(zip(combined['revenue_year'], combined['revenue_month'])))
        print(f"   -> 營收資料庫累積完成!總筆數: {len(combined):,},共 {len(ym_set)} 個月")
        if ym_set:
            print(f"      最新月份: {ym_set[-1][0]}-{ym_set[-1][1]:02d} / "
                  f"最舊月份: {ym_set[0][0]}-{ym_set[0][1]:02d}")
        # 提醒使用者:若 screening.py 想用 YoY,需累積到 12 + REVENUE_MONTHS 個月
        if len(ym_set) < 15:
            print(f"      [提示] 目前僅 {len(ym_set)} 個月,YoY 需要 ≥15 個月;"
                  f"未滿前 screening.py 會自動降級用 MoM。")
        cleanup_old_cache("revenue")   # 新檔已包含全部歷史月份,舊檔安全刪除
    else:
        print("    !!! 無任何資料可寫入 (歷史空 + OpenAPI 失敗),保留現況。")
else:
    print("[revenue] 月營收: 已有今日快取,略過")

print("=" * 65)
print("✅ 快取建置完美收工!請接著執行 screening.py")
print("=" * 65)
