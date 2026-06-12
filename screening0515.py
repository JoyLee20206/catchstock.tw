import os, sys, warnings
from datetime import datetime
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# 篩選參數 (可自由調整,不耗 API)
# ==========================================
CACHE_DIR = Path("cache")
# ──────────────────────────────────────────────────────────────
# 計分總覽 (滿分 12):
#   1. 投信買超(1)     2. 外資買超(1)        3. 投信+外資雙買超(1)
#   4. 券相關合併(1)   5. 400張大戶上升(2)   6. 散戶下降(2)
#   7. 技術面三合一(1) 8. KD 低檔金叉(1)     9. 月營收達標(1)
#   10. RS vs 大盤(1) (僅多頭計分;空頭時記 0 — 見動態門檻調整)
#   ※ 大戶/散戶加重為 2 分:訊號歸因唯二明顯正 edge(+9.2%/+6.4%),
#     門檻 7 分下等於「無籌碼訊號就進不來」(其餘條件加總最高 6 分)
#
# 動態門檻調整:
#   - 大盤跌破季線 → PASS_SCORE 自動 +1 (空頭從嚴);且 RS 訊號不計分
#     (歸因顯示做頭往下時追強勢 edge 為負,故空頭關閉 RS,只靠籌碼/基本面)
#   - 大盤 ^TWII 抓取失敗 → PASS_SCORE 自動 -1 (RS 訊號失效補償)
#
# 籌碼共振 (大戶↑+散戶↓):「不再額外計分」,但作為排序優先序
# (大戶與散戶各自獨立 1 分;同時成立自然累積到 2 分)
# ──────────────────────────────────────────────────────────────
PASS_SCORE = 7             # 滿分 12 的過關門檻 (大盤空頭時自動 +1;RS 失效時自動 -1)
                           # 空頭門檻 8:基本盤 5(RS 關)+ 大戶 2 = 7 不夠 → 需大戶+散戶雙籌碼才過

# 法人 / 籌碼
LOOKBACK_DAYS = 5          # 投信/外資觀察天數
IT_MIN_BUY_DAYS = 3        # 投信至少幾日買超
FI_MIN_BUY_DAYS = 2        # 外資至少幾日買超 (略寬於投信:外資單日量大,但仍需穩定性,避免 1 日大買 4 日小賣就 flag)
# (自營商訊號已移除:短線避險居多、訊號最弱;改以「投信+外資雙買超」取代)

# 大戶/散戶 (TDCC 週資料)
LARGE_HOLDER_LEVELS = [12, 13, 14, 15]  # 400張以上 (TDCC HoldingSharesLevel:12=400,001~600,000股;13=600,001~800,000;14=800,001~1,000,000;15=1,000,001以上)
SMALL_HOLDER_LEVELS = [2, 3, 4]         # 1~15 張 (Level 1 為零股,雜訊大不納入)
LARGE_HOLDER_WEEKS = 4                  # 主路徑需要的最小週數 (4 週才能算「最近 3 週累計變化」)
LARGE_HOLDER_3W_CHANGE_MIN = 0.15       # 大戶近 3 週累計變化(%)門檻 (3 週累計過濾單週雜訊)
LARGE_HOLDER_CHANGE_MIN = 0.05          # Fallback:資料不足 4 週時用 1 週比較的舊門檻
SMALL_HOLDER_3W_CHANGE_MAX = -0.15      # 散戶近 3 週累計變化(%)門檻 (負值,越負代表跑越多)
SMALL_HOLDER_CHANGE_MAX = -0.05         # Fallback:資料不足 4 週時的 1 週比較

# 融資/融券
SHORT_MARGIN_RATIO = 0.2   # 券資比門檻: 融券餘額 / 融資餘額 > 此值 視為軋空潛力
# 註:「資減券增」訊號額外要求「現價 > MA20」(在計分迴圈合成),排除下跌段散戶斷頭+高檔放空的偽軋空

# 技術面
HIGH_BREAK_DAYS    = 60    # N 日新高 (60=近3月新高;若改 240=近52週新高,需同步把 fetch_cache 的 d180 改為 d400)
HIGH_TOLERANCE     = 0.995 # 容許 0.5% 誤差 (突破要嚴一點才有意義)
BREAKOUT_VOL_RATIO = 1.5   # 「量價齊揚突破」當日量倍數 (突破日量能要求嚴)
VOL_SURGE_RATIO    = 1.2   # 一般量增門檻 (近 N 日量 > MA20 × 此倍數)
VOL_SURGE_WINDOW   = 3     # 量增觀察視窗 (天)
VOL_SURGE_DAYS     = 2     # 視窗內需 ≥ 此日數量增 (3 日中 2 日量增,過濾單日噪音)

# 基本面
REVENUE_YOY_MIN    = 0     # 連 3 月 YoY 最低門檻(%),0=只要 3 月都為正
REVENUE_MONTHS     = 3     # 連續 N 個月年增達標
REVENUE_MOM_MIN    = 0     # MoM 降級門檻(%),0=只要連 N 月 MoM 為正
REVENUE_MOM_MONTHS = 3     # MoM 連續 N 個月達標 (路徑 B 需要連續 N 個月有效 MoM,不會被「免死金牌」稀釋)
SKIP_MOM_MONTHS = {2, 3}     # 春節因素:當 MoM 序列的「目標月」(win.iloc[1:].month) 含此集合中的月份時整段 MoM 停用
                             # (1→2 春節雪崩、2→3 春節後反彈,皆非趨勢訊號;1 月作為目標月對應 12→1,屬正常季節性,不跳)

# KD
# ──────────────────────────────────────────────────────────────
# KD 設計理念 (方向 A:配合「找低檔吃貨即將起漲」策略):
# 不是要求「現在 K 還在低檔」,而是要求「最近曾從低檔金叉啟動」
# 這樣才能跟技術面 (突破新高/量增) 同時觸發,讓真正的起漲股拿到 2 分
#
# 條件:
#   1. 今日 K > D (金叉狀態維持中,尚未死叉)
#   2. 過去 KD_LOOKBACK 天內,某日發生「昨 K ≤ 昨 D 且 今 K > 今 D」的交叉動作
#   3. 當天交叉時的 K 值 < KD_LOW_FROM (證明從低檔啟動)
#   4. 今日 K 值 < KD_HIGH_CAP_NOW (避免追已超買的高檔股)
# ──────────────────────────────────────────────────────────────
KD_N             = 9     # RSV 計算天數
KD_LOW_FROM      = 30    # 「低檔啟動」門檻:交叉當天 K 須 < 此值
KD_LOOKBACK      = 5    # 觀察區間:近 N 個交易日內是否發生低檔金叉
KD_HIGH_CAP_NOW  = 80    # 今日 K 上限:超過此值代表已超買,即使曾低檔金叉也不算

# 大盤趨勢 + RS (動態 PASS_SCORE + 計分項)
MARKET_INDEX_TICKER = "^TWII"  # 加權指數 (yfinance ticker)
MARKET_MA_DAYS      = 60       # 大盤季線判斷 (大盤跌破則 PASS_SCORE 自動 +1)
MARKET_MA_SHORT     = 20       # 大盤月線:站上季線但跌破月線 (或近 20 日下跌) → 視為「盤整/修正」
RS_LOOKBACK         = 20       # 個股 vs 大盤的 N 日漲幅比較 (RS 計分用)

# 預篩條件 (不計分,未達標直接剔除)
MIN_AVG_VOL_LOTS = 400     # 近 20 日均量下限 (張),過濾流動性不足
ATR_MAX_PCT      = 12.0     # ATR(14) / 現價 上限(%),過濾波動過大的飆股

# 恐慌煞車 (接止跌判讀 bottom_signal 的排程結果)
# 歸因實證:崩盤期照常選股,入選股後 5 日平均 -6.89% —— 此階段選什麼都賠。
# 大盤破季線(空頭從嚴)是落後指標,急跌初段擋不住;VIXTWN 閘門反應快得多。
PANIC_GUARD_LEVEL_STOP   = "高度恐慌"  # 止跌判讀為此分級 → 當日暫停推薦新股
PANIC_GUARD_MAX_AGE_DAYS = 4           # latest.json 超過 N 天未更新 → 視為未知,不啟用煞車

# ==========================================

def latest(name, required=True):
    """讀取最新快取,required=False 時找不到回傳空 DataFrame"""
    files = sorted(CACHE_DIR.glob(f"{name}_*.parquet"))
    if not files:
        if required:
            print(f"!!! 致命錯誤:找不到必備的 {name} 快取,請先執行 fetch_cache.py"); sys.exit(1)
        print(f"   [警告] 找不到選配的 {name} 快取,該項條件將以 0 分計算。")
        return pd.DataFrame()
    return pd.read_parquet(files[-1])

def _load_bottom_panic_level(max_age_days=PANIC_GUARD_MAX_AGE_DAYS):
    """讀止跌判讀排程結果(cache/bottom_signal_latest.json),回 (level_label, age_days)。

    bottom_push 排程每天 16:10/21:30 更新此檔。檔案不存在/壞檔/超過
    max_age_days 天未更新(排程斷線)→ 回 (None, age):煞車不啟用,選股照常,
    寧可放行也不要因為一個輔助檔壞掉就永遠停止選股。
    """
    import json
    try:
        f = CACHE_DIR / "bottom_signal_latest.json"
        if not f.exists():
            return None, None
        data = json.loads(f.read_text(encoding="utf-8"))
        gen = str(data.get("generated_at", ""))[:10]
        age = (datetime.now().date() - datetime.strptime(gen, "%Y-%m-%d").date()).days
        if age > max_age_days:
            print(f"   [恐慌煞車] 止跌判讀檔已 {age} 天未更新,視為未知、不啟用煞車")
            return None, age
        return data.get("level_label"), age
    except Exception as e:
        print(f"   [恐慌煞車] 止跌判讀檔讀取失敗(不啟用): {str(e)[:80]}")
        return None, None


def _calc_kd_series(highs, lows, closes, n=9):
    """Taiwan 常用 KD: K(t)=(2*K(t-1)+RSV(t))/3, D(t)=(2*D(t-1)+K(t))/3, 初值 50
    回傳完整的 K、D 序列 (從第 n-1 個 bar 開始有值,前面填 None)
    這樣可以掃描歷史找「曾在低檔金叉」的點"""
    if len(closes) < n + 1:
        return None, None
    k, d = 50.0, 50.0
    k_list = [None] * (n - 1)
    d_list = [None] * (n - 1)
    for i in range(n - 1, len(closes)):
        hh = highs.iloc[i - n + 1:i + 1].max()
        ll = lows.iloc[i - n + 1:i + 1].min()
        rsv = 50.0 if hh == ll else (closes.iloc[i] - ll) / (hh - ll) * 100
        k = (2 * k + rsv) / 3
        d = (2 * d + k) / 3
        k_list.append(k)
        d_list.append(d)
    return k_list, d_list

def _norm_stock_code(c):
    """標準化股票代號:
    - 純數字: 補回 4 位 leading zero (int 50 → "0050", str "0050" 原樣)
    - 含字母 (特別股 2881A): 原樣返回
    - 6 位數海外股 (910482) / 5 位 ETF (00878): zfill 不會截短,保留原長度
    write_xq_dsl 和 write_precision_xls 都用同一份,避免兩處規則不同步 (Bug #3 修正)
    """
    s = str(c).strip()
    return s.zfill(4) if s.isdigit() else s


def write_xq_dsl(stock_codes, output_path, template_path='自選二.dsl'):
    """
    產生嘉實XQ金好康可匯入的 .dsl 自選股檔。
    格式為 OLE2 compound document，stream 內容：
        1,XQSYSLIST2;[Big5群組名],[XXXX.TW,XXXX.TW,...]
    以 自選二.dsl 為模板，直接 patch mini-FAT / stream size。
    單檔上限約 45 檔；超過時自動分批產出 _part1.dsl, _part2.dsl, ...
    """
    import struct
    tpl = Path(template_path)
    if not tpl.exists():
        print(f"⚠ 找不到 {template_path}，略過 .dsl 輸出")
        return

    with open(tpl, 'rb') as f:
        template_data = bytes(f.read())

    # 以下位移均由 自選二.dsl 逆向確認
    MINIFAT_OFF     = 1536   # mini-FAT sector (sector 2)
    ROOT_SIZE_OFF   = 1144   # Root Entry 目錄項 size 欄位
    STREAM_SIZE_OFF = 1528   # FileContentSymbolList_0 目錄項 size 欄位
    CONTENT_OFF     = 2176   # FileContentSymbolList_0 資料起始 (mini-sector 2)
    MS0             = 2      # FileContentSymbolList_0 從 mini-sector 2 開始
    MAX_MS          = 6      # sector 3 最多容納 8 個 ms，扣掉 ms0/1 剩 6 個
    PREFIX          = b'1,XQSYSLIST2;'
    END_OF_CHAIN    = 0xFFFFFFFE
    FREESECT        = 0xFFFFFFFF

    # === 模板自檢:從 mini-FAT 動態讀取 cur_ms,並驗證鏈結構正常 ===
    # 不再硬寫 cur_ms = 3,模板若被改過(例如重存後 mini-sector 配置不同)能即時抓到
    cur_ms = 0
    cursor = MS0
    while True:
        if cur_ms >= MAX_MS:
            print(f"⚠ {template_path} mini-stream 鏈長度 ≥ {MAX_MS},超出支援範圍,略過 .dsl 輸出")
            return
        entry = struct.unpack_from('<I', template_data, MINIFAT_OFF + cursor * 4)[0]
        cur_ms += 1
        if entry == END_OF_CHAIN:
            break
        if entry == FREESECT or entry < MS0 or entry >= MS0 + MAX_MS:
            print(f"⚠ {template_path} mini-FAT 鏈異常 (cursor={cursor}, entry={entry:#x}),"
                  f"模板可能已損壞或非預期版本,略過 .dsl 輸出")
            return
        cursor = entry

    # === 動態抽出 Big5 群組名:從 "1,XQSYSLIST2;" 結尾掃到下一個逗號 ===
    # 不再寫死「6 byte」,避免模板群組名長度不同 (改用 2 或 4 字 Big5) 時靜默產出壞檔
    name_start = CONTENT_OFF + len(PREFIX)
    name_end_limit = CONTENT_OFF + cur_ms * 64
    comma_idx = template_data.find(b',', name_start, name_end_limit)
    if comma_idx < 0 or comma_idx == name_start:
        print(f"⚠ {template_path} 找不到群組名後的分隔逗號 (或群組名為空),略過 .dsl 輸出")
        return
    LIST_NAME_B5 = template_data[name_start:comma_idx]

    # === 計算每批最多能塞多少檔 ===
    header_len = len(PREFIX) + len(LIST_NAME_B5) + 1   # +1 是群組名後的逗號
    max_payload = MAX_MS * 64 - header_len

    # 多批時群組名要加 "_N" 後綴避免 XQ 匯入後同名覆蓋;預留 4 byte (支援到 _999)
    # ASCII 數字 + 底線屬於 Big5 相容字元,直接 append 安全
    SUFFIX_RESERVE = 4

    def _encode_batch(codes, suffix_len=0):
        """編碼一批代號為 ascii bytes;塞不下回傳 None。suffix_len 為群組名後綴預留位"""
        body = ','.join(f"{_norm_stock_code(c)}.TW" for c in codes).encode('ascii')
        return body if len(body) <= max_payload - suffix_len else None

    # === 分批策略 ===
    # 先嘗試「不分批」全部塞 1 個檔 (此時群組名不必加後綴);裝不下才走多批模式
    # 這樣 ≤ ~45 檔的常見情境保持原行為:單一檔 + 原始群組名
    stock_codes = list(stock_codes)
    if _encode_batch(stock_codes) is not None:
        batches = [stock_codes]
    else:
        # 多批模式:每批貪婪用二分法塞到最滿,但要為 "_N" 後綴預留容量
        batches = []
        remaining = stock_codes
        while remaining:
            lo, hi, best = 1, len(remaining), 0
            while lo <= hi:
                mid = (lo + hi) // 2
                if _encode_batch(remaining[:mid], SUFFIX_RESERVE) is not None:
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            if best == 0:
                # 單一代號就爆 (理論上不會,XXXX.TW 才 7~8 byte),保險處理
                print(f"⚠ 代號 {remaining[0]} 編碼後超過單檔容量,略過剩餘 {len(remaining)} 檔")
                break
            batches.append(remaining[:best])
            remaining = remaining[best:]

    if not batches:
        return

    # === 逐批寫出 ===
    base = Path(output_path)
    out_paths = []
    multi = len(batches) > 1
    for idx, batch in enumerate(batches):
        data = bytearray(template_data)
        body = _encode_batch(batch, SUFFIX_RESERVE if multi else 0)
        # 多批時群組名加 "_N" 後綴 (e.g. "自選二_1"、"自選二_2"),避免匯入 XQ 後同名互覆蓋
        list_name = LIST_NAME_B5 + f"_{idx + 1}".encode('ascii') if multi else LIST_NAME_B5
        content = PREFIX + list_name + b',' + body
        new_size = len(content)
        need_ms = (new_size + 63) // 64

        # 更新 mini-FAT 鏈
        if need_ms > cur_ms:
            for i in range(cur_ms, need_ms):
                struct.pack_into('<I', data, MINIFAT_OFF + (MS0 + i - 1) * 4, MS0 + i)
                struct.pack_into('<I', data, MINIFAT_OFF + (MS0 + i) * 4, END_OF_CHAIN)
        elif need_ms < cur_ms:
            struct.pack_into('<I', data, MINIFAT_OFF + (MS0 + need_ms - 1) * 4, END_OF_CHAIN)
            for i in range(need_ms, cur_ms):
                struct.pack_into('<I', data, MINIFAT_OFF + (MS0 + i) * 4, FREESECT)

        struct.pack_into('<I', data, ROOT_SIZE_OFF,   (2 + need_ms) * 64)
        struct.pack_into('<I', data, STREAM_SIZE_OFF, new_size)

        allocated = need_ms * 64
        data[CONTENT_OFF: CONTENT_OFF + allocated] = content + b'\x00' * (allocated - new_size)
        if need_ms < cur_ms:
            freed_start = CONTENT_OFF + allocated
            data[freed_start: freed_start + (cur_ms - need_ms) * 64] = b'\x00' * ((cur_ms - need_ms) * 64)

        # 單批用原檔名;多批就加 _partN 後綴
        if len(batches) == 1:
            path = base
        else:
            path = base.with_name(f"{base.stem}_part{idx + 1}{base.suffix}")
        with open(path, 'wb') as f:
            f.write(data)
        out_paths.append((path, len(batch)))

    for p, n in out_paths:
        print(f"📋 .dsl 匯入檔：{os.path.abspath(p)}  ({n} 檔)")
    if len(out_paths) > 1:
        total = sum(n for _, n in out_paths)
        print(f"   共拆成 {len(out_paths)} 個檔案,合計 {total} 檔 (超過單檔上限自動分批)")
        print(f"   群組名已自動加 _1 / _2 / ... 後綴,避免 XQ 匯入後同名互覆蓋")


def write_precision_xls(stock_codes, name_map, output_path):
    """
    產生精誠「HiStock」可匯入的 .xls 自選股檔。
    格式：第一列 header (商品名稱 / 代碼)，之後每列一檔。
    代碼欄以數值格式寫入 (和原始檔一致)；ETF / 特別股等非純數字代號則以字串寫入。
    依賴：xlwt (pip install xlwt)
    """
    try:
        import xlwt
    except ImportError:
        print("⚠ 找不到 xlwt，略過精誠 .xls 輸出 (pip install xlwt)")
        return

    wb  = xlwt.Workbook(encoding='utf-8')
    ws  = wb.add_sheet('Sheet1')

    # Header
    ws.write(0, 0, '商品名稱')
    ws.write(0, 1, '代碼')

    for row_idx, code in enumerate(stock_codes, start=1):
        norm = _norm_stock_code(code)
        name = name_map.get(code, name_map.get(norm, ''))
        ws.write(row_idx, 0, name)
        # 純數字代號以整數寫入 (與原始檔格式一致,康和讀數值欄會自動補 4 位);
        # 英數混合 (特別股 2881A 等) 保持字串
        if norm.isdigit():
            ws.write(row_idx, 1, int(norm))
        else:
            ws.write(row_idx, 1, norm)

    # Bug #2 修正:檔案被開啟時 wb.save 會擲 PermissionError;
    # 此呼叫在 xlsx/dsl 都產出完才執行,直接 crash 會讓使用者誤以為前面的輸出也壞了
    try:
        wb.save(output_path)
    except PermissionError:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = Path(output_path)
        output_path = str(base.with_name(f"{base.stem}_{stamp}{base.suffix}"))
        wb.save(output_path)
        print(f"   ⚠ 原檔被開啟中,改存為 {output_path}")
    print(f"📋 精誠自選股匯入檔：{os.path.abspath(output_path)}  ({len(stock_codes)} 檔)")


def _calc_atr(highs, lows, closes, n=14):
    if len(closes) < n + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs.iloc[i] - lows.iloc[i],
            abs(highs.iloc[i] - closes.iloc[i - 1]),
            abs(lows.iloc[i] - closes.iloc[i - 1])
        )
        trs.append(tr)
    tr_s = pd.Series(trs)
    return float(tr_s.tail(n).mean())

# MoM 月份連續性檢查的小工具 (放在迴圈外,避免每股重建 function object)
def _next_ym(y, m):
    return (y, m + 1) if m < 12 else (y + 1, 1)


PRESETS = {
    'default':  {},  # all defaults
    'bull':     dict(pass_score=6, kd_lookback=10, kd_low_from=40, kd_high_cap_now=85),
    'bear':     dict(pass_score=8, kd_lookback=3, kd_low_from=25, kd_high_cap_now=75, min_avg_vol_lots=500),
    # 低檔發動(取代舊「KD 起漲」):找「KD 從更深超賣(<25)剛翻揚、且今日仍在低檔(<55,還沒漲上去)」的股。
    # 門檻 6(原本 5 太鬆 → 選太多;光籌碼大戶↑+散戶↓+券就 3 分,隨便湊就過 5)。
    # 流動性提高到 500 張,濾掉冷門股、再減量。偏「抄底/跌深翻揚」,與「追強勢」組合互補;務必先回測驗證 edge。
    'low_launch': dict(pass_score=6, kd_lookback=5, kd_low_from=25, kd_high_cap_now=55, min_avg_vol_lots=500),
}


def run_screening(
    pass_score=PASS_SCORE,
    lookback_days=LOOKBACK_DAYS,
    it_min_buy_days=IT_MIN_BUY_DAYS,
    fi_min_buy_days=FI_MIN_BUY_DAYS,
    kd_lookback=KD_LOOKBACK,
    kd_low_from=KD_LOW_FROM,
    kd_high_cap_now=KD_HIGH_CAP_NOW,
    min_avg_vol_lots=MIN_AVG_VOL_LOTS,
    atr_max_pct=ATR_MAX_PCT,
    output_dir=None,
):
    """執行選股流程。
    output_dir=None → 當前目錄 (CLI 模式);傳 Path → 指定目錄 (UI 模式用 tempdir)。
    回傳 (df, file_paths) 其中 file_paths 為 {'xlsx': Path, 'dsl': Path, 'xls': Path}。
    若無標的達標,df 為空 DataFrame,file_paths 為 {}。
    """
    PASS_SCORE = pass_score
    LOOKBACK_DAYS = lookback_days
    IT_MIN_BUY_DAYS = it_min_buy_days
    FI_MIN_BUY_DAYS = fi_min_buy_days
    KD_LOOKBACK = kd_lookback
    KD_LOW_FROM = kd_low_from
    KD_HIGH_CAP_NOW = kd_high_cap_now
    MIN_AVG_VOL_LOTS = min_avg_vol_lots
    ATR_MAX_PCT = atr_max_pct
    output_dir = Path(output_dir) if output_dir else Path('.')

    print("=" * 60)
    print(">>> 本地計分選股 (滿分 12;法人 3 / 籌碼 5[大戶、散戶各 2] / 技術 2 / 基本 1 / 大盤 1)")
    print(">>> 預篩 (不計分): 日均量 < {} 張 或 ATR% > {}% 直接剔除".format(MIN_AVG_VOL_LOTS, ATR_MAX_PCT))
    print("=" * 60)
    print(">>> 讀取快取...")
    info       = latest("info", required=True)
    inst_df    = latest("institutional", required=True)
    margin_df  = latest("margin", required=True)
    daily_df   = latest("daily", required=True)
    # 'date' 是 RS 對齊、TWII 裁切、KD/技術面排序都仰賴的欄位;缺了就讓問題提前噴出而非整段 RS 靜默歸零
    assert 'date' in daily_df.columns, "daily 快取缺 'date' 欄位,請檢查 fetch_cache 輸出格式"
    # 效能優化:預先把 date 整欄轉成 datetime,避免後面 groupby 每組再呼叫 pd.to_datetime 一次
    # 用 ~2000 檔 × 250 天估計,可省下約 2000 次重複解析。同時讓後續 sort_values('date') 走數值比較,
    # 比字串字典序更快、也更不會踩到 "12" < "2" 這類字典序陷阱
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    holders_df = latest("holders", required=False)
    revenue_df = latest("revenue", required=False)

    # Bug #2 修正:info 若不慎含重複 stock_id (例如上市/上櫃資料合併時撞號),
    # 同一檔會在迴圈處理兩次,Excel 與 .dsl 都出現重複。drop_duplicates 統一以「最後一筆」為準,
    # 與 set_index().to_dict() 對重複 key 的行為一致 (避免 dict 取 last、list 取 first 兩邊不同步)。
    info     = info.drop_duplicates(subset='stock_id', keep='last')
    name_map = info.set_index('stock_id')['stock_name'].to_dict()
    ind_map  = info.set_index('stock_id')['industry_category'].to_dict()
    all_ids  = info['stock_id'].tolist()

    # --- 大盤資料 (^TWII): 用於 RS 計分 + 大盤趨勢動態 PASS_SCORE ---
    # 失敗時 fallback 為「跳過」(market_bullish=True、twii_lookback_change=0),不會中斷選股
    print(">>> 抓取大盤指數 (^TWII) 用於 RS 與趨勢過濾...")
    market_bullish        = True
    market_consolidating  = False   # 站上季線但跌破月線/近 20 日下跌 → 盤整修正 (RS 關閉 + 籌碼硬門票)
    twii_lookback_change  = 0.0
    market_data_ok        = False
    twii_close            = None    # Bug #3 修正:提到 try 外讓個股 RS 迴圈能用,做日期對齊查表
    twii_now              = None    # UI 用:meta 回傳的「大盤狀態橫幅」需要
    twii_ma               = None
    cache_max_date        = daily_df["date"].max() if not daily_df.empty else None  # UI 用:cache 新鮮度
    try:
        # ── 路徑 A:本地 parquet (最快,由 fetch_cache.py 排程寫入) ──
        # 避免冷啟動每次打 yfinance ~3 秒,且 GA cloud IP 偶會被擋
        twii = pd.DataFrame()
        twii_files = sorted(CACHE_DIR.glob("twii_*.parquet"))
        if twii_files:
            try:
                _df_local = pd.read_parquet(twii_files[-1])
                if not _df_local.empty:
                    # parquet 是 fetch_cache.py 寫的格式:date 欄位 + Close 等
                    if "date" in _df_local.columns:
                        _df_local = _df_local.set_index(
                            pd.to_datetime(_df_local["date"])
                        ).drop(columns=["date"], errors="ignore")
                    twii = _df_local
                    print(f"   ✓ 大盤資料來源:本地 parquet ({twii_files[-1].name}, {len(twii)} 筆)")
            except Exception as _le:
                print(f"   ⚠ 讀本地 twii parquet 失敗,fallback yfinance: {_le}")
                twii = pd.DataFrame()

        # ── 路徑 B:沒 parquet 才打 yfinance ──
        if twii.empty:
            print("   (本地無 twii parquet,改打 yfinance)")
            import yfinance as yf
            twii = yf.download(MARKET_INDEX_TICKER, period="180d", auto_adjust=True,
                               progress=False, threads=False)
        # 防呆:yfinance ≥ 0.2.5x 對單一 ticker 也回傳 MultiIndex 欄位 → 先展平再判斷
        if isinstance(twii.columns, pd.MultiIndex):
            twii.columns = twii.columns.get_level_values(0)
        # 對齊:盤中跑時 ^TWII 含當日盤中價,但個股 daily 還是昨日收盤,RS 計算會錯位 1 天
        # → 把 ^TWII 裁切到 daily 快取的 max_date,確保 RS 兩邊基準日一致
        # ('date' 欄位存在性已在載入後 assert 保證,daily_df['date'] 也已轉成 datetime)
        if not twii.empty and not daily_df.empty:
            # Bug #1 修正:yfinance 不同版本對單一 ticker 可能回傳 tz-naive 或 tz-aware index
            # 對已是 tz-naive 的呼叫 tz_localize(None) 會擲 TypeError → 整段被外層 except 吞掉,
            # 使用者只會看到「抓取失敗」而不知是 tz 程式錯,RS 訊號靜默失效
            if twii.index.tz is not None:
                twii.index = twii.index.tz_localize(None)
            daily_max_date = daily_df["date"].max()
            twii = twii[twii.index <= daily_max_date]
        twii_candidate = None
        if not twii.empty and 'Close' in twii.columns and len(twii) >= MARKET_MA_DAYS:
            twii_candidate = twii['Close']
            if isinstance(twii_candidate, pd.DataFrame):
                twii_candidate = twii_candidate.squeeze()
            # Bug C 修正：先過濾 NaN，避免 float(NaN) 污染 market_bullish 判斷
            twii_candidate = twii_candidate.sort_index().dropna()
            if len(twii_candidate) < MARKET_MA_DAYS:
                print("   [警告] ^TWII 去除 NaN 後資料不足,跳過大盤過濾與 RS 計分")
                twii_candidate = None

        if twii_candidate is not None:
            twii_close = twii_candidate
            twii_now   = float(twii_close.iloc[-1])
            twii_ma    = float(twii_close.tail(MARKET_MA_DAYS).mean())
            market_bullish = twii_now > twii_ma
            if len(twii_close) > RS_LOOKBACK:
                ref = twii_close.iloc[-RS_LOOKBACK - 1]
                if pd.notna(ref) and ref > 0:
                    twii_lookback_change = float(
                        (twii_close.iloc[-1] / ref - 1) * 100
                    )
            # 盤整/修正判斷:站上季線 (非空頭) 但「跌破月線」或「近 20 日大盤下跌」。
            # 動機:5 月中~6 月初大盤全程站上季線,二分法把修正盤整當多頭跑,
            # RS 持續計分 → 專挑漲多補跌股 (歸因 RS edge -4.59%)。盤整時 RS 比照空頭關閉。
            twii_ma_short = float(twii_close.tail(MARKET_MA_SHORT).mean())
            market_consolidating = market_bullish and (
                twii_now < twii_ma_short or twii_lookback_change < 0
            )
            market_data_ok = True
            if market_bullish and market_consolidating:
                state_txt = "盤整修正(站上季線但跌破月線/近20日下跌)"
            elif market_bullish:
                state_txt = "多頭(站上季線)"
            else:
                state_txt = "空頭(跌破季線)"
            print(f"   大盤 {twii_now:.0f} / MA{MARKET_MA_DAYS} {twii_ma:.0f} / "
                  f"MA{MARKET_MA_SHORT} {twii_ma_short:.0f} → {state_txt};"
                  f"近 {RS_LOOKBACK} 日漲幅 {twii_lookback_change:+.2f}% (大盤最末日基準)")
        else:
            print("   [警告] ^TWII 資料不足,跳過大盤過濾與 RS 計分")
    except Exception as e:
        print(f"   [警告] ^TWII 抓取失敗 ({e}),跳過大盤過濾;PASS_SCORE 不調整、RS 訊號全部記 0")

    # 動態 PASS_SCORE:
    #   - 大盤跌破季線 → +1 (空頭從嚴)
    #   - 大盤資料失效 → -1 (RS 訊號全部失效,門檻補償;否則等於滿分變 9 卻仍要求 6)
    # 用獨立變數避免污染原 PASS_SCORE 常數 (Jupyter 重複執行時直接 += 會持續累加)
    effective_pass_score = PASS_SCORE
    if not market_data_ok:
        effective_pass_score = max(1, effective_pass_score - 1)
        print(f"   [RS 失效補償] 大盤資料缺,門檻從 {PASS_SCORE} 降至 {effective_pass_score}")
    elif not market_bullish:
        effective_pass_score = effective_pass_score + 1
        print(f"   [空頭從嚴] 大盤破季線,門檻從 {PASS_SCORE} 提高至 {effective_pass_score}")
    elif market_consolidating:
        # 盤整不動門檻:RS 關閉本身已少 1 分可拿,再加籌碼硬門票 (大戶↑/散戶↓至少其一),
        # 若門檻再 +1 等於三重緊縮,極易選不到股。
        print(f"   [盤整慎選] 大盤站上季線但跌破月線/近20日下跌:RS 不計分、"
              f"需「大戶↑或散戶↓」至少其一才入選;門檻維持 {effective_pass_score}")
    else:
        print(f"   [多頭] 大盤站上季線,門檻維持 {effective_pass_score}")

    # ── 恐慌煞車:止跌判讀「高度恐慌」→ 當日暫停推薦新股 ─────────────
    panic_level, _panic_age = _load_bottom_panic_level()
    if panic_level == PANIC_GUARD_LEVEL_STOP:
        print(f"🛑 [恐慌煞車] 止跌判讀分級「{panic_level}」:今日暫停推薦新股,"
              f"待恐慌降溫(🟡 剛降溫以下)自動恢復。")
        twii_pct = 0.0
        bias_ma60 = 0.0
        if market_data_ok and twii_close is not None and len(twii_close) >= 2:
            c_now, c_prev = twii_close.iloc[-1], twii_close.iloc[-2]
            twii_pct = (c_now - c_prev) / c_prev * 100
            if twii_ma and twii_ma > 0:
                bias_ma60 = (c_now - twii_ma) / twii_ma * 100
        meta = {
            'market_data_ok':       market_data_ok,
            'market_bullish':       market_bullish,
            'market_consolidating': market_consolidating,
            'market_state':         'bear' if not market_bullish else
                                    ('consolidation' if market_consolidating else 'bull'),
            'market_status':        "恐慌觀望",
            'twii_now':             twii_now,
            'twii_ma':              twii_ma,
            'twii_pct':             twii_pct,
            'twii_bias':            bias_ma60,
            'score_note':           "🛑 止跌判讀:高度恐慌——今日暫停推薦新股,待恐慌降溫自動恢復",
            'base_pass_score':      pass_score,
            'effective_pass_score': effective_pass_score,
            'cache_max_date':       cache_max_date,
            'twii_lookback_change': twii_lookback_change,
            'panic_guard':          True,
        }
        return pd.DataFrame(), {}, meta

    # --- 訊號 1: 投信買超 ---
    print(">>> [1/9] 投信買超 (近 {} 日總淨額 > 0 且 ≥ {} 日買超)...".format(LOOKBACK_DAYS, IT_MIN_BUY_DAYS))
    it_sig = {}
    if not inst_df.empty:
        it_df = inst_df[inst_df['name'].str.contains('Investment_Trust|投信', na=False)].copy()
        it_df['net'] = it_df['buy'] - it_df['sell']
        for sid, g in it_df.groupby('stock_id'):
            g = g.sort_values('date').tail(LOOKBACK_DAYS)
            total = g['net'].sum()
            days = (g['net'] > 0).sum()
            it_sig[sid] = {'flag': int(total > 0 and days >= IT_MIN_BUY_DAYS), 'net': int(total)}

    # --- 訊號 2: 外資買超 ---
    print(">>> [2/9] 外資買超 (近 {} 日總淨額 > 0 且 ≥ {} 日買超)...".format(LOOKBACK_DAYS, FI_MIN_BUY_DAYS))
    fi_sig = {}
    if not inst_df.empty:
        fi_df = inst_df[inst_df['name'].str.contains('Foreign_Investor|外資', na=False) &
                        ~inst_df['name'].str.contains('Dealer', na=False)].copy()
        fi_df['net'] = fi_df['buy'] - fi_df['sell']
        for sid, g in fi_df.groupby('stock_id'):
            g = g.sort_values('date').tail(LOOKBACK_DAYS)
            total = g['net'].sum()
            days  = (g['net'] > 0).sum()
            fi_sig[sid] = {'flag': int(total > 0 and days >= FI_MIN_BUY_DAYS), 'net': int(total)}

    # 訊號 3「投信+外資雙買超」會在計分迴圈合成,不需另外計算

    # --- 訊號 4: 券相關合併 (資減券增 + 趨勢過濾 OR 券資比軋空潛力) ---
    print(">>> [3/9] 券相關 (資減券增[+趨勢] OR 券資比>{}%)...".format(int(SHORT_MARGIN_RATIO*100)))
    margin_raw_sig = {}    # 「資減券增」原始訊號;最終訊號需配合「現價 > MA20」過濾 (在計分迴圈合成)
    sm_ratio_sig = {}
    sm_ratio_map = {}
    if not margin_df.empty:
        for sid, g in margin_df.groupby('stock_id'):
            g = g.sort_values('date')
            if len(g) < 2: continue
            m_today = g['MarginPurchaseTodayBalance'].iloc[-1]
            m_prev  = g['MarginPurchaseTodayBalance'].iloc[-2]
            s_today = g['ShortSaleTodayBalance'].iloc[-1]
            s_prev  = g['ShortSaleTodayBalance'].iloc[-2]

            # NaN 防呆:任一筆缺值時,該訊號判定中性 0,避免 NaN < NaN 默默回 False 變成「悄悄記 0」
            if not all(pd.notna(x) for x in (m_today, m_prev, s_today, s_prev)):
                continue

            margin_raw_sig[sid] = int(m_today < m_prev and s_today > s_prev)

            if m_today > 0:
                ratio = s_today / m_today
                sm_ratio_sig[sid] = int(ratio > SHORT_MARGIN_RATIO)
                sm_ratio_map[sid] = round(ratio * 100, 1)

    # --- 訊號 5+6: 400張大戶 + 散戶下降 (各自獨立計分;共振僅作排序優先序) ---
    print(f">>> [4/9] 400張大戶近 {LARGE_HOLDER_WEEKS-1} 週累計 ≥ {LARGE_HOLDER_3W_CHANGE_MIN}% (1 分)")
    print(f">>> [5/9] 散戶 1~15 張近 {LARGE_HOLDER_WEEKS-1} 週累計變化 ≤ {SMALL_HOLDER_3W_CHANGE_MAX}% (1 分)")
    print(f"    註:大戶↑+散戶↓ 共振 → 自然累積 2 分,並作為排序優先序 (不再額外加分)")
    large_sig = {}
    large_change_map = {}
    large_pct_map = {}
    large_weeks_map = {}      # 該股大戶變化採用的週數 (3=主路徑3週累計;1=Fallback) → 給報表標明語意
    small_change_map = {}
    small_decrease_sig = {}
    small_weeks_map = {}      # 同上,散戶版本

    if not holders_df.empty:
        holders_df['HoldingSharesLevel'] = pd.to_numeric(holders_df['HoldingSharesLevel'], errors='coerce')
        # 防禦:TDCC CSV 的 percent 若為字串 (含 "%" 或空白),.sum() 會字串串接 → 強制數值化
        holders_df['percent'] = pd.to_numeric(
            holders_df['percent'].astype(str).str.replace('%', '', regex=False).str.strip(),
            errors='coerce'
        )
        holders_df = holders_df.dropna(subset=['HoldingSharesLevel', 'percent'])

        # 5a. 大戶 (400 張以上):主路徑 3 週累計;Fallback 1 週比較
        # 必須用 .isin,否則納入 Level 16 (合計,永遠100%) 和 Level 17 (差異) 會使 .sum() 爆表
        large = holders_df[holders_df['HoldingSharesLevel'].isin(LARGE_HOLDER_LEVELS)]
        for sid, g in large.groupby('stock_id'):
            s = g.groupby('date')['percent'].sum().sort_index()
            if len(s) >= LARGE_HOLDER_WEEKS:
                change = s.iloc[-1] - s.iloc[-LARGE_HOLDER_WEEKS]   # 4 週前到本週 = 3 週累計
                large_sig[sid] = int(change > LARGE_HOLDER_3W_CHANGE_MIN)
                large_weeks_map[sid] = LARGE_HOLDER_WEEKS - 1       # 3 週累計
            elif len(s) >= 2:
                change = s.iloc[-1] - s.iloc[-2]
                large_sig[sid] = int(change > LARGE_HOLDER_CHANGE_MIN)
                large_weeks_map[sid] = 1                            # 1 週 (fallback)
            else:
                continue
            large_change_map[sid] = round(change, 3)
            large_pct_map[sid] = round(s.iloc[-1], 2)

        # 5b. 散戶 (1~15 張):同樣主 3 週累計、Fallback 1 週
        small = holders_df[holders_df['HoldingSharesLevel'].isin(SMALL_HOLDER_LEVELS)]
        for sid, g in small.groupby('stock_id'):
            s = g.groupby('date')['percent'].sum().sort_index()
            if len(s) >= LARGE_HOLDER_WEEKS:
                change = s.iloc[-1] - s.iloc[-LARGE_HOLDER_WEEKS]
                small_decrease_sig[sid] = int(change < SMALL_HOLDER_3W_CHANGE_MAX)
                small_weeks_map[sid] = LARGE_HOLDER_WEEKS - 1
            elif len(s) >= 2:
                change = s.iloc[-1] - s.iloc[-2]
                small_decrease_sig[sid] = int(change < SMALL_HOLDER_CHANGE_MAX)
                small_weeks_map[sid] = 1
            else:
                continue
            small_change_map[sid] = round(change, 3)

    # --- 訊號 7+8+10: 技術面 + KD + RS (共用 daily 迴圈 + 預篩) ---
    print(">>> [6/9] 技術面三合一 (均線多頭含現價>MA20 / 量價齊揚突破 / 連續量增)")
    print(f">>> [7/9] KD 低檔金叉 (近 {KD_LOOKBACK} 日內曾從 K<{KD_LOW_FROM} 金叉,且今日 K<{KD_HIGH_CAP_NOW} 仍維持金叉狀態)")
    print(">>> [8/9] RS vs 大盤")
    print("    同時進行預篩: 20日均量 < {} 張 或 ATR% > {}% 直接剔除".format(MIN_AVG_VOL_LOTS, ATR_MAX_PCT))
    ma_sig, breakout_sig, vol_sig = {}, {}, {}
    tech_sig = {}
    kd_sig   = {}
    kd_cross_k_map  = {}   # 觸發低檔金叉時的 K 值 (供報表診斷)
    kd_cross_ago_map = {}  # 距今幾個交易日前發生金叉
    rs_sig   = {}
    rs_diff_map = {}    # 個股 - 大盤 N 日漲幅差 (%)
    price_map   = {}
    ma20_map    = {}    # 給「資減券增」趨勢過濾使用
    avg_vol_map = {}
    atr_pct_map = {}
    reject_vol  = set()
    reject_atr  = set()

    if not daily_df.empty:
        daily_df = daily_df.sort_values(['stock_id', 'date'])
        vol_col  = 'Trading_Volume' if 'Trading_Volume' in daily_df.columns else 'volume'
        high_col = 'max' if 'max' in daily_df.columns else ('high' if 'high' in daily_df.columns else None)
        low_col  = 'min' if 'min' in daily_df.columns else ('low'  if 'low'  in daily_df.columns else None)
        for sid, g in daily_df.groupby('stock_id'):
            closes = g['close'].reset_index(drop=True)
            opens  = g['open'].reset_index(drop=True)  if 'open' in g.columns else None
            vols   = g[vol_col].reset_index(drop=True) if vol_col in g.columns else None
            highs  = g[high_col].reset_index(drop=True) if high_col else None
            lows   = g[low_col].reset_index(drop=True)  if low_col  else None
            # Bug #3 修正:取出個股實際日期,RS 計分時用此對齊 TWII (避免停牌/新股 RS 失真)
            # 'date' 欄位已在載入時 assert 過存在且整欄轉成 datetime,這裡直接 reset_index 即可
            dates  = g['date'].reset_index(drop=True)

            if len(closes) == 0:
                continue
            price_now = float(closes.iloc[-1])
            price_map[sid] = round(price_now, 2)

            # --- 預篩 A: 20 日均量 (張) ---
            # Bug #4 修正:無法驗證流動性 (新股 < 20 天 或 缺 volume 欄位) → 一律保守剔除,
            # 避免新掛牌股繞過流動性檢查、僅靠法人+籌碼分數蒙混過關
            if vols is not None and len(vols) >= 20:
                avg_vol_lots = vols.tail(20).mean() / 1000.0
                # Bug D 修正：vols 全為 NaN 時 mean=NaN → int(NaN) 拋 ValueError 整個流程崩潰
                if pd.notna(avg_vol_lots):
                    avg_vol_map[sid] = int(avg_vol_lots)
                    if avg_vol_lots < MIN_AVG_VOL_LOTS:
                        reject_vol.add(sid)
                else:
                    reject_vol.add(sid)
            else:
                reject_vol.add(sid)

            # --- 預篩 B: ATR% 波動 ---
            # 缺 high/low、price ≤ 0、或天數不足 (< 15 天) 一律保守剔除
            if len(closes) >= 15 and price_now > 0:
                if highs is not None and lows is not None:
                    atr = _calc_atr(highs, lows, closes, n=14)
                    if atr is not None:
                        atr_pct = atr / price_now * 100
                        atr_pct_map[sid] = round(atr_pct, 2)
                        if atr_pct > ATR_MAX_PCT:
                            reject_atr.add(sid)
                    else:
                        reject_atr.add(sid)        # 理論上不會走到 (len 已 >= 15),保險
                else:
                    reject_atr.add(sid)            # 缺 high/low 欄位 → 無法驗證波動,保守剔除
            else:
                reject_atr.add(sid)                # Bug #4:天數不足或無效價,無法驗證波動 → 保守剔除

            # MA20 獨立計算 (給「資減券增」趨勢過濾使用) — 只需 20 天即可
            # 拆出來才不會讓「資料不足 60 天」的新股拿不到 MA20 → 主迴圈趨勢過濾被誤判為 0
            if len(closes) >= 20:
                ma20_map[sid] = float(closes.tail(20).mean())

            # 均線多頭 (MA5 > MA20 > MA60 AND 現價 > MA20) — 需要 60 天
            if len(closes) >= 60:
                ma5  = float(closes.tail(5).mean())
                ma20 = ma20_map[sid]    # 已算過,直接複用
                ma60 = float(closes.tail(60).mean())
                ma_sig[sid] = int(ma5 > ma20 > ma60 and price_now > ma20)

            # 量價齊揚突破 (突破 N 日新高 AND 當日量 ≥ MA20 × BREAKOUT_VOL_RATIO)
            if len(closes) >= HIGH_BREAK_DAYS + 1 and vols is not None and len(vols) >= 20:
                # 應改為（排除今日，才是真正「突破過去 N 日高點」）
                high_n = closes.iloc[:-1].tail(HIGH_BREAK_DAYS).max()
                vol_today = vols.iloc[-1]
                vol_ma20  = vols.tail(20).mean()
                breakout_sig[sid] = int(
                    closes.iloc[-1] >= high_n * HIGH_TOLERANCE and
                    vol_today >= vol_ma20 * BREAKOUT_VOL_RATIO
                )

            # 視窗 3 天內，有 ≥ 2 天量增即可(近 N 日中 ≥ M 日量 > MA20 × VOL_SURGE_RATIO 且最新一日紅 K)
            if vols is not None and opens is not None and len(vols) >= 20:
                vol_ma20   = vols.tail(20).mean()
                vol_window = vols.tail(VOL_SURGE_WINDOW)
                surge_days = int((vol_window > vol_ma20 * VOL_SURGE_RATIO).sum())
                price_up   = closes.iloc[-1] > opens.iloc[-1]
                vol_sig[sid] = int(surge_days >= VOL_SURGE_DAYS and price_up)

            # 三合一: 任一成立即得 1 分
            tech_sig[sid] = int(
                ma_sig.get(sid, 0) or breakout_sig.get(sid, 0) or vol_sig.get(sid, 0)
            )

            # KD 低檔金叉訊號 (方向 A:近 N 日內曾從低檔啟動,今日仍維持金叉狀態)
            # 條件:
            #   1. 過去 KD_LOOKBACK 天內 (1~N 天前,不含今天),某日「昨 K ≤ 昨 D 且 今 K > 今 D」
            #   2. 該交叉日的 K 值 < KD_LOW_FROM (從低檔啟動)
            #   3. 今日 K > D (金叉狀態維持中,沒死叉回去)
            #   4. 今日 K < KD_HIGH_CAP_NOW (避免追到已超買的股)
            if highs is not None and lows is not None and len(closes) >= KD_N + 1:
                k_list, d_list = _calc_kd_series(highs, lows, closes, n=KD_N)
                if k_list is not None and k_list[-1] is not None and k_list[-2] is not None:
                    k_today = k_list[-1]
                    d_today = d_list[-1]
                    # 條件 3 + 4 先檢查 (今日狀態)
                    if k_today > d_today and k_today < KD_HIGH_CAP_NOW:
                        # 掃描範圍:昨日 (距今 1 天) 到 KD_LOOKBACK 天前 (距今 N 天),今日不算
                        # i 為「金叉發生日」的 index,需要 i-1 才能比較,故 i-1 須有 K/D 值
                        # k_list 前 KD_N-1 個是 None,所以 i-1 >= KD_N-1 → i >= KD_N
                        last_idx   = len(k_list) - 1            # 今日 index
                        newest_i   = last_idx - 1               # 最新可掃日 = 昨日 (距今 1 天)
                        oldest_i   = last_idx - KD_LOOKBACK     # 最舊可掃日 (距今 KD_LOOKBACK 天)
                        oldest_i   = max(oldest_i, KD_N)        # 不能早於有 KD 值的最早一日
                        found = False
                        # 註:oldest_i = max(..., KD_N) 已保證 i ≥ KD_N,故 i-1 ≥ KD_N-1,
                        #     k_list[i-1]/d_list[i-1] 必有值,不需再檢查 None
                        for i in range(newest_i, oldest_i - 1, -1):
                            crossed_today = k_list[i-1] <= d_list[i-1] and k_list[i] > d_list[i]

                            if crossed_today:
                                # 找到了「最近一次」發動的金叉，立刻判定是不是低檔
                                if k_list[i] < KD_LOW_FROM:
                                    kd_sig[sid] = 1
                                    kd_cross_k_map[sid] = round(k_list[i], 1)
                                    kd_cross_ago_map[sid] = last_idx - i
                                    found = True
                                # 關鍵修正：不管這最近一次金叉是否在低檔，都必須 break！
                                # 因為它才是決定現在 K>D 狀態的唯一發動點，不能再往更早的歷史找了。
                                break
                        if not found:
                            kd_sig[sid] = 0
                    else:
                        kd_sig[sid] = 0

            # RS vs 大盤 (個股 N 日漲幅 > 大盤同期)
            # Bug #3 修正:過去用全域 twii_lookback_change (大盤最末日基準),停牌/新股若最末日早於大盤,
            # RS 會錯位 N 天。改成用該股自己的 RS_LOOKBACK 起訖日去 TWII 查表 (asof 找最近 ≤ 該日的值),
            # 確保兩邊基準一致;查不到 TWII 值就放棄計分,而非用錯位值湊數。
            if market_data_ok and twii_close is not None and len(closes) > RS_LOOKBACK:
                end_d   = dates.iloc[-1]
                start_d = dates.iloc[-RS_LOOKBACK - 1]
                twii_end   = twii_close.asof(end_d)
                twii_start = twii_close.asof(start_d)
                if pd.notna(twii_end) and pd.notna(twii_start) and twii_start > 0:
                    stock_change = (closes.iloc[-1] / closes.iloc[-RS_LOOKBACK - 1] - 1) * 100
                    twii_change  = (twii_end / twii_start - 1) * 100
                    rs_diff = stock_change - twii_change
                    rs_diff_map[sid] = round(rs_diff, 2)
                    # RS 看大盤臉色:純多頭才獎勵相對強勢;空頭與「盤整修正」皆不計分。
                    # 依據訊號歸因回測:做頭往下時「過去最強的股」反而摔最重(RS edge 為負);
                    # 2026/5 中~6 月初的修正盤整期 (全程站上季線) RS edge 實測 -4.59%,
                    # 證實「站上季線但月線下/近20日下跌」時追強勢同樣專挑會補跌的名字。
                    rs_sig[sid] = int(rs_diff > 0) if (market_bullish and not market_consolidating) else 0

    # --- 訊號 9: 月營收 — 優先 YoY,YoY 不可行時降級 MoM ---
    # MoM 春節過濾:依據「MoM 序列的目標月份」逐股判斷,而非執行月。
    # 例如 5 月執行時最新營收若為 4 月,MoM 序列含 1→2、2→3、3→4,前兩者受春節污染,整段須停用。
    print(f">>> [9/9] 月營收: 連 {REVENUE_MONTHS} 月 YoY ≥ {REVENUE_YOY_MIN}%;"
          f"YoY 不可行時降級用連 {REVENUE_MOM_MONTHS} 月 MoM ≥ {REVENUE_MOM_MIN}% "
          f"(MoM 序列若任一目標月 ∈ {sorted(SKIP_MOM_MONTHS)} 或月份不連續 → 整段停用)")
    rev_sig = {}
    rev_map = {}       # 最近 1 個月 YoY 或 MoM
    rev3_map = {}      # 最近 N 個月 YoY 或 MoM 字串
    rev_mode_map = {}  # 'YoY' / 'MoM' / 'MoM(未達)'

    if not revenue_df.empty:
        # 防禦:OpenAPI/舊 MOPS 快取混用時,revenue 可能為字串 "1,234,567" 或數值 → 統一清洗
        revenue_df['revenue'] = pd.to_numeric(
            revenue_df['revenue'].astype(str).str.replace(',', '', regex=False).str.strip(),
            errors='coerce'
        )
        # 防禦:revenue_year / revenue_month 若為字串,字典序排序會把 "12" 排到 "2" 前面 → MoM/YoY 全錯
        revenue_df['revenue_year']  = pd.to_numeric(revenue_df['revenue_year'],  errors='coerce').astype('Int64')
        revenue_df['revenue_month'] = pd.to_numeric(revenue_df['revenue_month'], errors='coerce').astype('Int64')
        revenue_df = revenue_df.dropna(subset=['revenue', 'revenue_year', 'revenue_month'])
        for sid, g in revenue_df.groupby('stock_id'):
            g = g.sort_values(['revenue_year', 'revenue_month']).reset_index(drop=True)
            yoy_done = False  # 標記:YoY 路徑是否成功計算 (即使結果是 0 分也算成功)

            # --- 路徑 A: YoY (需 12 + REVENUE_MONTHS 個月) ---
            if len(g) >= 12 + REVENUE_MONTHS:
                last_n = g.tail(REVENUE_MONTHS)
                yoys = []
                ok = True
                for _, row in last_n.iterrows():
                    base = g[(g['revenue_year'] == row['revenue_year'] - 1) &
                             (g['revenue_month'] == row['revenue_month'])]
                    if base.empty or base['revenue'].iloc[0] == 0:
                        ok = False; break
                    yoy = (row['revenue'] - base['revenue'].iloc[0]) / base['revenue'].iloc[0] * 100
                    yoys.append(yoy)
                if ok and yoys:
                    rev_sig[sid]      = int(all(y >= REVENUE_YOY_MIN for y in yoys))
                    rev_map[sid]      = round(yoys[-1], 2)
                    rev3_map[sid]     = ", ".join(f"{y:.1f}%" for y in yoys)
                    rev_mode_map[sid] = "YoY"
                    yoy_done = True

            # --- 路徑 B: MoM 降級 (僅在 YoY 路徑沒成功 + 資料夠用時啟動) ---
            # 注意:必須有「連續 REVENUE_MOM_MONTHS 個月」全部 MoM ≥ 門檻才得分,不會被「跳過某些月」稀釋
            if not yoy_done and len(g) >= REVENUE_MOM_MONTHS + 1:
                win = g.tail(REVENUE_MOM_MONTHS + 1).reset_index(drop=True)
                target_months = win['revenue_month'].iloc[1:].astype(int).tolist()

                # B-1 春節過濾:任一 MoM 的「目標月」屬於 SKIP_MOM_MONTHS 即整段停用
                if any(m in SKIP_MOM_MONTHS for m in target_months):
                    rev_mode_map[sid] = "MoM(春節跳過)"   # 留下診斷標記,讓報表能看出原因
                    continue

                # B-2 月份連續性:確保 win 中各月為相鄰月份 (含跨年 12→1),否則 MoM 不可靠
                ym = list(zip(win['revenue_year'].astype(int), win['revenue_month'].astype(int)))
                if not all(ym[i] == _next_ym(*ym[i-1]) for i in range(1, len(ym))):
                    rev_mode_map[sid] = "MoM(月份不連續)"
                    continue

                moms = []
                ok = True
                for i in range(1, len(win)):
                    if win['revenue'].iloc[i-1] == 0:
                        ok = False; break
                    mom = (win['revenue'].iloc[i] - win['revenue'].iloc[i-1]) / win['revenue'].iloc[i-1] * 100
                    moms.append(mom)
                # 防稀釋:必須有完整 REVENUE_MOM_MONTHS 個 MoM,且全部達標
                if ok and len(moms) >= REVENUE_MOM_MONTHS and all(m >= REVENUE_MOM_MIN for m in moms):
                    rev_sig[sid]      = 1
                    rev_map[sid]      = round(moms[-1], 2)
                    rev3_map[sid]     = ", ".join(f"{m:.1f}%" for m in moms)
                    rev_mode_map[sid] = "MoM"
                else:
                    # 即使沒過,也記錄計算結果方便事後檢視
                    rev_sig[sid] = 0
                    if moms:
                        rev_map[sid]      = round(moms[-1], 2)
                        rev3_map[sid]     = ", ".join(f"{m:.1f}%" for m in moms)
                        rev_mode_map[sid] = "MoM(未達)"

    # ==========================================
    # 計分 + 出表
    # ==========================================
    # 預篩 C: 無 daily K 資料的股 (新掛牌/暫停交易/yfinance 抓不到)
    # 沒有價量,所有技術面/KD/RS 訊號都無從計算,只剩法人/籌碼/營收 → 仍可能累積到過關門檻
    # 為避免「無基本面驗證」的股魚目混珠,直接剔除
    reject_no_daily = set(all_ids) - set(price_map.keys())
    reject_set = reject_vol | reject_atr | reject_no_daily
    both       = reject_vol & reject_atr
    print(">>> 預篩結果:")
    print(f"    · 流動性不足 (20日均量 < {MIN_AVG_VOL_LOTS} 張): {len(reject_vol):>4} 檔")
    print(f"    · 波動過大  (ATR% > {ATR_MAX_PCT}%)          : {len(reject_atr):>4} 檔")
    print(f"    · 無 daily K 資料                           : {len(reject_no_daily):>4} 檔")
    print(f"    · vol+atr 皆觸發 (已計入上述)                : {len(both):>4} 檔")
    print(f"    · 合計剔除 (聯集)                            : {len(reject_set):>4} 檔")
    print(f">>> 計分與排序... (門檻 = {effective_pass_score} / 12)")
    # 盤整期籌碼硬門票:依分數區間實證 (8分勝率53%/+0.71% vs 7分43%/-0.35%),
    # 且大戶↑ edge +6.01%、散戶↓ +3.16% 為僅有的明顯正向訊號 → 盤整時升級為入場條件
    chip_gate_on = market_data_ok and market_consolidating
    chip_gate_rejected = 0
    results = []
    for sid in all_ids:
        if sid in reject_set:
            continue

        it = it_sig.get(sid, {'flag': 0, 'net': 0})
        fi = fi_sig.get(sid, {'flag': 0, 'net': 0})

        # 訊號 3:投信+外資雙買超 (兩者旗標皆為 1 才得分)
        it_fi_dual = int(it['flag'] == 1 and fi['flag'] == 1)

        # 券相關合併: 資減券增 + 趨勢過濾 (現價 > MA20) OR 券資比軋空潛力
        margin_raw = margin_raw_sig.get(sid, 0)
        sm         = sm_ratio_sig.get(sid, 0)
        ma20       = ma20_map.get(sid)
        price_now  = price_map.get(sid)

        # 資減券增需要「現價 > MA20」過濾,排除下跌段散戶斷頭+高檔放空的偽軋空 (對應 line 46 設計意圖)
        # 券資比軋空潛力「不」套用趨勢過濾:真正的軋空常發生在 MA20 附近的轉折/剛突破時,
        # 套趨勢過濾會誤殺底部反彈剛起動的軋空標的
        margin_with_trend = int(margin_raw == 1 and ma20 is not None and price_now is not None and price_now > ma20)
        margin_combined   = int(margin_with_trend or sm == 1)

        l         = large_sig.get(sid, 0)            # 大戶上升 (1 分)
        sd        = small_decrease_sig.get(sid, 0)   # 散戶下降 (1 分,獨立計分)
        chip_sync = int(l == 1 and sd == 1)          # 籌碼共振 (僅供排序,不計分)
        # 籌碼信心分級 (方案 C:把最有 edge 的大戶↑/散戶↓ 當「分級標籤」而非「篩選門票」,
        # 完全不影響誰入選,只標出值得優先看的股。依訊號歸因:大戶↑ edge +6%、散戶↓ +3%。
        # 累積一段時間後可用 sig 欄回測「高信心 vs 一般」前進報酬,證實有效再升級為硬門票。)
        if chip_sync:
            chip_tier = "🔥 高信心"      # 大戶↑ 且 散戶↓ 共振
        elif l == 1 or sd == 1:
            chip_tier = "⭐ 中信心"      # 大戶↑ 或 散戶↓ 其一
        else:
            chip_tier = "一般"

        tech = tech_sig.get(sid, 0)
        kd   = kd_sig.get(sid, 0)
        rv   = rev_sig.get(sid, 0)
        rs   = rs_sig.get(sid, 0)

        # 滿分 12:法人 3 (投信、外資、雙買) + 籌碼 5 (券 1、大戶 2、散戶 2) + 技術 2 (技術面、KD) + 基本 1 + 大盤 1
        # 大戶/散戶各 2 分:訊號歸因實證它們是唯二明顯正 edge(+9.2% / +6.4%),其餘訊號
        # 接近門票性質。加重後門檻 7 分 = 「基本盤(投信+外資+雙買+技術+RS+券=6)之外,
        # 至少要一個籌碼訊號」,籌碼實質升級為全時段入場條件
        score = (it['flag'] + fi['flag'] + it_fi_dual +
                 margin_combined + 2 * l + 2 * sd +
                 tech + kd + rv + rs)

        if score < effective_pass_score:
            continue

        # 盤整期硬門票:達分數門檻後,還需「大戶↑ 或 散戶↓」至少其一 (⭐中信心以上)。
        # 只在盤整 regime 啟用;多頭/空頭維持原邏輯 (標籤不影響入選),保留對照組可持續回測。
        if chip_gate_on and not (l == 1 or sd == 1):
            chip_gate_rejected += 1
            continue

        results.append({
            "代號": sid, "名稱": name_map.get(sid, ""), "總分": score,
            "籌碼信心": chip_tier,    # 方案 C 分級標籤 (🔥高信心=共振 / ⭐中信心=其一 / 一般);不影響入選
            # 法人三大訊號
            "投信買超": it['flag'], "外資買超": fi['flag'], "投信+外資雙買": it_fi_dual,
            # 券相關
            "券相關": margin_combined,
            "·資減券增(原始)": margin_raw,
            "·資減券增(過趨勢)": margin_with_trend,
            f"·券資比>{int(SHORT_MARGIN_RATIO*100)}%": sm,
            # 大戶 / 散戶 (各自計分) + 共振 (僅排序)
            "400張大戶上升": l,
            "·大戶累計變化(%)": large_change_map.get(sid),
            "·大戶採用週數":     large_weeks_map.get(sid),     # 3=主路徑3週累計;1=Fallback 1週
            "·400張大戶總比例(%)": large_pct_map.get(sid),
            "散戶下降": sd,
            "·散戶累計變化(%)": small_change_map.get(sid),
            "·散戶採用週數":     small_weeks_map.get(sid),
            "★籌碼共振(大戶↑散戶↓)": chip_sync,    # 不計分,僅作排序優先序
            # 技術面
            "技術面": tech,
            "·均線多頭(含現價>MA20)": ma_sig.get(sid, 0),
            f"·{HIGH_BREAK_DAYS}日量價齊揚突破": breakout_sig.get(sid, 0),
            f"·連續量增({VOL_SURGE_DAYS}/{VOL_SURGE_WINDOW}日)": vol_sig.get(sid, 0),
            f"KD低檔金叉(近{KD_LOOKBACK}日內<{KD_LOW_FROM})": kd,
            "·金叉日 K 值": kd_cross_k_map.get(sid),
            "·金叉發生(N天前)": kd_cross_ago_map.get(sid),
            # 月營收
            f"連月營收達標(YoY≥{REVENUE_YOY_MIN}% 或 MoM≥{REVENUE_MOM_MIN}%)": rv,
            "營收模式": rev_mode_map.get(sid, ""),
            # RS vs 大盤
            "RS優於大盤": rs,
            "·個股-大盤漲幅差(%)": rs_diff_map.get(sid),
            # 數值欄位 (TWSE/TPEx 原始為「股」,/1000 → 「張」)
            "投信5日淨額(張)": int(it['net'] / 1000),
            "外資5日淨額(張)": int(fi['net'] / 1000),
            "券資比(%)":          sm_ratio_map.get(sid),
            "最新月營收增率(%)":  rev_map.get(sid),
            "近月增率序列":       rev3_map.get(sid, ""),
            "現價":               price_map.get(sid),
            "20日均量(張)":       avg_vol_map.get(sid),
            "ATR%":               atr_pct_map.get(sid),
            "產業":               ind_map.get(sid, "")
        })

    if results:
        # 排序: 總分 → 籌碼共振 → RS → 外資淨額
        df = pd.DataFrame(results).sort_values(
            ["總分", "★籌碼共振(大戶↑散戶↓)", "RS優於大盤", "外資5日淨額(張)"],
            ascending=[False, False, False, False]
        )
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out = output_dir / f"選股結果_{stamp}.xlsx"
        try:
            df.to_excel(out, index=False)
        except PermissionError:
            out = output_dir / f"選股結果_{stamp}_alt.xlsx"
            df.to_excel(out, index=False)
            print(f"   ⚠ 原 xlsx 被開啟中,改存為 {out}")
        print("=" * 60)
        print(f"✅ 篩選完成!共 {len(df)} 檔達標 (總分 >= {effective_pass_score} / 12)")
        if chip_gate_on:
            print(f"   · [盤整慎選] 籌碼硬門票剔除 (達分但無大戶↑/散戶↓): {chip_gate_rejected} 檔")
        chip_sync_count = (df["★籌碼共振(大戶↑散戶↓)"] == 1).sum()
        rs_count        = (df["RS優於大盤"] == 1).sum()
        dual_count      = (df["投信+外資雙買"] == 1).sum()
        print(f"   · 籌碼共振 (大戶↑+散戶↓) : {chip_sync_count} 檔")
        high_conf = int((df["籌碼信心"] == "🔥 高信心").sum())
        mid_conf  = int((df["籌碼信心"] == "⭐ 中信心").sum())
        print(f"   · 籌碼信心分級           : 🔥高信心 {high_conf} / ⭐中信心 {mid_conf} / 一般 {len(df) - high_conf - mid_conf}")
        print(f"   · 投信+外資雙買超         : {dual_count} 檔")
        print(f"   · RS 優於大盤             : {rs_count} 檔")
        print(f"📍 報表已產出:{os.path.abspath(out)}")
        dsl_out = output_dir / f"嘉實自選股匯入檔_{stamp}.dsl"
        write_xq_dsl(df["代號"].tolist(), str(dsl_out))
        khsg_out = output_dir / f"精誠自選股匯入檔_{stamp}.xls"
        write_precision_xls(df["代號"].tolist(), name_map, str(khsg_out))
        print("=" * 60)
        file_paths = {'xlsx': out, 'dsl': dsl_out, 'xls': khsg_out}
    else:
        if chip_gate_on and chip_gate_rejected > 0:
            # 達分股全被籌碼門票刷掉 → 提示真正原因,別誤導使用者去調低門檻
            print(f"❌ 有 {chip_gate_rejected} 檔達 {effective_pass_score} 分,但均無「大戶↑或散戶↓」訊號,"
                  f"被盤整慎選門票剔除。盤整期寧缺勿濫,屬正常現象。")
        else:
            print(f"❌ 無標的達 {effective_pass_score} 分,建議調低 PASS_SCORE 或放寬個別條件。")
        df = pd.DataFrame()
        file_paths = {}

    # UI 用 meta:大盤橫幅、effective_pass_score、cache 日期都靠這個
    # === [修改處] 優化 meta 資料，供 Telegram 推播戰情摘要使用 ===    
    # 計算大盤今日漲跌幅 (需檢查 twii_close 是否有足夠資料計算昨日)   
    twii_pct = 0.0
    bias_ma60 = 0.0  # 👈 修正：確保無論如何都有這個變數，防止 NameError
    
    if market_data_ok and len(twii_close) >= 2:
        c_now = twii_close.iloc[-1]
        c_prev = twii_close.iloc[-2]
        twii_pct = ((c_now - c_prev) / c_prev) * 100
        # 2. 計算與季線的乖離率
        if twii_ma > 0:
            bias_ma60 = ((c_now - twii_ma) / twii_ma) * 100
    # 3. 產生門檻變動文字標註
    score_note = ""
    if market_data_ok and not market_bullish:
        score_note = f"⚠️ 大盤跌破季線，門檻已由 {pass_score} 提高至 {effective_pass_score}"
    elif market_data_ok and market_consolidating:
        score_note = "⚠️ 大盤盤整修正中：RS 不計分，且需「大戶↑或散戶↓」至少其一才入選"
    elif not market_data_ok:
        score_note = "⚠️ 大盤數據取得失敗，門檻自動調整"

    if not market_bullish:
        market_state, market_status = 'bear', "謹慎保守"
    elif market_consolidating:
        market_state, market_status = 'consolidation', "盤整慎選"
    else:
        market_state, market_status = 'bull', "偏多操作"

    meta = {
        'market_data_ok':        market_data_ok,
        'market_bullish':        market_bullish,
        'market_consolidating':  market_consolidating,
        'market_state':          market_state,   # 'bull' / 'consolidation' / 'bear'
        'market_status':         market_status,
        'twii_now':              twii_now,
        'twii_ma':               twii_ma,
        'twii_pct':              twii_pct,
        'twii_bias':             bias_ma60,
        'score_note':            score_note,
        'base_pass_score':       pass_score,           # Bug 1 修正：原始門檻（調整前）
        'effective_pass_score':  effective_pass_score,
        'cache_max_date':        cache_max_date,
        'twii_lookback_change':  twii_lookback_change, # Bug 2 修正：key 名稱與 UI 對齊
        'panic_guard':           False,                # 恐慌煞車未觸發 (觸發時在前面提早 return)
    }
    return (df, file_paths, meta)


if __name__ == '__main__':
    run_screening()

def get_stock_history(stock_id: str, n_days: int = 200) -> pd.DataFrame:
    """
    從 daily 快取讀取單一股票的 OHLCV 歷史，供 UI K 線圖使用。
    使用 pyarrow filters 在讀取層就過濾，只載入該股資料，不讀整張表。
    n_days=0 代表不限筆數，回傳全部歷史。
    """
    files = sorted(CACHE_DIR.glob('daily_*.parquet'))
    if not files:
        return pd.DataFrame()
    try:
        g = pd.read_parquet(
            files[-1],
            filters=[('stock_id', '==', str(stock_id))]  # ← pyarrow 層過濾
        )
        if g.empty:
            return pd.DataFrame()
        g['date'] = pd.to_datetime(g['date'])
        g = g.sort_values('date').reset_index(drop=True)
        if n_days and len(g) > n_days:
            g = g.tail(n_days).reset_index(drop=True)
        return g
    except Exception:
        return pd.DataFrame()