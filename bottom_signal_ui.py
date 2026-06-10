# -*- coding: utf-8 -*-
"""止跌判讀分頁 UI(Streamlit)

視覺設計沿用《tw_stock_bottom_checklist.html》:
- 恐慌溫度計(漸層軌道 + 滑標 + 閘門鎖頭 🔒)
- 核心三開關 chips
- 六大類卡片 + 每項 ✅/⬜/❓ + 實際數值 + 點開白話說明(<details>)
- 歷史色帶(每天一格,紅黃橘綠)

與既有 app 的整合方式同「大盤情緒」分頁:
- run_all_checks 用 @st.cache_data(ttl=30min) 包住
- 寫檔(歷史/人工勾選)在 cache 外執行,避免被擋
- 抓取約需 1 分鐘 → 進分頁先按「開始判讀」,不拖慢整個 app 啟動
"""
import streamlit as st
import streamlit.components.v1 as components

from bottom_signal import (
    run_all_checks, apply_manual_flags,
    load_manual_flags, save_manual_flags,
    load_bottom_history, persist_bottom_history,
    load_bottom_latest, persist_bottom_latest,
    format_bottom_for_tg, LEVELS, CORE_KEYS, EXPLAIN,
)

# 與參考網頁一致的四級顏色(紅 → 黃 → 青 → 藍)
_TONE = ["#FF6B4A", "#F5B544", "#36C8B2", "#3F9DFF"]
_POS = [8, 33, 62, 90]          # 溫度計滑標位置(%)
_CORE_NAMES = {"vix_fall": "VIXTWN翻落", "tsmc_no_low": "台積止穩",
               "news_dulled": "利空鈍化"}

_CSS = """
<style>
  :root{
    --bg:#0E1217; --surface:#161C24; --surface2:#1C232E; --border:#2A323D;
    --text:#E8ECF2; --muted:#8B95A5; --faint:#5C6675;
    --hot:#FF6B4A; --warm:#F5B544; --cool:#36C8B2; --done:#3F9DFF; --star:#F5B544;
    --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
    --cjk:"PingFang TC","Noto Sans TC","Microsoft JhengHei",system-ui,sans-serif;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);font-family:var(--cjk);
    -webkit-font-smoothing:antialiased;line-height:1.55;padding:6px 4px}
  .gauge{background:var(--surface);border:1px solid var(--border);border-radius:14px;
    padding:16px 16px 18px;margin-bottom:14px}
  .verdict{display:flex;align-items:center;gap:10px;margin-bottom:6px}
  .vdot{width:12px;height:12px;border-radius:50%;flex:none;box-shadow:0 0 12px currentColor}
  .vlabel{font-size:21px;font-weight:800}
  .vnote{color:var(--muted);font-size:14px;margin:0 0 14px;padding-left:22px}
  .track{position:relative;height:12px;border-radius:99px;margin-top:18px;
    background:linear-gradient(90deg,var(--hot) 0%,var(--warm) 34%,var(--cool) 66%,var(--done) 100%);
    opacity:.85}
  .marker{position:absolute;top:50%;width:20px;height:20px;border-radius:50%;background:#fff;
    border:3px solid var(--bg);transform:translate(-50%,-50%);box-shadow:0 2px 8px rgba(0,0,0,.5)}
  .gatelock{position:absolute;top:-22px;left:28%;transform:translateX(-50%);font-size:13px}
  .scale{display:flex;justify-content:space-between;font-family:var(--mono);font-size:11.5px;
    color:var(--faint);margin-top:9px;letter-spacing:.5px}
  .switches{display:flex;gap:7px;flex-wrap:wrap;margin-top:14px}
  .chip{font-family:var(--mono);font-size:12.5px;padding:5px 11px;border-radius:99px;
    border:1px solid var(--border);color:var(--faint);background:var(--surface2);
    display:flex;align-items:center;gap:5px}
  .chip.on{color:var(--cool);border-color:var(--cool)}
  .chip .s{font-size:10px}
  .count{font-family:var(--mono);font-size:13px;color:var(--muted);margin-top:12px}
  .count b{color:var(--text)}
  .ctx{margin:0 0 14px;background:var(--surface);border:1px solid var(--border);border-radius:10px;
    padding:10px 13px;font-family:var(--mono);font-size:13px;color:var(--muted);
    display:flex;flex-wrap:wrap;gap:4px 16px}
  .ctx b{color:var(--text);font-weight:600}

  /* 兩欄並排(01+02 一排、03+04 一排…),窄螢幕自動退回單欄 */
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start}
  @media (max-width:680px){.grid{grid-template-columns:1fr}}
  .cat{background:var(--surface);border:1px solid var(--border);border-radius:12px;
    margin-bottom:0;overflow:hidden}
  .cat-h{display:flex;align-items:baseline;gap:9px;padding:14px 14px 9px}
  .cat-n{font-family:var(--mono);font-size:13px;color:var(--faint);font-weight:600}
  .cat-t{font-size:17px;font-weight:700}
  .cat-tag{margin-left:auto;font-family:var(--mono);font-size:11px;letter-spacing:.5px;color:var(--warm);
    border:1px solid rgba(245,181,68,.45);padding:2px 8px;border-radius:99px;align-self:center}
  .items{padding:0 8px 8px}
  .item{display:flex;gap:10px;align-items:flex-start;padding:11px 8px;border-radius:9px}
  .item.core{background:linear-gradient(90deg,rgba(245,181,68,.09),transparent)}
  .item.gate{background:linear-gradient(90deg,rgba(255,107,74,.11),transparent)}
  .box{flex:none;width:22px;height:22px;margin-top:1px;border:2px solid var(--faint);
    border-radius:6px;display:grid;place-items:center;font-size:14px;line-height:1;color:#06231F}
  .item.checked .box{background:var(--cool);border-color:var(--cool)}
  .item.na .box{border-style:dashed;color:var(--faint)}
  .ltext{flex:1;min-width:0}
  .label{font-size:16.5px;line-height:1.5}
  .item.checked .label{color:var(--muted)}
  .val{display:block;font-family:var(--mono);font-size:13.5px;color:var(--faint);margin-top:3px}
  .item.checked .val{color:var(--cool)}
  .badge{font-family:var(--mono);font-size:11px;margin-left:6px;padding:1px 7px;
    border-radius:99px;white-space:nowrap}
  .badge.star{color:var(--star);border:1px solid rgba(245,181,68,.5)}
  .badge.gatebadge{color:var(--hot);border:1px solid rgba(255,107,74,.55)}
  .badge.manual{color:var(--done);border:1px solid rgba(63,157,255,.5)}
  details{margin:0 8px 5px 41px}
  details summary{cursor:pointer;list-style:none;font-family:var(--mono);font-size:12.5px;
    color:var(--faint);user-select:none}
  details summary:hover{color:var(--cool)}
  details[open] summary{color:var(--cool)}
  details .body{border-left:2px solid rgba(54,200,178,.55);padding:8px 0 8px 12px;
    margin-top:6px;font-size:14.5px;line-height:1.7;color:var(--muted)}

  .hist{display:flex;flex-wrap:wrap;gap:4px;padding:4px 2px}
  .cell{width:18px;height:18px;border-radius:4px;flex:none}
  .legend{display:flex;gap:14px;font-family:var(--mono);font-size:12.5px;color:var(--muted);
    margin-top:10px;flex-wrap:wrap}
  .legend span{display:flex;align-items:center;gap:5px}
  .dot{width:10px;height:10px;border-radius:3px;display:inline-block}
  .foot{margin-top:8px;color:var(--faint);font-size:13px;line-height:1.7}
</style>
"""


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _build_gauge_html(result: dict) -> str:
    lv = result["level"]
    tone = _TONE[lv]
    note = LEVELS[lv].get("note", "")
    d = {it["key"]: it for it in result["items"]}
    gate_on = d["gate"]["ok"] is True
    n_total = len(result["items"])
    n_core_on = sum(1 for k in CORE_KEYS if d.get(k, {}).get("ok") is True)

    chips = "".join(
        f'<span class="chip{" on" if d.get(k, {}).get("ok") is True else ""}">'
        f'<span class="s">{"●" if d.get(k, {}).get("ok") is True else "○"}</span>'
        f'{_CORE_NAMES.get(k, k)}</span>'
        for k in CORE_KEYS)

    vix = result.get("vixtwn")
    ctx = (f'<div class="ctx"><span>資料日 <b>{_esc(result["asof"])}</b></span>'
           f'<span>VIXTWN <b>{vix:.2f}</b></span>'
           f'<span>閘門 <b>{"🔓 已開" if gate_on else "🔒 未開"}</b></span></div>'
           if vix is not None else "")

    return f"""{_CSS}{ctx}
<div class="gauge">
  <div class="verdict">
    <span class="vdot" style="color:{tone}"></span>
    <span class="vlabel" style="color:{tone}">{_esc(result["level_label"])}</span>
  </div>
  <div class="vnote">{_esc(note)}</div>
  <div class="track">
    <span class="gatelock" style="opacity:{'0.45' if gate_on else '1'}">{"🔓" if gate_on else "🔒"}</span>
    <span class="marker" style="left:{_POS[lv]}%"></span>
  </div>
  <div class="scale"><span>🔥 高度恐慌</span><span>剛降溫</span><span>在打底</span><span>❄️ 止跌確認</span></div>
  <div class="switches">{chips}</div>
  <div class="count">已成立 <b>{result["n_ok"]}</b> / {n_total} 項　·　核心三開關 <b>{n_core_on}</b> / 3</div>
</div>"""


def _build_checklist_html(result: dict) -> str:
    # 依 group 分組(維持原順序)。人工項不放這張唯讀卡
    # ——改用下方「真的能按」的 st.checkbox 區塊,避免使用者點了沒反應。
    groups, order = {}, []
    for it in result["items"]:
        if it["manual"]:
            continue
        if it["group"] not in groups:
            groups[it["group"]] = []
            order.append(it["group"])
        groups[it["group"]].append(it)

    cards = []
    for g in order:
        n, _, t = g.partition(" ")
        tag = ('<span class="cat-tag">閘門·最關鍵</span>' if n == "01" else
               '<span class="cat-tag">決定性</span>' if "利空" in t else
               '<span class="cat-tag">只加分</span>' if "輔助" in t else "")
        rows = []
        for it in groups[g]:
            ok = it["ok"]
            cls = "item"
            if ok is True:
                cls += " checked"
            elif ok is None:
                cls += " na"
            if it["key"] == "gate":
                cls += " gate"
            elif it["key"] in CORE_KEYS:
                cls += " core"
            mark = "✓" if ok is True else ("?" if ok is None else "")
            badge = ('<span class="badge gatebadge">閘門</span>' if it["key"] == "gate" else
                     '<span class="badge star">★ 核心</span>' if it["key"] in CORE_KEYS else "")
            if it["manual"]:
                badge += '<span class="badge manual">人工</span>'
            val = "" if it["manual"] else f'<span class="val">{_esc(it["value"])}</span>'
            rows.append(
                f'<div class="{cls}"><span class="box">{mark}</span>'
                f'<span class="ltext"><span class="label">{_esc(it["name"])}</span>{badge}{val}</span></div>')
            exp = EXPLAIN.get(it["key"])
            if exp:
                rows.append(f'<details><summary>? 白話說明</summary>'
                            f'<div class="body">{_esc(exp)}</div></details>')
        cards.append(
            f'<div class="cat"><div class="cat-h"><span class="cat-n">{_esc(n)}</span>'
            f'<span class="cat-t">{_esc(t)}</span>{tag}</div>'
            f'<div class="items">{"".join(rows)}</div></div>')
    foot = ('<div class="foot">最重要的規則:只要「恐慌指數沒跌破 40」,判讀就會停在最恐慌那一格'
            '——其他再多打勾也一樣。這是幫你做判斷的框架,不是投資建議。</div>')
    return f'{_CSS}<div class="grid">{"".join(cards)}</div>{foot}'


def _build_history_html(history: list) -> str:
    if not history:
        return _CSS + '<div class="foot">尚無歷史紀錄。每天收盤後跑一次判讀,色帶就會逐日累積。</div>'
    cells = "".join(
        f'<span class="cell" style="background:{_TONE[h.get("level", 0)]}" '
        f'title="{_esc(h["date"])} {_esc(LEVELS[h.get("level", 0)]["label"])}'
        f'(成立 {h.get("n_ok", "?")} 項)"></span>'
        for h in history)
    legend = "".join(
        f'<span><span class="dot" style="background:{_TONE[i]}"></span>{lv["label"]}</span>'
        for i, lv in enumerate(LEVELS))
    return (f"{_CSS}<div class='hist'>{cells}</div><div class='legend'>{legend}</div>"
            f"<div class='foot'>每格一天(滑鼠停留看日期),由左到右為舊到新,最多保留 120 天。</div>")


# ══════════════════════════════════════════════════════════════════════
# Streamlit 進入點
# ══════════════════════════════════════════════════════════════════════
def render_bottom_tab(cache_dir):
    """止跌判讀分頁主體。在 screening_ui16 的 with _tab_bottom: 裡呼叫。

    速度設計:預設「只讀檔」——排程(bottom_push.py)算好的完整結果存在
    bottom_signal_latest.json,打開網頁秒開、零網路請求。
    只有使用者明確按「立即重抓」才現場抓一次(約 1 分鐘),抓完也存回檔,
    所以每次點擊最多抓一次,不會因快取過期讓整個 app 卡住。
    """
    st.markdown("##### 🛑 台股止跌判讀")
    st.caption("每天收盤後自動逐項勾稽 21 項訊號 · VIXTWN 跌破 40 是閘門 · 點各項「? 白話說明」看判讀理由")
    # 放大 checkbox 標籤字體(配合檢查表卡片的字級)
    st.markdown("<style>div[data-testid='stCheckbox'] label p{font-size:1.06rem}</style>",
                unsafe_allow_html=True)

    # 使用者按了「立即重抓」→ 這一輪現抓一次,存檔後旗標歸零
    if st.session_state.pop("bs_live_fetch", False):
        with st.spinner("抓取期交所 / 證交所 / 國際行情中…(約 1 分鐘)"):
            result = run_all_checks(cache_dir=cache_dir)
        persist_bottom_latest(cache_dir, result)
        persist_bottom_history(cache_dir, result)
    else:
        result = load_bottom_latest(cache_dir)

    if result is None:
        # 還沒有排程算好的檔(第一次部署/本機第一次用)→ 給手動觸發鈕
        st.info("尚無排程算好的結果。部署後排程每天 16:10 / 21:30 會自動算好;"
                "現在也可以按下面的按鈕現場抓一次(約 1 分鐘)。")
        if st.button("▶️ 立即判讀", type="primary"):
            st.session_state["bs_live_fetch"] = True
            st.rerun()
        history = load_bottom_history(cache_dir)
        if history:
            st.markdown("###### 📜 分級歷史")
            components.html(_build_history_html(history), height=130, scrolling=False)
        return

    # 人工勾選(存檔在 cache 外,改了立即重算分級,不重抓)
    flags = load_manual_flags(cache_dir)
    result = apply_manual_flags(result, flags)

    # 告警(VIXTWN 雙來源失敗等)不能靜默
    for a in result.get("alerts", []):
        st.warning(a)

    components.html(_build_gauge_html(result), height=345, scrolling=False)

    rc1, rc2 = st.columns([3, 1])
    with rc1:
        st.caption(f"資料日 {result['asof']} · 算於 {result.get('generated_at', '?')}"
                   "(排程每日 16:10 / 21:30 自動更新)")
    with rc2:
        if st.button("🔄 立即重抓", help="不等排程,現場抓最新資料重算一次(約 1 分鐘)"):
            st.session_state["bs_live_fetch"] = True
            st.rerun()

    # 歷史由排程(bottom_push)與「立即重抓」負責寫入;
    # 讀檔顯示時不寫——避免假日打開網頁把舊資料記成今天的紀錄。

    # ── 檢查表卡片(自動判定的 19 項,唯讀,兩欄並排) ──
    auto_items = [it for it in result["items"] if not it["manual"]]
    group_sizes = []
    for it in auto_items:           # 依出現順序數每組項數
        if not group_sizes or it["group"] != group_sizes[-1][0]:
            group_sizes.append([it["group"], 0])
        group_sizes[-1][1] += 1
    sizes = [n for _, n in group_sizes]
    # 兩欄:每排高度取左右較高者(每項約 100px + 卡片標頭 75px)
    est_h = sum(max(sizes[i:i + 2]) * 100 + 75
                for i in range(0, len(sizes), 2)) + 90
    components.html(_build_checklist_html(result), height=min(est_h, 2500), scrolling=True)

    # ── 07 利空消息:真的能按的人工勾選(勾完分級立即重算) ──
    # 用 on_change 只在「使用者真的點了那一下」才寫檔;
    # 不能用「widget 值 vs 檔案值」比對——多開視窗時舊視窗會把新勾選蓋掉。
    def _save_flags_cb():
        save_manual_flags(cache_dir, {
            "news_dulled": bool(st.session_state.get("bs_news_dulled")),
            "news_resolved": bool(st.session_state.get("bs_news_resolved")),
        })

    with st.container(border=True):
        st.markdown("**07 利空消息(人工勾選)** ⭐ 核心")
        st.checkbox("**利空鈍化** — 壞消息再出來,股價卻跌不動了",
                    value=bool(flags.get("news_dulled")),
                    key="bs_news_dulled", on_change=_save_flags_cb)
        with st.expander("? 白話說明(利空鈍化)"):
            st.write(EXPLAIN["news_dulled"])
        st.checkbox("**利空解除** — 造成恐慌的原因本身落幕、明朗化",
                    value=bool(flags.get("news_resolved")),
                    key="bs_news_resolved", on_change=_save_flags_cb)
        with st.expander("? 白話說明(利空解除)"):
            st.write(EXPLAIN["news_resolved"])

    # ── 歷史色帶 ──
    st.markdown("###### 📜 分級歷史")
    history = load_bottom_history(cache_dir)
    components.html(_build_history_html(history), height=130, scrolling=False)

    # ── Telegram 預覽(第 3 步接排程推播用同一格式) ──
    with st.expander("📲 Telegram 訊息預覽"):
        st.code(format_bottom_for_tg(result), language=None)
