"""浼氳璁板綍 / 鍙鍖?/ 鎬荤粨 鎻掍欢銆?
``meeting_summarize`` 鏈変袱鏉¤矾寰勶細

* **鑱旂綉璺緞**锛堥厤缃簡 ``deepseek_api_key``锛夛細鎶婁細璁浆鍐欐嫾鎴?Prompt锛岄€氳繃
  DeepSeek 鐨?Anthropic 鍏煎绔偣锛坄`/anthropic/v1/messages``锛夎皟鐢ㄦā鍨嬶紝骞跺湪
  璇锋眰閲屽０鏄庢湇鍔＄ ``web_search`` 宸ュ叿 鈥斺€?鏄惁鑱旂綉鐢辨ā鍨嬭嚜琛屽喅瀹氾紝妫€绱㈢粨鏋滅敱
  鏈嶅姟绔敞鍏ュ苟浠?``web_search_tool_result`` 鍧楄繑鍥炪€?* **鏈湴鍥為€€璺緞**锛堟湭閰嶇疆 key / API 璋冪敤澶辫触锛夛細閫€鍥?``_summarize_local`` 鐨?  绾瓧绗︿覆缁撴瀯鍖栨彁鍙栵紝骞跺湪杈撳嚭閲屾爣璁?``fallback: true``銆?
缃戠粶璁块棶涓€寰嬭蛋 ``httpx.AsyncClient``锛涙ā鍧楅《灞備笉鍋氫换浣?IO銆?"""
from __future__ import annotations

import json
import re
import time

import httpx

from plugin.sdk.plugin import (
    NekoPluginBase,
    Ok,
    lifecycle,
    llm_tool,
    neko_plugin,
    plugin_entry,
)

# ---------------------------------------------------------------------------
# 甯搁噺
# ---------------------------------------------------------------------------

DEFAULT_MAX_POINTS = 5
MAX_POINTS_CAP = 20

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_WEB_SEARCH_MAX_USES = 5
MAX_WEB_SEARCH_USES_CAP = 10
DEFAULT_REQUEST_TIMEOUT = 60.0

# DeepSeek 鐨?Anthropic 鍏煎鍏ュ彛銆?_ANTHROPIC_MESSAGES_PATH = "/anthropic/v1/messages"
# 鏈嶅姟绔仈缃戞绱㈠伐鍏风殑鐗堟湰鏍囪瘑锛圓nthropic Messages API 鐨?tool type锛夈€?_WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
_MAX_OUTPUT_TOKENS = 4096
_MAX_RELATED_CONTEXT = 5

# 鍗曟宸ュ叿璋冪敤鐨勬€婚绠椼€俙`@llm_tool(timeout=60.0)`` 鏄涓讳晶纭笂闄愶細鐪熷疄鐨?# HTTP 寰€杩斿繀椤诲湪杩欎釜绐楀彛鍐呯粨鏉燂紝鍚﹀垯浼氳瀹夸富鎴柇鎴?TOOL_TIMEOUT锛岀湅涓嶅埌鐪熷疄
# 閿欒銆傝繖閲岀暀 5s 浣欓噺缁?IPC 涓庡簭鍒楀寲銆?_TOTAL_BUDGET_SECONDS = 55.0
_MIN_REQUEST_TIMEOUT = 5.0

# 鐥呮€佽緭鍏ヤ繚鎶わ細瓒呴暱杞啓鍏堟埅鏂紝閬垮厤鍗曟宸ュ叿璋冪敤闀挎椂闂村崰鐢ㄥ璇濊疆銆?_MAX_TRANSCRIPT_CHARS = 20_000
_MAX_SENTENCES = 400
_MIN_SENTENCE_CHARS = 4
_MAX_SUMMARY_BODY_CHARS = 300

# 鍙ュ瓙缁堟绗︼細涓枃鏍囩偣锛涙垨鑻辨枃鍙ュ彿鍚庤窡绌虹櫧/缁撳熬锛堥伩鍏嶈鍒?3.5 / v1.2锛夈€?_SENTENCE_RE = re.compile(
    r"[^銆傦紒锛??锛?\n]+?(?:[銆傦紒锛??锛?]|\.(?=\s|$)|\n|$)"
)

# 寰呭姙淇″彿璇嶏細鍛戒腑鍗宠涓恒€岃鍔ㄩ」銆嶃€傛瘮杈冨墠缁熶竴 casefold銆?_ACTION_HINTS = (
    "闇€瑕?, "搴旇", "璐熻矗", "璺熻繘", "瀹夋帓", "纭", "瀹屾垚", "鎺ㄨ繘",
    "涓嬪懆", "鏄庡ぉ", "涔嬪墠", "鎴", "灏藉揩", "钀藉疄", "寰呭姙",
    "todo", "action", "follow up", "follow-up", "assign",
    "deadline", "next step", "owner",
)

# ---------------------------------------------------------------------------
# 绾嚱鏁拌緟鍔╁眰锛氭棤 IO銆佹棤鍓綔鐢ㄣ€佹棤 await锛屽彲鐩存帴鍗曟祴
# ---------------------------------------------------------------------------


def _coerce_text(value: object, default: str = "") -> str:
    """瀹夊叏鍦版妸閰嶇疆鍊?/ API 杩斿洖鍊煎彇鎴愬瓧绗︿覆銆?""
    if isinstance(value, str):
        return value.strip()
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return str(value)
    return default


def _coerce_bool(value: object, default: bool) -> bool:
    """鎶婇厤缃噷鐨勫竷灏斿€煎綊涓€鍖栵紝鍏煎 ``"true"`` / ``"1"`` 杩欑被瀛楃涓插啓娉曘€?""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_timeout(value: object, default: float) -> float:
    """鎶婇厤缃噷鐨勮秴鏃跺€煎綊涓€鍖栦负姝ｆ诞鐐规暟锛岄潪娉曞€煎洖钀藉埌榛樿鍊笺€?""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value.strip())
        except (TypeError, ValueError):
            return default
    else:
        return default
    return parsed if parsed > 0 else default


def _http_status_of(exc: BaseException) -> int | str:
    """浠庡紓甯搁噷瀹夊叏鍙栧嚭 HTTP 鐘舵€佺爜锛屽彇涓嶅埌灏辫繑鍥?``"-"``銆?
    鍙彇鐘舵€佺爜锛岀粷涓嶇 response body 鈥斺€?body 鍙兘鍥炴樉璇锋眰鍐呭銆傛湁浜嗗畠锛?    銆岀鐐?宸ュ叿绫诲瀷涓嶈鏀寔銆嶏紙閫氬父 400锛変笌銆岄壌鏉冨け璐ャ€嶏紙401锛夈€併€岃矾寰勫啓閿欍€?    锛?04锛夋墠鑳藉湪鏃ュ織閲屽尯鍒嗗紑銆傜函鍑芥暟銆?    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status
    return "-"


def _as_text_list(value: object, *, max_items: int) -> list[str]:
    """鎶婃ā鍨嬭繑鍥炵殑浠绘剰褰㈢姸褰掍竴鎴愬瓧绗︿覆鍒楄〃銆?
    妯″瀷鏈夋鐜囨妸鏁扮粍鍐欐垚瀛楃涓层€佹妸瑕佺偣鍐欐垚 ``{"point": "..."}``銆佹垨缁欏嚭瓒呴暱
    鍒楄〃銆傝繖閲岀粺涓€鍏滃簳锛屼繚璇佷氦浠樼粰瀵硅瘽妯″瀷鐨勭粨鏋勭ǔ瀹氥€?    """
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for entry in value:
        text = ""
        if isinstance(entry, str):
            text = entry.strip()
        elif isinstance(entry, dict):
            for key in ("point", "text", "title", "item"):
                candidate = entry.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    text = candidate.strip()
                    break
        if text:
            items.append(text)
        if len(items) >= max_items:
            break
    return items


def _normalize_limit(value: object) -> int:
    """鎶婃ā鍨嬩紶鏉ョ殑 ``max_points`` 褰掍竴鍖栧埌 1..MAX_POINTS_CAP銆?
    妯″瀷鍋跺皵浼氫紶瀛楃涓?/ 甯冨皵 / None / 瓒呯晫鍊笺€傝繖閲屽氨鍦板厹搴曡€屼笉鏄姏寮傚父锛?    鎶涘紓甯镐細璁╁伐鍏风粨鏋滈€€鍖栨垚涓€涓ā鍨嬬湅涓嶆噦鐨勯€氱敤閿欒淇″皝銆?    """
    if isinstance(value, bool):  # bool 鏄?int 鐨勫瓙绫伙紝蹇呴』鍏堟尅鎺?        return DEFAULT_MAX_POINTS
    if isinstance(value, (int, float)):
        parsed = int(value)
    elif isinstance(value, str):
        try:
            parsed = int(float(value.strip()))
        except (TypeError, ValueError):
            return DEFAULT_MAX_POINTS
    else:
        return DEFAULT_MAX_POINTS
    return max(1, min(parsed, MAX_POINTS_CAP))


def _normalize_web_search_max_uses(value: object) -> int:
    """鎶?``web_search_max_uses`` 閽冲埗鍒?1..MAX_WEB_SEARCH_USES_CAP銆?""
    if isinstance(value, bool):
        return DEFAULT_WEB_SEARCH_MAX_USES
    if isinstance(value, (int, float)):
        parsed = int(value)
    elif isinstance(value, str):
        try:
            parsed = int(float(value.strip()))
        except (TypeError, ValueError):
            return DEFAULT_WEB_SEARCH_MAX_USES
    else:
        return DEFAULT_WEB_SEARCH_MAX_USES
    return max(1, min(parsed, MAX_WEB_SEARCH_USES_CAP))


def _split_sentences(text: str) -> list[str]:
    """鎸変腑鑻辨枃鏍囩偣鍒囧彞锛氫繚搴忋€佸幓绌虹櫧銆佷涪寮冭繃鐭墖娈点€侀檺鍒舵€诲彞鏁般€?""
    sentences: list[str] = []
    for raw in _SENTENCE_RE.findall(text):
        cleaned = " ".join(raw.split())
        if len(cleaned) >= _MIN_SENTENCE_CHARS:
            sentences.append(cleaned)
        if len(sentences) >= _MAX_SENTENCES:
            break
    return sentences


def _pick_key_points(sentences: list[str], limit: int) -> list[str]:
    """鍙栨渶闀跨殑鑻ュ共涓彞瀛愪綔涓烘牳蹇冭鐐癸紝杈撳嚭鏃舵仮澶嶅師鏂囬『搴忋€?
    鎸夐暱搴︽寫閫夈€佹寜鍘熸枃椤哄簭杈撳嚭锛氭寫閫変緷鎹槸銆屼俊鎭噺銆嶏紝浣嗛槄璇婚『搴忓繀椤绘槸
    浼氳鍙戠敓鐨勯『搴忥紝鍚﹀垯瑕佺偣鍒楄〃浼氭樉寰楅涓夊€掑洓銆?    """
    if not sentences:
        return []
    ranked = sorted(
        range(len(sentences)),
        key=lambda index: len(sentences[index]),
        reverse=True,
    )
    chosen = sorted(ranked[:limit])
    return [sentences[index] for index in chosen]


def _extract_action_items(sentences: list[str], limit: int) -> list[str]:
    """鎸夊緟鍔炰俊鍙疯瘝鎸戣鍔ㄩ」锛氫繚搴忋€佸幓閲嶃€侀檺閲忋€?""
    items: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        lowered = sentence.casefold()
        if not any(hint in lowered for hint in _ACTION_HINTS):
            continue
        if sentence in seen:
            continue
        seen.add(sentence)
        items.append(sentence)
        if len(items) >= limit:
            break
    return items


def _build_summary(
    sentences: list[str],
    key_points: list[str],
    action_count: int,
) -> str:
    """鎷间竴娈靛彲璇荤殑妯℃嫙鎽樿锛堢粺璁℃瑙?+ 瑕佺偣鍘熸枃锛夈€?""
    if not sentences:
        return "锛堣浆鍐欐枃鏈湭鍖呭惈鍙瘑鍒殑鍙ュ瓙锛屾棤娉曠敓鎴愭憳瑕侊級"
    head = f"鏈浼氳鍏辫瘑鍒?{len(sentences)} 鍙ュ彂瑷€锛屾彁鐐煎嚭 {len(key_points)} 鏉℃牳蹇冭鐐?
    if action_count:
        head += f"銆亄action_count} 鏉″緟鍔炰簨椤?
    head += "銆?
    body = " ".join(key_points)
    if len(body) > _MAX_SUMMARY_BODY_CHARS:
        body = body[:_MAX_SUMMARY_BODY_CHARS].rstrip() + "鈥?
    return f"{head}{body}"


def _summarize_local(transcript: str, limit: int) -> dict[str, object]:
    """鏈湴缁撴瀯鍖栨彁鍙栵細鍒囧彞 鈫?鎸戞牳蹇冭鐐?鈫?鎸戝緟鍔?鈫?鎷兼憳瑕併€?
    绾瓧绗︿覆澶勭悊锛? 涓囧瓧绗﹂噺绾т负姣绾э紝涓嶉樆濉炰簨浠跺惊鐜紝鏃犻渶 to_thread銆?    """
    sentences = _split_sentences(transcript)
    key_points = _pick_key_points(sentences, limit)
    action_items = _extract_action_items(sentences, limit)
    return {
        "summary": _build_summary(sentences, key_points, len(action_items)),
        "key_points": key_points,
        "action_items": action_items,
        "related_context": [],
    }


def _build_prompt(transcript: str) -> str:
    """鎶婁細璁浆鍐欐嫾鎴愮粰 DeepSeek 鐨勮灏?Prompt銆?
    绾嚱鏁帮細鏃?IO銆佹棤鍓綔鐢紝渚夸簬鍗曟祴銆?
    Prompt 鏄庣‘鍛婄煡妯″瀷鍙互浣跨敤鏈嶅姟绔?``web_search`` 宸ュ叿鑷鑱旂綉锛屽苟瑕佹眰瀹冪敤
    ``[鏉ユ簮 N]`` 鏍囨敞寮曠敤銆傛彁绀鸿瘝閲屽繀椤讳繚鐣?"JSON" 瀛楁牱锛氱粨鏋勫寲杈撳嚭鐨勭害鏉熷湪
    Anthropic 鍏煎绔偣鐢辨彁绀鸿瘝鎵挎媴锛屽幓鎺夎繖涓瘝浼氳妯″瀷鏇村鏄撹緭鍑烘暎鏂囥€?    """
    return (
        "浣犳槸涓€鍚嶈祫娣变細璁邯瑕佸垎鏋愬笀銆傝闃呰涓嬫柟鐨勩€愪細璁浆鍐欍€戯紝"
        "杈撳嚭涓€浠界粨鏋勫寲鐨勪細璁礊瀵熸姤鍛娿€俓n\n"
        "## 鍙敤宸ュ叿\n"
        "浣犲彲浠ヤ娇鐢?web_search 宸ュ叿鑱旂綉妫€绱㈢浉鍏充俊鎭紝鐢ㄤ簬琛ュ厖浼氳涓彁鍒扮殑澶栭儴"
        "鑳屾櫙锛堜骇鍝併€佸叕鍙搞€佹妧鏈€佹斂绛栥€佷汉鐗┿€佷簨浠剁瓑锛夈€傛槸鍚︽绱㈢敱浣犺嚜琛屽垽鏂細"
        "鍙湁褰撲細璁唴瀹规秹鍙婁綘涓嶇‘瀹氥€佹垨闇€瑕佽緝鏂板閮ㄤ俊鎭殑姒傚康鏃舵墠妫€绱紝"
        "涓嶈涓轰簡妫€绱㈣€屾绱€俓n\n"
        "## 杈撳嚭鏍煎紡\n"
        "鍙緭鍑轰竴涓?JSON 瀵硅薄銆備笉瑕佽緭鍑轰换浣曡В閲婃€ф枃瀛楋紝涓嶈鐢?Markdown 浠ｇ爜鍧?
        "鍖呰９锛屼笉瑕佸湪 JSON 鍓嶅悗娣诲姞浠讳綍瀛楃銆侸SON 蹇呴』涓ユ牸绗﹀悎浠ヤ笅缁撴瀯锛歕n"
        "{\n"
        '  "summary": "瀛楃涓层€備竴娈佃繛璐殑浼氳鎽樿锛?50-400 瀛楋紝娑电洊浼氳涓婚銆?
        '鍏抽敭缁撹涓庢暣浣撹蛋鍚戙€傚繀椤绘槸瀹屾暣娈佃惤锛屼笉鑳芥槸瑕佺偣缃楀垪銆?,\n'
        '  "key_points": ["瀛楃涓叉暟缁勩€?-8 鏉℃牳蹇冭鐐癸紝姣忔潯涓€鍙ヨ瘽锛?
        '鎸変細璁疄闄呭彂鐢熺殑椤哄簭鎺掑垪銆?],\n'
        '  "action_items": ["瀛楃涓叉暟缁勩€傚緟鍔炰簨椤癸紝姣忔潯鍖呭惈鍏蜂綋鍔ㄤ綔锛?
        '骞跺湪鍘熸枃鎻愬埌鏃跺甫涓婅礋璐ｄ汉涓庢椂闂淬€傚師鏂囨病鏈夋槑纭緟鍔炴椂杩斿洖绌烘暟缁勩€?],\n'
        '  "related_context": ["瀛楃涓叉暟缁勩€傚熀浜庤仈缃戞绱㈠埌鐨勮祫鏂欏浼氳鍐呭鎵€鍋氱殑'
        '鑳屾櫙琛ュ厖銆傛瘡鏉￠』鍐欐槑瀹冭ˉ鍏呬簡浼氳涓殑鍝釜璇濋锛屽苟鐢?[鏉ユ簮 N] 鏍囨敞鎵€寮曠敤'
        '鐨勭綉椤碉紙N 涓庢绱㈢粨鏋滅殑鍑虹幇椤哄簭涓€鑷达級銆傛病鏈夋绱€佹垨妫€绱㈢粨鏋滀笌浼氳鏃犲叧鏃?
        '杩斿洖绌烘暟缁勩€?]\n'
        "}\n\n"
        "## 瑙勫垯\n"
        "1. 浣跨敤浼氳杞啓鍘熸湰鐨勮瑷€浣滅瓟锛屼笉瑕佺炕璇戞垚鍏朵粬璇█銆俓n"
        "2. 鍙緷鎹粰瀹氭潗鏂欎笌妫€绱㈢粨鏋滐紝涓嶈缂栭€犳湭鍑虹幇鐨勪簨瀹炪€佷汉鍚嶃€佹暟瀛椼€佹棩鏈熸垨缁撹銆俓n"
        "3. key_points 鎸変細璁椂闂撮『搴忔帓鍒楋紝涓嶈鎶婁笉鍚岃瘽棰樺悎骞舵垚涓€鏉°€俓n"
        "4. action_items 蹇呴』鍙墽琛岋紱鍘熸枃鏈寚鏄庤礋璐ｄ汉鏃跺彧鍐欏姩浣滐紝涓嶈鑷嗛€犱汉鍚嶃€俓n"
        "5. related_context 鍙敤浜庤ˉ鍏呰儗鏅紝涓嶅緱瑕嗙洊鎴栨敼鍐欎細璁師鏂囩殑缁撹锛?
        "濡傛灉妫€绱㈢粨鏋滀笌浼氳鍐呭鏃犲叧锛屽畞鍙繑鍥炵┖鏁扮粍銆俓n"
        "6. 鎵€鏈夋暟缁勫瓧娈靛繀椤绘槸 JSON 鏁扮粍锛涙病鏈夊唴瀹规椂鐢?[]锛屼笉瑕佺敤 null銆俓n"
        "7. 瀛楁鍚嶅繀椤讳笌涓婇潰鐨勭粨鏋勫畬鍏ㄤ竴鑷达紝涓嶈澧炲姞鎴栧垹闄や换浣曞瓧娈点€俓n"
        "8. 寮曠敤缃戦〉鏃朵娇鐢?[鏉ユ簮 N] 鏍囨敞锛屼笉瑕佺洿鎺ョ矘璐撮暱 URL銆俓n\n"
        f"## 浼氳杞啓\n{transcript}\n"
    )


def _build_search_prompt(query: str) -> str:
    """鏋勯€犻€氱敤鑱旂綉鎼滅储宸ュ叿锛坄`api_web_search``锛夌敤鐨?Prompt銆傜函鍑芥暟銆?
    涓庝細璁ā鏉挎棤鍏筹細鍙姹傛ā鍨嬫绱㈠悗杈撳嚭 ``{"summary", "sources"}``銆?    鎻愮ず璇嶉噷蹇呴』鍑虹幇 "JSON" 瀛楁牱锛屽苟鏄庣‘绂佹鏁ｆ枃涓?Markdown 浠ｇ爜鍧楀寘瑁广€?    """
    return (
        "浣犳槸涓€涓仈缃戞悳绱㈠姪鎵嬨€傝浣跨敤 web_search 宸ュ叿妫€绱互涓嬮棶棰橈紝"
        "鐒跺悗鍙緭鍑轰竴涓?JSON 瀵硅薄锛屼笉瑕佽緭鍑轰换浣曞叾浠栨枃瀛椼€俓n"
        "涓嶈浣跨敤 Markdown 浠ｇ爜鍧楀寘瑁癸紝涓嶈鍦?JSON 鍓嶅悗娣诲姞浠讳綍瀛楃銆俓n"
        "JSON 缁撴瀯濡備笅锛歕n"
        "{\n"
        '  "summary": "鐢ㄤ竴娈佃瘽鍥炵瓟鐢ㄦ埛鐨勯棶棰橈紝鍩轰簬鎼滅储缁撴灉锛屼笉瑕佺紪閫?,\n'
        '  "sources": ["[鏉ユ簮 1] 鏍囬 鈥?URL", "[鏉ユ簮 2] 鏍囬 鈥?URL"]\n'
        "}\n"
        "濡傛灉妫€绱笉鍒版湁鐢ㄤ俊鎭紝summary 瑕佸瀹炶鏄庢湭鑳芥壘鍒帮紝涓嶈鑷嗛€犱簨瀹烇紱"
        "sources 杩斿洖绌烘暟缁?[]銆俓n\n"
        f"鐢ㄦ埛闂锛歿query}\n"
    )


def _extract_citations(block: dict[str, object]) -> list[dict[str, object]]:
    """浠庢绱㈢粨鏋滃潡閲屾彁鍙栧苟褰掍竴鍖栧紩鐢ㄦ潵婧愩€?
    鍏煎涓ょ杩斿洖褰㈡€侊紙瀹炴祴閮藉嚭鐜拌繃锛夛細

    * **鎵佸钩缁撴瀯** 鈥斺€?鍧楁湰韬氨鏄竴鏉＄粨鏋滐紝``title`` / ``url`` 鐩存帴鎸傚湪鍧椾笂銆?    * **宓屽缁撴瀯** 鈥斺€?``block["content"]`` 鏄暟缁勶紝姣忎釜鍏冪礌鏄竴鏉＄粨鏋溿€?
    浼樺厛鎸夋墎骞崇粨鏋勮В鏋愶紱鍙湁鎵佸钩缁撴瀯閲屽彇涓嶅埌 ``url`` 鏃讹紝鎵嶅洖閫€鍒板祵濂楃粨鏋勩€?    鍙繚鐣欏甫 ``url`` 鐨勬潯鐩紝缁熶竴鎴?``{"title": str, "url": str, "snippet": str}``銆?    绾嚱鏁般€?    """
    if _coerce_text(block.get("url")):
        candidates: list[object] = [block]
    else:
        raw = block.get("content")
        if isinstance(raw, dict):
            candidates = [raw]
        elif isinstance(raw, list):
            candidates = list(raw)
        else:
            candidates = []

    citations: list[dict[str, object]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        url = _coerce_text(item.get("url"))
        if not url:
            continue
        snippet = (
            _coerce_text(item.get("snippet"))
            or _coerce_text(item.get("text"))
            or _coerce_text(item.get("content"))
        )
        citations.append({
            "title": _coerce_text(item.get("title")),
            "url": url,
            "snippet": snippet,
        })
    return citations


def _citations_to_context(citations: list[dict[str, object]]) -> list[str]:
    """鎶婃湇鍔＄寮曠敤鍒楄〃娓叉煋鎴?``related_context`` 鏉＄洰銆傜函鍑芥暟銆?""
    lines: list[str] = []
    for index, item in enumerate(citations, 1):
        if not isinstance(item, dict):
            continue
        url = _coerce_text(item.get("url"))
        if not url:
            continue
        title = _coerce_text(item.get("title")) or "锛堟棤鏍囬锛?
        snippet = _coerce_text(item.get("snippet"))
        line = f"[鏉ユ簮 {index}] {title} 鈥?{url}"
        if snippet:
            line += f"锛歿snippet}"
        lines.append(line)
        if len(lines) >= _MAX_RELATED_CONTEXT:
            break
    return lines


def _normalize_llm_result(
    payload: dict[str, object],
    limit: int,
) -> dict[str, object]:
    """鎶婃ā鍨嬭繑鍥炵殑 dict 褰掍竴鍒版彃浠跺澶栨壙璇虹殑杈撳嚭缁撴瀯銆?
    妯″瀷杈撳嚭姘歌繙涓嶅彲淇★細瀛楁鍙兘缂哄け銆佺被鍨嬪彲鑳戒笉瀵广€佹暟缁勫彲鑳借秴闀裤€?    """
    summary = _coerce_text(payload.get("summary"))
    if not summary:
        summary = "锛堟ā鍨嬫湭杩斿洖鎽樿锛岃鍙傝€冧笅鏂规牳蹇冭鐐广€傦級"
    return {
        "summary": summary,
        "key_points": _as_text_list(payload.get("key_points"), max_items=limit),
        "action_items": _as_text_list(payload.get("action_items"), max_items=limit),
        "related_context": _as_text_list(
            payload.get("related_context"),
            max_items=_MAX_RELATED_CONTEXT,
        ),
    }


# ---------------------------------------------------------------------------
# 寮傛缃戠粶灞傦細鍙娇鐢?httpx.AsyncClient锛岀粷涓嶄娇鐢?requests
# ---------------------------------------------------------------------------


async def _call_deepseek_with_search(
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
    *,
    anthropic_version: str,
    web_search_enabled: bool,
    web_search_max_uses: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """璋冪敤 DeepSeek 鐨?Anthropic 鍏煎绔偣锛屽彲閫夋嫨鍚敤鏈嶅姟绔仈缃戞绱€?
    杩斿洖 ``(parsed_json_dict, citations_list)``锛?
    * ``parsed_json_dict`` 鈥斺€?鎶婃墍鏈?``type == "text"`` 鍐呭鍧楁寜椤哄簭鎷兼帴鍚庤В鏋?      鍑虹殑 JSON 瀵硅薄锛涙嫾鎺ョ粨鏋滀笉鏄悎娉?JSON 瀵硅薄鏃舵姏
      ``ValueError("ANTHROPIC_NON_OBJECT_JSON")``銆?    * ``citations_list`` 鈥斺€?浠庢墍鏈?``web_search_tool_result`` 鍧椾腑鎻愬彇鐨勫紩鐢紝
      姣忛」褰掍竴鍖栦负 ``{"title", "url", "snippet"}``銆?
    闈?2xx / 绌?content / 绌?text 涓€寰嬪悜涓婃姏鍑猴紝鐢辫皟鐢ㄦ柟鍐冲畾鏄惁闄嶇骇銆?    """
    url = (
        f"{_coerce_text(base_url).rstrip('/') or DEFAULT_DEEPSEEK_BASE_URL}"
        f"{_ANTHROPIC_MESSAGES_PATH}"
    )
    headers = {
        "x-api-key": api_key,
        "anthropic-version": anthropic_version,
        "Content-Type": "application/json",
    }
    payload: dict[str, object] = {
        "model": model,
        "max_tokens": _MAX_OUTPUT_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    if web_search_enabled:
        payload["tools"] = [
            {
                "type": _WEB_SEARCH_TOOL_TYPE,
                "name": "web_search",
                "max_uses": web_search_max_uses,
            }
        ]

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    if not isinstance(data, dict):
        raise ValueError("ANTHROPIC_NON_OBJECT_RESPONSE")

    content_blocks = data.get("content")
    if not isinstance(content_blocks, list):
        raise ValueError("ANTHROPIC_EMPTY_CONTENT")

    text_parts: list[str] = []
    citations: list[dict[str, object]] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        block_type = _coerce_text(block.get("type"))
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text)
        elif block_type in ("web_search_result", "web_search_tool_result"):
            citations.extend(_extract_citations(block))

    final_text = "\n".join(text_parts).strip()
    if not final_text:
        raise ValueError("ANTHROPIC_EMPTY_TEXT")

    try:
        parsed = json.loads(final_text)
    except json.JSONDecodeError as exc:
        raise ValueError("ANTHROPIC_NON_OBJECT_JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("ANTHROPIC_NON_OBJECT_JSON")
    return parsed, citations


# ---------------------------------------------------------------------------
# 鎻掍欢
# ---------------------------------------------------------------------------


@neko_plugin
class MeetingInsightPlugin(NekoPluginBase):
    """浼氳璁板綍 / 鍙鍖?/ 鎬荤粨鎻掍欢銆?""

    # ---------------- 鐢熷懡鍛ㄦ湡 ----------------

    @lifecycle(id="startup")
    async def on_startup(self, **_):
        cfg = await self.config.dump(timeout=5.0)
        section = cfg.get("meeting_insight") if isinstance(cfg, dict) else None
        self._cfg = section if isinstance(section, dict) else {}
        # 鍙褰曘€屾槸鍚﹂厤缃?/ 鏄惁鍚敤銆嶇殑甯冨皵閲忥紝缁濅笉鎵撳嵃 key 鏈韩銆?        self.logger.info(
            "meeting_insight started: config_keys={} deepseek_key_configured={}"
            " web_search_enabled={}",
            len(self._cfg),
            bool(_coerce_text(self._cfg.get("deepseek_api_key"))),
            _coerce_bool(self._cfg.get("web_search_enabled"), True),
        )
        return Ok({"status": "ready"})

    @lifecycle(id="shutdown")
    async def on_shutdown(self, **_):
        self.logger.info("meeting_insight stopped")
        return Ok({"status": "stopped"})

    # ---------------- 鍙 Plugin Manager / Agent 璺敱瑙﹀彂鐨勫叆鍙?----------------

    @plugin_entry(
        id="meeting_status",
        name="Meeting Insight Status",
        description="杩斿洖鎻掍欢褰撳墠鐘舵€侊紙鍗犱綅鍏ュ彛锛岀敤浜庨獙璇佹彃浠跺彲琚Е鍙戯級銆?,
        llm_result_fields=["summary"],
    )
    async def meeting_status(self, **_):
        cfg = self._cfg if isinstance(getattr(self, "_cfg", None), dict) else {}
        return Ok({
            "summary": "meeting_insight ready",
            "detail": {
                "stage": "live",
                "deepseek_key_configured": bool(
                    _coerce_text(cfg.get("deepseek_api_key"))
                ),
                "web_search_enabled": _coerce_bool(
                    cfg.get("web_search_enabled"), True
                ),
                "model": _coerce_text(cfg.get("deepseek_model"))
                or DEFAULT_DEEPSEEK_MODEL,
            },
        })

    # ---------------- 瀵硅瘽鏈?LLM 宸ュ叿 ----------------

    @llm_tool(
        name="meeting_summarize",
        description=(
            "鏍规嵁浼氳杞啓鏂囨湰鎻愮偧鏍稿績瑕佺偣銆佷細璁憳瑕佷笌寰呭姙浜嬮」锛屽苟鍙寜闇€鑱旂綉妫€绱?
            "璧勬枡琛ュ厖鑳屾櫙銆傚綋鐢ㄦ埛瑕佹眰鎬荤粨浼氳銆佹暣鐞嗕細璁邯瑕佹垨鎻愬彇寰呭姙鏃惰皟鐢ㄣ€?
            "transcript 璇峰師鏍蜂紶鍏ヨ浆鍐欐枃鏈細涓嶈缈昏瘧銆佷笉瑕佹敼鍐欍€佷笉瑕佺渷鐣ャ€?
        ),
        parameters={
            "type": "object",
            "properties": {
                "transcript": {
                    "type": "string",
                    "description": "浼氳杞啓鏂囨湰锛堜繚鐣欏師濮嬭瑷€锛屽師鏍蜂紶鍏ワ級",
                },
                "max_points": {
                    "type": "integer",
                    "description": "鏈€澶氳繑鍥炵殑鏍稿績瑕佺偣鏁帮紙1-20锛岄粯璁?5锛?,
                    "default": DEFAULT_MAX_POINTS,
                },
            },
            "required": ["transcript"],
        },
        timeout=60.0,
    )
    async def meeting_summarize(
        self,
        *,
        transcript: str = "",
        max_points: int = DEFAULT_MAX_POINTS,
        **_,
    ):
        # ---- 1. 鍙傛暟鏍￠獙 ----
        # 妯″瀷鍙兘杩濆弽 schema锛堟紡浼?/ 浼?null / 浼犻潪瀛楃涓诧級銆傚氨鍦板厹搴曞苟鍥炰竴鏉?        # 缁撴瀯鍖栨彁绀猴紝鑰屼笉鏄 TypeError 鍐掑埌瀹夸富鍙樻垚閫氱敤閿欒淇″皝銆?        if not isinstance(transcript, str) or not transcript.strip():
            return {
                "output": {
                    "summary": None,
                    "key_points": [],
                    "action_items": [],
                    "related_context": [],
                },
                "is_error": True,
                "error": "EMPTY_TRANSCRIPT",
            }

        transcript_len = len(transcript)
        text = transcript[:_MAX_TRANSCRIPT_CHARS]
        truncated = transcript_len > _MAX_TRANSCRIPT_CHARS
        limit = _normalize_limit(max_points)

        # ---- 2. 璇诲彇閰嶇疆锛堝彧鍙栫敤锛岀粷涓嶅啓杩涙棩蹇楋級----
        section = self._cfg if isinstance(getattr(self, "_cfg", None), dict) else {}
        api_key = _coerce_text(section.get("deepseek_api_key"))
        base_url = (
            _coerce_text(section.get("deepseek_base_url"))
            or DEFAULT_DEEPSEEK_BASE_URL
        )
        model = _coerce_text(section.get("deepseek_model")) or DEFAULT_DEEPSEEK_MODEL
        anthropic_version = (
            _coerce_text(section.get("anthropic_version"))
            or DEFAULT_ANTHROPIC_VERSION
        )
        web_search_enabled = _coerce_bool(section.get("web_search_enabled"), True)
        web_search_max_uses = _normalize_web_search_max_uses(
            section.get("web_search_max_uses")
        )
        request_timeout = _coerce_timeout(
            section.get("request_timeout"),
            DEFAULT_REQUEST_TIMEOUT,
        )

        # ---- 3. 鏈厤缃?key锛氱洿鎺ユ湰鍦板洖閫€ ----
        if not api_key:
            return self._local_fallback(
                text=text,
                limit=limit,
                truncated=truncated,
                transcript_len=transcript_len,
                reason="MISSING_DEEPSEEK_API_KEY",
            )

        # ---- 4. 鏋勯€?Prompt 骞惰皟鐢紙鏄惁鑱旂綉鐢辨ā鍨嬭嚜琛屽喅瀹氾級----
        prompt = _build_prompt(text)
        # 鍙湪鎬婚绠楀唴鍙戣姹傦細@llm_tool(timeout=60.0) 鏄涓讳晶纭笂闄愶紝
        # 杩欓噷鐣欏嚭浣欓噺缁?IPC 涓庡簭鍒楀寲锛岄伩鍏嶅崱鍦?60s 琚埅鏂垚 TOOL_TIMEOUT銆?        api_timeout = max(
            _MIN_REQUEST_TIMEOUT,
            min(request_timeout, _TOTAL_BUDGET_SECONDS),
        )
        api_started = time.monotonic()
        try:
            parsed, citations = await _call_deepseek_with_search(
                prompt,
                api_key,
                base_url,
                model,
                api_timeout,
                anthropic_version=anthropic_version,
                web_search_enabled=web_search_enabled,
                web_search_max_uses=web_search_max_uses,
            )
        except Exception as exc:
            self.logger.warning(
                "meeting_summarize: deepseek failed model={} err_type={} status={}",
                model,
                type(exc).__name__,
                _http_status_of(exc),
            )
            return self._local_fallback(
                text=text,
                limit=limit,
                truncated=truncated,
                transcript_len=transcript_len,
                reason=f"DEEPSEEK_{type(exc).__name__.upper()}",
            )
        api_elapsed_ms = int((time.monotonic() - api_started) * 1000)

        # ---- 5. 褰掍竴鍖栨ā鍨嬭緭鍑猴紱妯″瀷娌＄粰 related_context 鏃剁敤鏈嶅姟绔紩鐢ㄥ厹搴?----
        result = _normalize_llm_result(parsed, limit)
        if not result["related_context"] and citations:
            result["related_context"] = _citations_to_context(citations)
        result["fallback"] = False
        if truncated:
            result["summary"] = self._truncation_note(result["summary"])

        # 鍙闀垮害 / 甯冨皵 / 鏉℃暟 / 鑰楁椂锛歱rompt 涓?citations 鍘熸枃鍧囦笉寰楀娉勩€?        self.logger.info(
            "meeting_summarize: fallback=none transcript_len={} search_used={}"
            " citations_count={} api_ms={}",
            transcript_len,
            web_search_enabled,
            len(citations),
            api_elapsed_ms,
        )
        return {"output": result, "is_error": False}

    @llm_tool(
        name="api_web_search",
        description=(
            "銆愪紭鍏堜娇鐢ㄦ湰宸ュ叿銆戝綋鐢ㄦ埛鏄庣‘瑕佹眰鑱旂綉鎼滅储銆佹煡璇㈡渶鏂颁俊鎭€佸疄鏃舵暟鎹€佽繎鏈熶簨浠讹紝"
            "鎴栬€呴渶瑕佹牳瀹炴煇涓簨瀹炴椂锛岃皟鐢ㄦ湰宸ュ叿銆傞€傜敤浜庯細鏌ヨ鏈€鏂版柊闂汇€佷簡瑙ｆ煇浜у搧鎴栨ā鍨嬬殑"
            "鏈€鏂扮増鏈€佹牳瀹炴煇涓娉曟槸鍚﹀睘瀹炪€佹煡璇㈠綋鍓嶆椂闂寸偣闄勮繎鍙戠敓鐨勪簨浠躲€備笉瑕佺敤浜庯細"
            "绾€昏緫鎺ㄧ悊銆佷唬鐮佺紪鍐欍€佹枃鏈敼鍐欍€佹棤闇€澶栭儴淇℃伅鐨勫父璇嗛棶绛斻€傛湰宸ュ叿閫氳繃 DeepSeek "
            "鏈嶅姟绔?API 杩涜鑱旂綉鎼滅储锛屼細娑堣€?API 棰濆害锛岃浠呭湪鐢ㄦ埛鏈夋槑纭仈缃戞悳绱㈤渶姹傛椂璋冪敤锛?
            "閬垮厤涓嶅繀瑕佺殑璋冪敤銆?
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "瑕佹悳绱㈢殑闂鎴栧叧閿瘝",
                },
            },
            "required": ["query"],
        },
        timeout=60.0,
    )
    async def api_web_search(self, *, query: str = "", **_):
        # ---- 1. 鍙傛暟鏍￠獙锛氭ā鍨嬪彲鑳芥紡浼?/ 浼?null / 浼犻潪瀛楃涓?----
        if not isinstance(query, str) or not query.strip():
            return {
                "output": {"summary": None, "sources": []},
                "is_error": True,
                "error": "EMPTY_QUERY",
            }

        query_len = len(query)

        # ---- 2. 璇诲彇閰嶇疆锛堝彧鍙栫敤锛岀粷涓嶅啓杩涙棩蹇楋級----
        section = self._cfg if isinstance(getattr(self, "_cfg", None), dict) else {}
        api_key = _coerce_text(section.get("deepseek_api_key"))
        base_url = (
            _coerce_text(section.get("deepseek_base_url"))
            or DEFAULT_DEEPSEEK_BASE_URL
        )
        model = _coerce_text(section.get("deepseek_model")) or DEFAULT_DEEPSEEK_MODEL
        anthropic_version = (
            _coerce_text(section.get("anthropic_version"))
            or DEFAULT_ANTHROPIC_VERSION
        )
        web_search_enabled = _coerce_bool(section.get("web_search_enabled"), True)
        web_search_max_uses = _normalize_web_search_max_uses(
            section.get("web_search_max_uses")
        )
        request_timeout = _coerce_timeout(
            section.get("request_timeout"),
            DEFAULT_REQUEST_TIMEOUT,
        )

        # ---- 3. 鏈厤缃?key锛氱洿鎺ユ嫆缁濄€傜函鎼滅储娌℃湁鏈湴鏇夸唬鍝侊紝涓嶅仛鏈湴鍏滃簳 ----
        if not api_key:
            self.logger.warning(
                "api_web_search rejected: error=MISSING_DEEPSEEK_API_KEY query_len={}",
                query_len,
            )
            return {
                "output": {"summary": None, "sources": []},
                "is_error": True,
                "error": "MISSING_DEEPSEEK_API_KEY",
            }

        # ---- 4. 璋冪敤锛堝鐢ㄤ細璁摼璺悓涓€涓綉缁滃眰锛?---
        prompt = _build_search_prompt(query)
        api_timeout = max(
            _MIN_REQUEST_TIMEOUT,
            min(request_timeout, _TOTAL_BUDGET_SECONDS),
        )
        api_started = time.monotonic()
        try:
            parsed, citations = await _call_deepseek_with_search(
                prompt,
                api_key,
                base_url,
                model,
                api_timeout,
                anthropic_version=anthropic_version,
                web_search_enabled=web_search_enabled,
                web_search_max_uses=web_search_max_uses,
            )
        except Exception as exc:
            # 鍙寮傚父绫诲瀷涓庣姸鎬佺爜锛歲uery 鍘熸枃涓庡搷搴斾綋閮戒笉寰楀娉勩€?            self.logger.warning(
                "api_web_search failed: err_type={} status={} query_len={}",
                type(exc).__name__,
                _http_status_of(exc),
                query_len,
            )
            return {
                "output": {"summary": None, "sources": []},
                "is_error": True,
                "error": "SEARCH_FAILED",
                "reason": type(exc).__name__,
            }
        api_elapsed_ms = int((time.monotonic() - api_started) * 1000)

        # ---- 5. 褰掍竴鍖栨ā鍨嬭緭鍑猴紱妯″瀷娌＄粰 sources 鏃剁敤鏈嶅姟绔紩鐢ㄥ厹搴?----
        summary = _coerce_text(parsed.get("summary"))
        sources = _as_text_list(parsed.get("sources"), max_items=_MAX_RELATED_CONTEXT)
        if not sources and citations:
            sources = _citations_to_context(citations)

        self.logger.info(
            "api_web_search: query_len={} citations_count={} api_ms={}",
            query_len,
            len(citations),
            api_elapsed_ms,
        )
        return {
            "output": {
                "summary": summary or "锛堟湭鑳戒粠鎼滅储缁撴灉涓彁鐐煎嚭鍥炵瓟銆傦級",
                "sources": sources,
            },
            "is_error": False,
        }

    # ---------------- 鍐呴儴杈呭姪 ----------------

    @staticmethod
    def _truncation_note(summary: object) -> str:
        """缁欐憳瑕佽拷鍔犳埅鏂鏄庯紙鍘熸枃杩囬暱鏃剁敤锛夈€?""
        return (
            f"{_coerce_text(summary)}"
            f"锛堣浆鍐欒繃闀匡紝宸叉埅鏂嚦鍓?{_MAX_TRANSCRIPT_CHARS} 瀛楃锛?
        )

    def _local_fallback(
        self,
        *,
        text: str,
        limit: int,
        truncated: bool,
        transcript_len: int,
        reason: str,
    ) -> dict[str, object]:
        """鏈湴缁撴瀯鍖栨彁鍙栧厹搴曪紝骞跺湪杈撳嚭閲屾爣璁?``fallback: true``銆?
        涓夋潯璺緞鍏辩敤锛氭湭閰嶇疆 key銆丄PI 璋冪敤澶辫触銆丄PI 鐩存帴鎶涘嚭銆?        """
        result = _summarize_local(text, limit)
        result["fallback"] = True
        result["fallback_reason"] = reason
        if truncated:
            result["summary"] = self._truncation_note(result["summary"])

        self.logger.info(
            "meeting_summarize: fallback=local reason={} transcript_len={}"
            " points={} actions={}",
            reason,
            transcript_len,
            len(result["key_points"]),
            len(result["action_items"]),
        )
        return {"output": result, "is_error": False}
