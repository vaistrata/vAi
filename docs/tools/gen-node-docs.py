#!/usr/bin/env python3
"""vai/nodes/*.h 의 헤더 스탬프에서 06-nodes.md 를 언어별로 생성한다.

출력은 docs/ko/06-nodes.md 와 docs/en/06-nodes.md 두 개다. 노드 요약·서술·시그니처
본문은 헤더 스탬프에서 오고 스탬프는 영어이므로, 두 문서의 차이는 STRINGS 의 라벨뿐이다.

진실은 헤더에 있고 이 문서들은 그 투영이다. 손으로 06-nodes.md 를 고치지 말 것 —
--check 가 CI 에서 두 파일 모두의 드리프트를 잡는다.

읽는 것:
  - 스탬프  `/*==== */` 블록 안의 `<Name> - <요약>` 과 Input/Params/Output 시그니처
  - 슬롯    생성자의 addSlot("name", NodeSlot::kind) 호출 (주석보다 정확)
  - Config struct 필드 (타입/이름/기본값/줄끝 주석)
  - 본문에서 언급된 셰이더 이름

사용법:
  python docs/tools/gen-node-docs.py            생성
  python docs/tools/gen-node-docs.py --check    커밋본과 다르면 실패 (CI)

mkdocs serve 가 떠 있으면 헤더 스탬프를 고치고 저장하는 순간 갱신된다.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NODE_DIR = ROOT / "vai" / "nodes"
SHADER_DIR = ROOT / "vai" / "shaders"

# 언어별 출력 라벨. 노드 요약·서술·시그니처 본문은 헤더 스탬프에서 오므로 여기에 없다.
# sig_labels 의 값은 스탬프 표기(Input:/Output:/Params:)를 그대로 따른다.
STRINGS = {
    "ko": {
        "title": "제공 노드들",
        "intro": "빌트인 노드 {n}종. 개념은 [노드(Node)](03-usage/node.md)를, "
        "새 노드를 만드는 법은 [커스텀 노드와 커널](07-contributing-nodes.md)을 참조하세요.",
        "list": "목록",
        "list_head": "| 노드 | 요약 | 커널 |",
        "source": "소스:",
        "signature": "시그니처",
        "sig_labels": {"input": "입력", "output": "출력", "param": "파라미터"},
        "slots": "슬롯",
        "slots_head": "| 슬롯 | 종류 | 비고 |",
        "conditional": "옵션에 따라 존재",
        "config_head": "| 필드 | 타입 | 기본값 | 설명 |",
        "narrative": "동작 및 제약",
        "kernels": "커널",
        "no_kernel": "—",
        "no_default": "—",
    },
    "en": {
        "title": "Built-in nodes",
        "intro": "{n} built-in nodes. See [Node](03-usage/node.md) for the concept and "
        "[Writing custom nodes and kernels](07-contributing-nodes.md) for instructions "
        "on creating new nodes.",
        "list": "List",
        "list_head": "| Node | Summary | Kernel |",
        "source": "Source:",
        "signature": "Signature",
        "sig_labels": {"input": "Input", "output": "Output", "param": "Params"},
        "slots": "Slots",
        "slots_head": "| Slot | Type | Notes |",
        "conditional": "Present depending on config",
        "config_head": "| Field | Type | Default | Description |",
        "narrative": "Behavior and constraints",
        "kernels": "Kernels",
        "no_kernel": "None",
        "no_default": "Not specified",
    },
}

OUTS = {lang: ROOT / "docs" / lang / "06-nodes.md" for lang in STRINGS}

# 노드가 아닌 공용 헤더
SKIP = {"node-common.h"}

# 스탬프 펜스: `/*====` 로 열고 `====*/` 로 닫는 블록 주석
STAMP_OPEN = re.compile(r"^\s*/\*={10,}\s*$")
STAMP_CLOSE = re.compile(r"^\s*={10,}\*/\s*$")
# 스탬프 제목: `<Name> - <요약>`. 이름은 선언된 클래스와 대조해 거른다.
TITLE = re.compile(r"^([A-Z][A-Za-z0-9_]*)\s+-\s+(.*)$")
# 시그니처 라인: 콜론 주변 공백이 파일마다 다르다 (`Input:` / `Input :`)
SIGNATURE = re.compile(r"^(Input|Inputs|Params|Param|Output|Outputs)\s*:\s*(.*)$", re.I)
# 클래스 선언: vai:: 접두사는 있을 수도 없을 수도
CLASS_DECL = re.compile(r"^\s*(?:class|struct)\s+([A-Za-z0-9_]+)\s*:\s*public\s+(?:vai::)?Node\b")
# addSlot("name", NodeSlot::kind) — 조건부/동적 호출도 잡아서 뒤에서 구분한다
ADD_SLOT = re.compile(r'addSlot\(\s*("(?:[^"]*)"(?:\s*\+[^,]*)?)\s*,\s*NodeSlot::([a-z]+)\s*\)')
CONFIG_OPEN = re.compile(r"^\s*struct\s+Config\s*\{?\s*$")
# Config 필드: `<타입> <이름>[ = <기본값>];   // <주석>`
OPT_FIELD = re.compile(
    r"^\s*((?:const\s+)?[A-Za-z_][A-Za-z0-9_:<>, ]*?)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?:=\s*([^;]+?))?\s*;\s*(?://\s*(.*))?$"
)
# pipelinePool().request(...) 또는 헬퍼의 pool.request(...) — 커널의 진짜 출처.
# 파일명 규칙으로는 못 맞춘다 (attention.h -> "attention-fa").
# 첫 인자가 삼항일 수 있어(coopmat ? A : B) 특수화 상수 인자 앞까지 통째로
# 잡고 리터럴을 훑는다. 특수화 상수는 중괄호 목록이나 std::move()로 전달된다.
PIPELINE_REQUEST = re.compile(
    r"(?:pipelinePool\(\)|\bpool)\.request\(\s*(.*?),\s*(?:\{|std::move\()", re.S
)
SHADER_ALIAS = re.compile(r"const\s+char\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?);", re.S)
STRING_LITERAL = re.compile(r'"([^"]+)"')

SLOT_ORDER = {"input": 0, "output": 1, "persistent": 2, "temporary": 3}


class ParseError(Exception):
    pass


def stamp_body(line, indent):
    """블록 스탬프 본문의 공통 들여쓰기 두 칸을 제거한다."""
    if line.startswith(indent):
        line = line[len(indent) :]
    if line.startswith("  "):
        line = line[2:]
    return line.rstrip()


def parse_stamp(lines, stamp_idx):
    """펜스에서 시작하는 스탬프를 읽는다. 노드 스탬프가 아니면 None.

    제목이 두 줄로 넘어가는 경우가 있어(Linear, Attention),
    다음 줄이 시그니처도 빈 줄도 아니면 제목의 연속으로 이어 붙인다.
    """
    opening = lines[stamp_idx]
    indent_len = len(opening) - len(opening.lstrip())
    indent = opening[:indent_len]
    i = stamp_idx + 1
    if i >= len(lines):
        return None

    m = TITLE.match(stamp_body(lines[i], indent).strip())
    if not m:
        return None

    name, summary = m.group(1), m.group(2).strip()
    i += 1

    # 제목 연속 줄
    while i < len(lines) and not STAMP_CLOSE.match(lines[i]):
        body = stamp_body(lines[i], indent)
        if body.strip() == "" or SIGNATURE.match(body.strip()):
            break
        summary += " " + body.strip()
        i += 1

    signatures = []
    narrative = []
    while i < len(lines) and not STAMP_CLOSE.match(lines[i]):
        body = stamp_body(lines[i], indent)
        raw = body.strip()
        sig = SIGNATURE.match(raw)
        if sig:
            kind = sig.group(1).lower().rstrip("s")
            text = sig.group(2).strip()
            signatures.append([kind, text])
            i += 1
            # 시그니처가 다음 줄로 이어지는 경우 (들여쓴 연속 줄)
            while i < len(lines) and not STAMP_CLOSE.match(lines[i]):
                cont = stamp_body(lines[i], indent)
                if not cont.strip() or SIGNATURE.match(cont.strip()):
                    break
                if not cont.startswith("  "):
                    break
                signatures[-1][1] += " " + cont.strip()
                i += 1
            continue
        narrative.append(body.strip())
        i += 1

    if i >= len(lines):
        raise ParseError(f"{name}: 닫는 스탬프 펜스가 없다")

    while narrative and narrative[0] == "":
        narrative.pop(0)
    while narrative and narrative[-1] == "":
        narrative.pop()

    return {
        "name": name,
        "summary": " ".join(summary.split()),
        "signatures": [(k, " ".join(t.split())) for k, t in signatures],
        "narrative": narrative,
        "end": i + 1,
    }


def parse_slots(text):
    """addSlot() 호출에서 슬롯을 뽑는다. 이름 중복은 첫 등장만 남긴다."""
    slots = []
    seen = set()
    for line in text.splitlines():
        code = line.split("//")[0]
        for m in ADD_SLOT.finditer(code):
            expr, kind = m.group(1), m.group(2)
            dynamic = "+" in expr
            name = expr.split('"')[1]
            if dynamic:
                name = f"{name}0, {name}1, …"
            conditional = code.lstrip().startswith("if")
            if name in seen:
                continue
            seen.add(name)
            slots.append({"name": name, "kind": kind, "conditional": conditional})
    slots.sort(key=lambda s: (SLOT_ORDER.get(s["kind"], 9), s["name"]))
    return slots


def parse_config(lines):
    """가장 먼저 나오는 `struct Config` 의 필드를 읽는다."""
    for idx, line in enumerate(lines):
        if not CONFIG_OPEN.match(line):
            continue
        fields = []
        depth = 0
        started = False
        for cur in lines[idx:]:
            depth += cur.count("{") - cur.count("}")
            if "{" in cur and not started:
                started = True
                continue
            if started and depth <= 0:
                break
            if not started:
                continue
            if cur.strip().startswith("//") or not cur.strip():
                continue
            m = OPT_FIELD.match(cur)
            if m:
                fields.append(
                    {
                        "type": " ".join(m.group(1).split()),
                        "name": m.group(2),
                        "default": (m.group(3) or "").strip(),
                        "note": (m.group(4) or "").strip(),
                    }
                )
        return fields
    return []


def find_shaders(text):
    """노드가 실제로 요청하는 셰이더를 PipelinePool::request() 호출에서 뽑는다.

    존재하지 않는 이름은 버린다 — 오타나 파서 오작동이 문서에 죽은 링크로
    새어나가지 않게.
    """
    available = {p.stem: p.name for p in SHADER_DIR.glob("*.comp")}
    aliases = {name: expression for name, expression in SHADER_ALIAS.findall(text)}
    found = []
    for m in PIPELINE_REQUEST.finditer(text):
        argument = m.group(1).strip()
        expression = aliases.get(argument, argument)
        for name in STRING_LITERAL.findall(expression):
            if name in available and available[name] not in found:
                found.append(available[name])
    return found


def parse_node_file(path):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    declared = [m.group(1) for m in (CLASS_DECL.match(line) for line in lines) if m]

    nodes = []
    i = 0
    while i < len(lines):
        if STAMP_OPEN.match(lines[i]):
            stamp = parse_stamp(lines, i)
            # The title alone no longer identifies a node stamp, so a fence that
            # only divides sections is told apart by the declared class names.
            if stamp and (not declared or stamp["name"] in declared):
                nodes.append(stamp)
                i = stamp["end"]
                continue
        i += 1

    if declared and not nodes:
        raise ParseError(f"{path.name}: {declared} 를 선언했지만 읽을 수 있는 스탬프가 없다")

    slots = parse_slots(text)
    config = parse_config(lines)

    for n in nodes:
        n["file"] = path.name
        n["slots"] = slots if n["name"] in declared or len(nodes) == 1 else []
        n["config"] = config if n["name"] in declared or len(nodes) == 1 else []
        n["shaders"] = find_shaders(text)
        n["declared"] = n["name"] in declared

    return [n for n in nodes if n["declared"]]


def md_escape(s):
    return s.replace("|", r"\|")


def render(nodes, lang):
    s = STRINGS[lang]
    out = []
    w = out.append

    w(f"# {s['title']}")
    w("")
    w(s["intro"].format(n=len(nodes)))
    w("")

    w(f"## {s['list']}")
    w("")
    w(s["list_head"])
    w("|---|---|---|")
    for n in nodes:
        anchor = n["name"].lower()
        shaders = ", ".join(f"`{s}`" for s in n["shaders"]) or s["no_kernel"]
        w(f"| [{n['name']}](#{anchor}) | {md_escape(n['summary'])} | {shaders} |")
    w("")
    w("---")
    w("")

    for n in nodes:
        w(f"## {n['name']}")
        w("")
        w(md_escape(n["summary"]))
        w("")
        w(f"{s['source']} [`vai/nodes/{n['file']}`](../../vai/nodes/{n['file']})")
        w("")

        if n["signatures"]:
            w(f"**{s['signature']}**")
            w("")
            for kind, text in n["signatures"]:
                label = s["sig_labels"].get(kind, kind)
                w(f"- {label}: {text}")
            w("")

        if n["slots"]:
            w(f"**{s['slots']}**")
            w("")
            w(s["slots_head"])
            w("|---|---|---|")
            for slot in n["slots"]:
                note = s["conditional"] if slot["conditional"] else ""
                w(f"| `{slot['name']}` | {slot['kind']} | {note} |")
            w("")

        if n["config"]:
            w("**Config**")
            w("")
            w(s["config_head"])
            w("|---|---|---|---|")
            for o in n["config"]:
                default = f"`{o['default']}`" if o["default"] else s["no_default"]
                w(
                    f"| `{o['name']}` | `{md_escape(o['type'])}` | {md_escape(default)} | {md_escape(o['note'])} |"
                )
            w("")

        if n["narrative"]:
            w(f"**{s['narrative']}**")
            w("")
            for line in n["narrative"]:
                w(line)
            w("")

        if n["shaders"]:
            w(f"**{s['kernels']}**")
            w("")
            for shader in n["shaders"]:
                w(f"- `vai/shaders/{shader}`")
            w("")

        w("---")
        w("")

    return "\n".join(out).rstrip() + "\n"


def collect():
    """노드 헤더를 전부 읽는다. (nodes, errors) 를 돌려준다."""
    nodes, errors = [], []

    paths = sorted(p for p in NODE_DIR.glob("*.h") if p.name not in SKIP)
    if not paths:
        return nodes, [f"{NODE_DIR} 에서 노드 헤더를 찾지 못했다"]

    for p in paths:
        try:
            found = parse_node_file(p)
        except ParseError as e:
            errors.append(str(e))
            continue
        if not found:
            errors.append(f"{p.name}: 노드 스탬프를 읽지 못했다")
        nodes.extend(found)

    nodes.sort(key=lambda n: n["name"])
    return nodes, errors


def write(nodes, lang):
    """해당 언어의 06-nodes.md 를 쓴다. 내용이 같으면 건드리지 않는다.

    mkdocs serve 는 docs/ 를 감시하므로, 매번 쓰면 자기가 만든 변경을 보고
    다시 빌드하는 루프에 빠진다.
    """
    out = OUTS[lang]
    text = render(nodes, lang)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 메시지에 줄표(—)를 쓰지 않는다: 한국어 Windows 콘솔(cp949)에 없는 글자라
    # print 가 예외를 내고, mkdocs build / serve 가 거기서 멈춘다.
    if out.exists() and out.read_text(encoding="utf-8") == text:
        print(f"{out.relative_to(ROOT)}: 변경 없음 (노드 {len(nodes)}종)")
        return False

    out.write_text(text, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}: 노드 {len(nodes)}종")
    return True


def main():
    check = "--check" in sys.argv

    nodes, errors = collect()
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        print(
            "\n스탬프 형식이 표준을 벗어났거나 파서가 낡았다. 새 노드가 조용히 "
            "문서에서 빠지는 것을 막기 위해 실패로 처리한다.",
            file=sys.stderr,
        )
        return 1

    if check:
        drifted = False
        for lang in STRINGS:
            out = OUTS[lang]
            if not out.exists():
                print(
                    f"error: {out.relative_to(ROOT)} 가 없다. 생성기를 실행할 것.", file=sys.stderr
                )
                drifted = True
                continue
            if out.read_text(encoding="utf-8") != render(nodes, lang):
                print(
                    f"error: {out.relative_to(ROOT)} 가 헤더와 어긋난다. "
                    "`python docs/tools/gen-node-docs.py` 를 실행하고 결과를 커밋할 것.",
                    file=sys.stderr,
                )
                drifted = True
        if drifted:
            return 1
        print(f"ok: 노드 {len(nodes)}종, 드리프트 없음 ({', '.join(STRINGS)})")
        return 0

    for lang in STRINGS:
        write(nodes, lang)
    return 0


if __name__ == "__main__":
    sys.exit(main())
