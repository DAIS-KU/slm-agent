from __future__ import annotations

import re
import json
from typing import List


_CODE_FENCE_RE = re.compile(r"```(?:json|yaml|txt)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    m = _CODE_FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _try_parse_json_list(text: str) -> List[str] | None:
    """
    모델이 ["...", "..."] 또는 {"subtasks":[...]} 형태로 줄 때 대응.
    """
    t = text.strip()
    # JSON blob만 남겨보려 시도
    # (앞뒤 설명이 섞이면 실패할 수 있음)
    start = min([i for i in [t.find("["), t.find("{")] if i != -1], default=-1)
    if start == -1:
        return None
    cand = t[start:]
    try:
        obj = json.loads(cand)
    except Exception:
        return None

    if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
        return [x.strip() for x in obj if x.strip()]
    if isinstance(obj, dict):
        for k in ["subtasks", "tasks", "decomposition", "steps", "items"]:
            if k in obj and isinstance(obj[k], list):
                out = [str(x).strip() for x in obj[k] if str(x).strip()]
                return out if out else None
    return None


_BULLET_PREFIX_RE = re.compile(
    r"""^\s*(?:[-*•]|(?:\d+[\).\:]|[a-zA-Z][\).\:]))\s+""",
    re.VERBOSE,
)


def _split_by_lines(text: str) -> List[str]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    items: List[str] = []
    buf: List[str] = []

    def flush():
        nonlocal buf
        if buf:
            s = " ".join(x.strip() for x in buf).strip()
            if s:
                items.append(s)
            buf = []

    for ln in lines:
        raw = ln.strip()
        if not raw:
            # 빈 줄: 현재 아이템을 끊는 역할을 하게
            flush()
            continue

        # "Subtasks:" 같은 헤더 제거
        if raw.lower() in {
            "subtasks:",
            "subtasks",
            "tasks:",
            "tasks",
            "decomposition:",
            "decomposition",
        }:
            flush()
            continue

        # bullet/numbering으로 시작하면 새 아이템
        if _BULLET_PREFIX_RE.match(raw):
            flush()
            raw = _BULLET_PREFIX_RE.sub("", raw).strip()

            # "1. **foo**" 같은 마크다운 강조 제거
            raw = re.sub(r"^\*\*(.+?)\*\*$", r"\1", raw).strip()
            buf = [raw]
        else:
            # 이어붙이는 줄(멀티라인 설명)
            # 단, 너무 길게 이어져서 합쳐지는 게 싫으면 여기서 flush 조건을 추가 가능
            buf.append(raw)

    flush()
    return items


def _split_by_semicolons(text: str) -> List[str]:
    # "1) ...; 2) ...; 3) ..." 같이 한 줄에 올 수도 있음
    parts = [p.strip() for p in re.split(r"\s*;\s*", text) if p.strip()]
    # 너무 잘게 쪼개질 수 있어서, numbering 패턴이 있으면 lines 방식이 더 좋음
    return parts


def parse_subtask(output: str) -> List[str]:
    """
    다양한 출력 포맷을 robust하게 subtask list로 변환.
    - JSON list / dict
    - markdown bullets / numbering
    - plain lines
    - semicolon-separated
    """
    text = _strip_code_fences(output)

    # 1) JSON 가능하면 우선
    js = _try_parse_json_list(text)
    if js:
        return js

    # 2) 줄 기반 파싱(불릿/넘버링 포함)
    items = _split_by_lines(text)

    # 3) items가 1개뿐인데 문장 안에 ';'로 여러 개 가능하면 세미콜론으로 시도
    if len(items) <= 1:
        semi = _split_by_semicolons(text)
        # semi가 2개 이상이고 길이가 적당하면 채택
        if len(semi) >= 2:
            items = semi

    # 4) 후처리: 너무 짧거나 "None"류 제거, 중복 제거(순서 유지)
    cleaned: List[str] = []
    seen = set()
    for s in items:
        s2 = re.sub(r"\s+", " ", s).strip()
        if not s2:
            continue
        if s2.lower() in {"none", "n/a", "na"}:
            continue
        if len(s2) < 3:
            continue
        if s2 not in seen:
            seen.add(s2)
            cleaned.append(s2)

    return cleaned


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _tokenize_no_special(tok, text: str) -> Dict[str, torch.Tensor]:
    return tok(text, add_special_tokens=False, return_tensors="pt")


def _subtasks_context(subtasks: List[str]) -> str:
    lines = ["Subtasks:"]
    for i, s in enumerate(subtasks, 1):
        lines.append(f"{i}. {s}")
    lines.append("")
    return "\n".join(lines)


def _single_subtask_context(s: str) -> str:
    return f"Subtask:\n{s}\n\n"
