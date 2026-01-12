from __future__ import annotations

import re
import json
from typing import List
import yaml

_CODE_FENCE_RE = re.compile(r"```(?:json|yaml|txt)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def load_prompts(path):
    with open(path, "r") as f:
        prompts = yaml.safe_load(f)
    return prompts


def tokenize_no_special(tokenizer, text: str):
    return tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )


def format_prompt(
    *,
    task_text: str,
    outline_text: Optional[str],
    prev_subtasks: List[str],
) -> str:
    """
    Prompt = (t, [o], s1..s{i-1}) 를 한 문자열로 구성.
    포맷은 프로젝트 스타일에 맞게 바꿔도 됩니다(핵심은 일관성).
    """
    parts: List[str] = []
    parts.append("TASK:\n" + task_text.strip())

    if outline_text is not None and outline_text.strip():
        parts.append("OUTLINE:\n" + outline_text.strip())

    if prev_subtasks:
        # 이전 subtasks를 모델 입력에 명확히 주기 위해 번호/불릿 유지
        st = "\n".join([f"- {s.strip()}" for s in prev_subtasks if s and s.strip()])
        parts.append("SUBTASKS SO FAR:\n" + st)

    # 다음 subtask 생성 유도용 cue
    parts.append("NEXT SUBTASK:\n")
    return "\n\n".join(parts)


def strip_code_fences(text: str) -> str:
    m = _CODE_FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def try_parse_json_list(text: str) -> List[str] | None:
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


def split_by_lines(text: str) -> List[str]:
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


def split_by_semicolons(text: str) -> List[str]:
    # "1) ...; 2) ...; 3) ..." 같이 한 줄에 올 수도 있음
    parts = [p.strip() for p in re.split(r"\s*;\s*", text) if p.strip()]
    # 너무 잘게 쪼개질 수 있어서, numbering 패턴이 있으면 lines 방식이 더 좋음
    return parts


def parse_subtask(output: str) -> List[str]:
    """
    Extract Subgoal number + text (e.g., 'Subgoal 1: ...') from the given string.
    """
    pattern = r'"Subgoal\s*(\d+)"\s*:\s*"([^"]+)"'
    matches = re.findall(pattern, output)
    # preserve order as they appear
    return [f"Subgoal {num}: {text}" for num, text in matches]


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def tokenize_no_special(tok, text: str) -> Dict[str, torch.Tensor]:
    return tok(text, add_special_tokens=False, return_tensors="pt")


def subtasks_context(subtasks: List[str]) -> str:
    lines = ["Subtasks:"]
    for i, s in enumerate(subtasks, 1):
        lines.append(f"{i}. {s}")
    lines.append("")
    return "\n".join(lines)


def single_subtask_context(s: str) -> str:
    return f"Subtask:\n{s}\n\n"
