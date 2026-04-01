from collections import defaultdict
from typing import Any, Dict, Optional, Sequence

from agent_kb_retrieval import AgenticKnowledgeBase


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def _is_effectively_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    if isinstance(v, (list, tuple, dict)) and len(v) == 0:
        return True
    return False


class AKB_Manager:
    def __init__(self, json_file_paths=None):
        self.knowledge_base = AgenticKnowledgeBase(json_file_paths=json_file_paths)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        weights: Optional[Dict[str, float]] = None,
        task_spec: Optional[
            Any
        ] = None,  # ✅ 여기로 들어온 task_spec 기반으로 필드 결정
        field_weights: Optional[Dict[str, float]] = None,
        per_field_candidate_k: Optional[int] = None,
    ) -> list[dict]:
        """
        - task_spec이 주어지면: 값이 있는 task_spec 필드만 검색 대상
        - task_spec이 없거나(또는 전부 비었으면): 전체 필드 검색
        """

        weights = weights or {"text": 0.5, "semantic": 0.5}
        field_weights = field_weights or {}
        cand_k = per_field_candidate_k or (top_k * 4)

        # ✅ task_spec 필드 -> 인덱스 필드명 매핑
        spec_to_index_field = {
            "problem_type": "task_spec.problem_type",
            "domain": "task_spec.domain",
            "what_to_derive": "task_spec.what_to_derive",
            "approach": "task_spec.approach",
        }

        all_fields = list(
            getattr(self.knowledge_base, "INDEX_FIELDS", self.knowledge_base.field_components).keys()
        )

        # ✅ task_spec로부터 사용할 필드 자동 선택
        use_fields: Sequence[str]
        if task_spec is None:
            use_fields = all_fields
        else:
            # dataclass(TaskSpec)든 dict든 처리
            if isinstance(task_spec, dict):
                spec_dict = task_spec
            else:
                # dataclass / pydantic-like object
                spec_dict = {
                    k: getattr(task_spec, k, None) for k in spec_to_index_field.keys()
                }

            selected = []
            for spec_key, index_field in spec_to_index_field.items():
                if not _is_effectively_empty(spec_dict.get(spec_key)):
                    selected.append(index_field)

            # task_spec에 유효한 값이 하나도 없으면 전체로 fallback
            use_fields = selected if selected else all_fields

        # 방어: 실제 인덱싱 가능한 필드만
        use_fields = [f for f in use_fields if f in self.knowledge_base.INDEX_FIELDS]
        if not use_fields:
            use_fields = all_fields

        score_board = defaultdict(float)

        for field in use_fields:
            fw = float(field_weights.get(field, 1.0))

            # text
            for r in self.knowledge_base.field_text_search(query, field, cand_k):
                score_board[r["task_id"]] += weights["text"] * fw * r["score"]

            # semantic
            for r in self.knowledge_base.field_semantic_search(query, field, cand_k):
                score_board[r["task_id"]] += weights["semantic"] * fw * r["score"]

        sorted_results = sorted(score_board.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        detailed = []
        for task_id, total_score in sorted_results:
            t = self.knowledge_base.tasks.get(task_id)
            if not t:
                continue
            task_spec = getattr(t, "task_spec", None)
            detailed.append(
                {
                    "task_id": t.task_id,
                    "total_score": float(total_score),
                    "task": to_text(t.task),
                    "task_spec": {
                        "problem_type": to_text(getattr(task_spec, "problem_type", None)),
                        "domain": to_text(getattr(task_spec, "domain", None)),
                        "what_to_derive": to_text(getattr(task_spec, "what_to_derive", None)),
                        "approach": to_text(getattr(task_spec, "approach", None)),
                    },
                }
            )
        return detailed
