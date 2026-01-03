from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class MeceScore:
    coverage: float
    redundancy: float
    exclusivity: float
    inter_mece: float
    details: Dict[str, Any]


@dataclass
class DecompCandidate:
    subtasks: List[str]
    raw: str
    score: float
    mece: MeceScore
    details: Dict[str, Any]


@dataclass
class IntraPickDetails:
    """디버깅용: 집합 간 MECE 관련 계산들을 담는다."""

    selection_objective: float
    avg_intra_mece_to_selected: float
    intra_mece_to_selected: List[float]
