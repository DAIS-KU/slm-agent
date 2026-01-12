from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class SimScore:
    redundancy: float
    exclusivity: float
    inter_mece: float
    details: Dict[str, Any]


@dataclass
class EntropyScore:
    pairwise_js_mean: float
    pairwise_js_min: float
    pairwise_js_max: float
    pair_count: int
    details: Dict[str, Any]


@dataclass
class SurpriseScore:
    total_surprise: float  # sum_i -log P(si | prompt)
    per_subtask_surprise: List[float]  # each -logP
    details: Dict[str, Any]


@dataclass
class DecompCandidate:
    subtasks: List[str]
    raw: str
    score: float
    # optional payloads (engine-specific but standardized)
    mece: Optional[Any] = None  # MeceScore (loss-based or sim-based)
    surprise: Optional[Any] = None  # SurpriseScore
    entropy: Optional[Any] = None  # EntropyScore
    # always safe to attach extra info
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutlinePick:
    outlines: List[str]
    indices: List[int]
    score: float
    details: Dict[str, Any]
