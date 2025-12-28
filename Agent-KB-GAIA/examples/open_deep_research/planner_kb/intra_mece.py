from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from mece_utils import *
from mece_common import *


class IntraMeceEngine:
    """
    Intra-MECE Engine (집합 간 MECE)

    - InterMeceEngine(inter_engine)를 인스턴스로 보유하고,
      * 후보 decomposition의 inter-MECE(집합 내) 점수는 inter_engine.score(...)로 계산
      * 집합 간(intra) MECE는 NLL 기반 set-level overlap으로 계산

    mode:
      - 'loss'만 동작
      - 'sim'은 NotImplementedError (pass)
    """

    def __init__(
        self,
        tm,
        *,
        call_model_fn,
        call_model_kwargs,
        max_length: int = 2048,
    ):
        self.inter = InterMeceEngine(
            tm,
            call_model_fn=call_model_fn,
            call_model_kwargs=call_model_kwargs,
            max_length=max_length,
        )
        self.max_length = max_length
        # pairwise intra cache: (id(list_a), id(list_b)) -> intra_mece
        self._pair_cache: Dict[Tuple[int, int], float] = {}

    # -------------------------
    # Set-level intra-MECE (between sets)
    # -------------------------
    def _set_overlap_a_to_b_loss(self, a: List[str], b: List[str]) -> float:
        """
        overlap(A->B) = (NLL(B) - NLL(B|A)) / NLL(B)

        B는 "Subtasks:\n1...." 전체 문자열을 target으로 두고,
        A도 동일 포맷으로 context에 넣는다.
        """
        a = [s.strip() for s in a if s and s.strip()]
        b = [s.strip() for s in b if s and s.strip()]
        if not a or not b:
            return 0.0

        target_b = _subtasks_context(b)
        nll_b = self.inter.conditional_nll_per_token(
            "", target_b, max_length=self.max_length
        )
        nll_b_given_a = self.inter.conditional_nll_per_token(
            _subtasks_context(a), target_b, max_length=self.max_length
        )

        if not (math.isfinite(nll_b) and nll_b > 1e-8 and math.isfinite(nll_b_given_a)):
            return 0.0

        return _clamp01((nll_b - nll_b_given_a) / nll_b)

    def intra_mece_between_sets(self, a: List[str], b: List[str]) -> float:
        """
        intra-MECE(집합 간) = 1 - 0.5*(overlap(A->B)+overlap(B->A))
        """
        ka, kb = id(a), id(b)
        key = (ka, kb) if ka <= kb else (kb, ka)
        if key in self._pair_cache:
            return self._pair_cache[key]

        ab = self._set_overlap_a_to_b_loss(a, b)
        ba = self._set_overlap_a_to_b_loss(b, a)
        sym_overlap = 0.5 * (ab + ba)
        val = _clamp01(1.0 - sym_overlap)

        self._pair_cache[key] = val
        return val

    # -------------------------
    # Candidate sampling (reuses inter_engine's call_model)
    # -------------------------
    def _sample_candidates(
        self,
        *,
        task_text: str,
        task_decomposition_prompt: str,
        mode: str,
        num_samples: int,
        alpha_inter: float,
        min_subtasks: int,
        max_subtasks: int,
        dedup_raw: bool,
    ):
        """
        InterMeceEngine처럼 샘플링하되, 점수는 inter.score(mode, ...)를 사용.
        반환 타입은 기존 DecompCandidate 그대로 사용.
        """
        seen_raw = set()
        out: List[DecompCandidate] = []

        for _ in range(num_samples):
            raw = self.inter.call_model_fn(
                query=task_decomposition_prompt, **self.inter.call_model_kwargs
            )
            if not raw or not isinstance(raw, str):
                continue

            raw_norm = " ".join(raw.split())
            if dedup_raw and raw_norm in seen_raw:
                continue
            seen_raw.add(raw_norm)

            subtasks = parse_subtask(raw)
            if not (min_subtasks <= len(subtasks) <= max_subtasks):
                continue

            mece = self.inter.score(mode, subtasks, task_text, alpha=alpha_inter)

            out.append(
                DecompCandidate(
                    subtasks=subtasks,
                    raw=raw,
                    score=float(
                        mece.coverage
                    ),  # 우선 coverage를 기본 score로 (선택 objective에서 갱신)
                    score_mode=mode,
                    mece=mece,
                    details={"alpha_inter": alpha_inter},
                )
            )

        return out

    # -------------------------
    # Main API: pick top-K sets with intra-MECE
    # -------------------------
    def pick_topk(
        self,
        *,
        task_text: str,
        task_decomposition_prompt: str,
        mode: str = "loss",
        num_samples: int = 40,
        top_k: int = 5,
        alpha_inter: float = 0.5,
        # selection hyperparams
        coverage_pool: int = 30,
        beta_select: float = 0.7,  # 1이면 coverage만, 0이면 intra만
        gamma_diversity: float = 1.0,  # intra 항 스케일
        # filtering
        min_subtasks: int = 2,
        max_subtasks: int = 20,
        dedup_raw: bool = True,
        seed: Optional[int] = None,
        # ordering
        sort_by_inter_mece: bool = True,
    ) -> List[DecompCandidate]:
        """
        1) decomposition 후보 샘플링
        2) coverage 상위 coverage_pool로 풀 제한
        3) greedy로 top_k 선택:
             objective = beta*coverage + (1-beta)*gamma*avg_intra_mece_to_selected
        4) 반환 정렬:
             - 기본: 집합 내 MECE(inter) 높은 순(exclusivity), tie: coverage
             - sort_by_inter_mece=False면 selection objective(score)로 정렬 유지 가능
        """
        if seed is not None:
            random.seed(seed)

        if mode == "sim":
            raise NotImplementedError("sim-based intra-MECE not implemented yet.")
        if mode != "loss":
            raise ValueError("mode must be 'loss' or 'sim'")

        # 1) 후보 샘플링 (+ inter-MECE 점수는 inter.score에서 계산)
        candidates = self._sample_candidates(
            task_text=task_text,
            task_decomposition_prompt=task_decomposition_prompt,
            mode=mode,
            num_samples=num_samples,
            alpha_inter=alpha_inter,
            min_subtasks=min_subtasks,
            max_subtasks=max_subtasks,
            dedup_raw=dedup_raw,
        )
        if not candidates:
            return []

        # 2) coverage 상위 풀
        candidates.sort(key=lambda c: c.mece.coverage, reverse=True)
        pool = candidates[: max(top_k, coverage_pool)]

        # 3) greedy 선택
        selected: List[DecompCandidate] = [pool[0]]
        remaining = pool[1:]

        while remaining and len(selected) < max(1, top_k):
            best_idx = -1
            best_obj = -1e18
            best_avg_intra = 0.0
            best_intras: List[float] = []

            for idx, cand in enumerate(remaining):
                intras = [
                    self.intra_mece_between_sets(cand.subtasks, sel.subtasks)
                    for sel in selected
                ]
                avg_intra = (sum(intras) / max(1, len(intras))) if intras else 0.0

                obj = beta_select * cand.mece.coverage + (1.0 - beta_select) * (
                    gamma_diversity * avg_intra
                )

                if obj > best_obj:
                    best_obj = obj
                    best_idx = idx
                    best_avg_intra = avg_intra
                    best_intras = intras

            chosen = remaining.pop(best_idx)
            chosen.score = float(best_obj)
            chosen.details = {
                **chosen.details,
                "beta_select": beta_select,
                "gamma_diversity": gamma_diversity,
                "avg_intra_mece_to_selected": float(best_avg_intra),
                "intra_mece_to_selected": [float(x) for x in best_intras],
                "selection_objective": float(best_obj),
            }
            selected.append(chosen)

        # 4) 반환 정렬
        if sort_by_inter_mece:
            # 요구했던 “집합 내 MECE(inter) 순 정렬”
            selected.sort(
                key=lambda c: (c.mece.exclusivity, c.mece.coverage), reverse=True
            )
        else:
            # 선택 objective(score) 큰 순
            selected.sort(key=lambda c: c.score, reverse=True)

        return selected
