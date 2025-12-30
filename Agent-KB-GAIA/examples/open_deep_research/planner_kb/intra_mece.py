from __future__ import annotations

import itertools
import math
import random
from typing import Dict, List, Optional, Tuple

from .inter_mece import InterMeceEngine
from .mece_utils import *
from .mece_common import *


class IntraMeceEngine:
    """
    Intra-MECE Engine (집합 간 MECE)

    - InterMeceEngine(inter_engine)를 인스턴스로 보유하고,
      * 후보 decomposition의 inter-MECE(집합 내) 점수는 inter_engine.score(...)로 계산
      * 집합 간(intra) MECE는 NLL 기반 set-level overlap으로 계산

    mode:
      - 'loss'만 동작
      - 'sim'은 NotImplementedError
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

        target_b = subtasks_context(b)
        nll_b = self.inter.conditional_nll_per_token(
            "", target_b, max_length=self.max_length
        )
        nll_b_given_a = self.inter.conditional_nll_per_token(
            subtasks_context(a), target_b, max_length=self.max_length
        )

        if not (math.isfinite(nll_b) and nll_b > 1e-8 and math.isfinite(nll_b_given_a)):
            return 0.0

        return clamp01((nll_b - nll_b_given_a) / nll_b)

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
        val = clamp01(1.0 - sym_overlap)

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
    ) -> List[DecompCandidate]:
        """
        InterMeceEngine처럼 샘플링하되, 점수는 inter.score(mode, ...)를 사용.
        반환 타입은 기존 DecompCandidate 그대로 사용.
        """
        seen_raw = set()
        out: List[DecompCandidate] = []

        for sample_num in range(num_samples):
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
            print(f"Generate {sample_num}th decomposition.:")
            print(subtasks)
            if not (min_subtasks <= len(subtasks) <= max_subtasks):
                continue

            mece = self.inter.score(mode, subtasks, task_text, alpha=alpha_inter)
            print(f"Generate {sample_num}th subtasks.(score {mece})")

            out.append(
                DecompCandidate(
                    subtasks=subtasks,
                    raw=raw,
                    score=float(mece.coverage),  # 기본값 (선택 후 갱신 가능)
                    score_mode=mode,
                    mece=mece,
                    details={"alpha_inter": alpha_inter},
                )
            )

        return out

    # -------------------------
    # Inter-MECE scalar (weighted sum) for ranking
    # -------------------------
    @staticmethod
    def inter_mece_weighted_score(
        mece,
        *,
        w_coverage: float,
        w_exclusivity: float,
    ) -> float:
        """
        inter-MECE를 정렬용 스칼라로 만들기 위한 가중합.

        - mece.coverage, mece.exclusivity는 0~1 범위라고 가정(일반적인 clamp01 설계).
        - 가중치는 합이 1일 필요는 없지만, 보통 1로 맞추는 게 해석이 쉬움.
        """
        wc = float(w_coverage)
        we = float(w_exclusivity)
        return wc * float(mece.coverage) + we * float(mece.exclusivity)

    # -------------------------
    # Diversity objective: max-sum intra among a subset of size k
    # -------------------------
    def _pair_intra(self, i: int, j: int, pool: List[DecompCandidate]) -> float:
        return self.intra_mece_between_sets(pool[i].subtasks, pool[j].subtasks)

    def _subset_intra_sum(self, idxs: List[int], pool: List[DecompCandidate]) -> float:
        s = 0.0
        for a_pos in range(len(idxs)):
            a = idxs[a_pos]
            for b_pos in range(a_pos + 1, len(idxs)):
                b = idxs[b_pos]
                s += self._pair_intra(a, b, pool)
        return s

    def _select_max_intra_subset(
        self,
        pool: List[DecompCandidate],
        k: int,
        *,
        # exact search thresholds
        exact_max_pool: int = 22,
        exact_max_comb: int = 250_000,
        # approximate search knobs
        restarts: int = 8,
        local_iters: int = 400,
        seed: Optional[int] = None,
    ) -> Tuple[List[int], float]:
        """
        Returns (selected_indices, best_intra_sum).

        Objective (max-sum diversity):
            maximize  sum_{i<j in S} intra(i,j)
        """
        n = len(pool)
        k = max(1, min(k, n))

        if seed is not None:
            random.seed(seed)

        # ---- exact (bruteforce) when feasible
        if n <= exact_max_pool:

            def nCk(nn: int, kk: int) -> int:
                kk = min(kk, nn - kk)
                if kk < 0:
                    return 0
                num = 1
                den = 1
                for t in range(1, kk + 1):
                    num *= nn - kk + t
                    den *= t
                return num // den

            combs = nCk(n, k)
            if combs <= exact_max_comb:
                best_idxs: List[int] = []
                best_val = -1e18
                for idxs in itertools.combinations(range(n), k):
                    idxs_list = list(idxs)
                    v = self._subset_intra_sum(idxs_list, pool)
                    if v > best_val:
                        best_val = v
                        best_idxs = idxs_list
                return best_idxs, float(best_val)

        # ---- approximate: greedy + local swap improvement, with random restarts
        intra_mat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                v = self._pair_intra(i, j, pool)
                intra_mat[i][j] = v
                intra_mat[j][i] = v

        def intra_sum_of_set(sel: List[int]) -> float:
            s = 0.0
            for a_pos in range(len(sel)):
                a = sel[a_pos]
                for b_pos in range(a_pos + 1, len(sel)):
                    b = sel[b_pos]
                    s += intra_mat[a][b]
            return s

        def greedy_start(start_idx: int) -> List[int]:
            sel = [start_idx]
            remaining = [i for i in range(n) if i != start_idx]
            # add elements maximizing sum of intra to current set
            while len(sel) < k:
                best = None
                best_gain = -1e18
                for cand in remaining:
                    gain = sum(intra_mat[cand][s] for s in sel)
                    if gain > best_gain:
                        best_gain = gain
                        best = cand
                sel.append(best)  # type: ignore[arg-type]
                remaining.remove(best)  # type: ignore[arg-type]
            return sel

        def local_swap(sel: List[int]) -> List[int]:
            sel_set = set(sel)
            cur = sel[:]
            cur_val = intra_sum_of_set(cur)

            non_sel = [i for i in range(n) if i not in sel_set]

            for _ in range(local_iters):
                improved = False
                for _trial in range(min(40, len(cur) * max(1, len(non_sel)))):
                    out_i = random.choice(cur)
                    in_j = random.choice(non_sel)

                    new = cur[:]
                    new.remove(out_i)
                    new.append(in_j)

                    new_val = intra_sum_of_set(new)
                    if new_val > cur_val + 1e-12:
                        non_sel.remove(in_j)
                        non_sel.append(out_i)
                        cur = new
                        cur_val = new_val
                        improved = True
                        break

                if not improved:
                    break
            return cur

        best_sel: List[int] = []
        best_val = -1e18

        start_candidates = list(range(min(n, max(3, restarts))))
        while len(start_candidates) < restarts:
            start_candidates.append(random.randrange(n))

        for sidx in start_candidates:
            sel = greedy_start(sidx)
            sel = local_swap(sel)
            v = intra_sum_of_set(sel)
            if v > best_val:
                best_val = v
                best_sel = sel

        return best_sel, float(best_val)

    # -------------------------
    # Main API: pick top-K maximizing intra, then rank by weighted inter-MECE
    # -------------------------
    def pick_topk(
        self,
        *,
        task_text: str,
        task_decomposition_prompt: str,
        mode: str = "loss",
        num_samples: int = 10,
        top_k: int = 5,
        alpha_inter: float = 0.5,
        # filtering
        min_subtasks: int = 2,
        max_subtasks: int = 10,
        dedup_raw: bool = True,
        seed: Optional[int] = None,
        # pool control (performance cap; set None or <=0 to disable)
        pool_cap: int = 30,
        # ranking: weighted sum for inter-MECE
        w_coverage: float = 0.5,
        w_exclusivity: float = 0.5,
        # if False, keep subset order (not recommended unless you want it)
        sort_by_weighted_inter_mece: bool = True,
    ) -> List[DecompCandidate]:
        """
        Behavior:
          1) sample candidates (+ inter-MECE computed)
          2) choose subset of size top_k that maximizes intra MECE (max-sum diversity)
             - exact when feasible, otherwise approximate (greedy + local swaps)
          3) return that subset sorted by weighted inter-MECE score:
                inter_weighted = w_coverage*coverage + w_exclusivity*exclusivity
        """
        if seed is not None:
            random.seed(seed)

        if mode == "sim":
            raise NotImplementedError("sim-based intra-MECE not implemented yet.")
        if mode != "loss":
            raise ValueError("mode must be 'loss' or 'sim'")

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

        # optional: cap candidate count (just to keep intra optimization tractable)
        # NOTE: this is not part of the intra objective; it's a performance knob.
        if pool_cap and pool_cap > 0 and len(candidates) > pool_cap:
            candidates.sort(key=lambda c: c.mece.coverage, reverse=True)
            pool = candidates[:pool_cap]
        else:
            pool = candidates

        k = max(1, min(top_k, len(pool)))
        idxs, intra_sum = self._select_max_intra_subset(pool, k, seed=seed)

        selected = [pool[i] for i in idxs]

        # attach diagnostics + computed weighted inter score
        for cand in selected:
            inter_w = self.inter_mece_weighted_score(
                cand.mece, w_coverage=w_coverage, w_exclusivity=w_exclusivity
            )
            cand.details = {
                **cand.details,
                "selection_objective": "max_sum_intra_mece",
                "subset_intra_sum": float(intra_sum),
                "w_coverage": float(w_coverage),
                "w_exclusivity": float(w_exclusivity),
                "inter_mece_weighted": float(inter_w),
            }
            # optional: store as score for convenient downstream usage
            cand.score = float(inter_w)

        if sort_by_weighted_inter_mece:
            selected.sort(
                key=lambda c: self.inter_mece_weighted_score(
                    c.mece, w_coverage=w_coverage, w_exclusivity=w_exclusivity
                ),
                reverse=True,
            )

        return selected
