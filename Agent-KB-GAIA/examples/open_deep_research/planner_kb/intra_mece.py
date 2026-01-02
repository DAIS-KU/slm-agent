from __future__ import annotations

import itertools
import math
import random
from typing import Dict, List, Optional, Tuple

from .inter_mece import InterMeceEngine, SimInterMeceEngine
from .mece_utils import *
from .mece_common import *


class IntraMeceEngine:
    """
    Intra-MECE Engine (집합 간 MECE)

    - InterMeceEngine(inter_engine)를 인스턴스로 보유하고,
      * 후보 decomposition의 inter-MECE(집합 내) 점수는 inter_engine.score(...)로 계산
      * 집합 간(intra) MECE는 NLL 기반 set-level overlap으로 계산
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
        num_samples: int,
        alpha_inter: float,
        min_subtasks: int,
        max_subtasks: int,
        dedup_raw: bool,
    ) -> List[DecompCandidate]:
        """
        InterMeceEngine처럼 샘플링하되, 점수는 inter.score(...)를 사용.
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

            mece = self.inter.score(subtasks, task_text, alpha=alpha_inter)
            print(f"Generate {sample_num}th subtasks.(score {mece})")

            out.append(
                DecompCandidate(
                    subtasks=subtasks,
                    raw=raw,
                    score=float(mece.coverage),  # 기본값 (선택 후 갱신 가능)
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

        candidates = self._sample_candidates(
            task_text=task_text,
            task_decomposition_prompt=task_decomposition_prompt,
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


class SimBasedIntraMeceEngine:
    def __init__(
        self,
        tm,
        *,
        call_model_fn,
        call_model_kwargs,
        embed_texts_fn=None,
        max_length: int = 2048,
        eps: float = 1e-12,
    ):
        self.inter = SimInterMeceEngine(
            tm,
            call_model_fn=call_model_fn,
            call_model_kwargs=call_model_kwargs,
            embed_texts_fn=embed_texts_fn,
            max_length=max_length,
            eps=eps,
        )
        self.max_length = max_length
        self.eps = eps

    # -------------------------
    # Candidate sampling
    # -------------------------
    @torch.no_grad()
    def _sample_candidates(
        self,
        *,
        task_text: str,
        task_decomposition_prompt: str,
        num_samples: int,
        alpha_inter: float,
        min_subtasks: int,
        max_subtasks: int,
        dedup_raw: bool,
        score_kwargs: Optional[Dict] = None,
    ) -> List[DecompCandidate]:
        score_kwargs = score_kwargs or {}
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
            if not (min_subtasks <= len(subtasks) <= max_subtasks):
                continue

            mece = self.inter.score(
                "sim", subtasks, task_text, alpha=alpha_inter, **score_kwargs
            )

            out.append(
                DecompCandidate(
                    subtasks=subtasks,
                    raw=raw,
                    score=float(mece.inter_mece),
                    score_mode="sim",
                    mece=mece,
                    details={"alpha_inter": alpha_inter, **score_kwargs},
                )
            )
        return out

    # ============================================================
    # Silhouette precompute for a pool: points embeddings & dist mat
    # ============================================================
    @torch.no_grad()
    def _build_point_view(
        self,
        pool: List[DecompCandidate],
        *,
        metric: str = "cosine",
    ):
        """
        Build:
          - points: all subtasks across pool
          - labels: cluster id per point (0..n-1)
          - cluster_points: {cluster_id: [point_idx,...]}
          - Dist: [P,P] distance matrix among points
        """
        eps = self.eps
        points: List[str] = []
        labels: List[int] = []
        cluster_points: Dict[int, List[int]] = {}

        for ci, cand in enumerate(pool):
            subs = [s.strip() for s in cand.subtasks if s and s.strip()]
            for s in subs:
                idx = len(points)
                points.append(s)
                labels.append(ci)
                cluster_points.setdefault(ci, []).append(idx)

        if not points:
            # empty edge
            P = 0
            Dist = torch.empty((0, 0))
            return points, labels, cluster_points, Dist

        E = self.inter.embed_texts(points)  # [P,D]
        if metric != "cosine":
            raise ValueError("only metric='cosine' supported")

        En = E / torch.linalg.norm(E, dim=-1, keepdim=True).clamp_min(eps)
        C = En @ En.T
        Dist = (1.0 - C).clamp_min(0.0)
        return points, labels, cluster_points, Dist

    # ============================================================
    # Subset silhouette objective: mean silhouette over all points
    # ============================================================
    @torch.no_grad()
    def _subset_silhouette_mean(
        self,
        subset_clusters: List[int],
        *,
        cluster_points: Dict[int, List[int]],
        Dist: torch.Tensor,
    ) -> float:
        """
        Compute mean silhouette over points belonging to subset_clusters only.
        Return in [-1,1]. If invalid (e.g., <2 clusters or no points), return -1e9.
        """
        eps = self.eps
        if len(subset_clusters) < 2:
            return -1e9

        # collect points in subset
        pts = []
        for c in subset_clusters:
            pts.extend(cluster_points.get(c, []))
        if len(pts) == 0:
            return -1e9

        # precompute list of clusters actually having >=1 point
        active_clusters = [
            c for c in subset_clusters if len(cluster_points.get(c, [])) > 0
        ]
        if len(active_clusters) < 2:
            return -1e9

        # silhouette per point
        s_sum = 0.0
        count = 0

        for i in pts:
            ci = None
            # find i's cluster by scanning (fast enough if cluster_points is small; else build inverse map)
            # We'll build inverse map once for speed:
        # build inverse map for subset points
        inv = {}
        for c in active_clusters:
            for p in cluster_points[c]:
                inv[p] = c

        for i in pts:
            ci = inv[i]
            same = cluster_points[ci]
            # a(i)
            if len(same) <= 1:
                a = torch.tensor(0.0, device=Dist.device)
            else:
                a = Dist[i, same].sum() / (len(same) - 1)

            # b(i)
            b = None
            for cj in active_clusters:
                if cj == ci:
                    continue
                idxs = cluster_points[cj]
                mean_ij = Dist[i, idxs].mean()
                b = mean_ij if b is None else torch.minimum(b, mean_ij)

            if b is None:
                continue

            denom = torch.maximum(a, b).clamp_min(eps)
            s = (b - a) / denom
            s_sum += float(s.item())
            count += 1

        if count == 0:
            return -1e9
        return s_sum / count

    # ============================================================
    # Select K clusters maximizing subset silhouette
    #  - exact when feasible
    #  - else greedy + local swaps (+ restarts)
    # ============================================================
    @torch.no_grad()
    def _select_subset_max_silhouette(
        self,
        *,
        pool: List[DecompCandidate],
        k: int,
        metric: str = "cosine",
        # exact caps
        exact_max_pool: int = 18,
        exact_max_comb: int = 200_000,
        # approximate knobs
        restarts: int = 8,
        local_iters: int = 250,
        seed: Optional[int] = None,
    ) -> Tuple[List[int], float, Dict]:
        """
        Returns:
          (selected_cluster_indices_in_pool, best_sil_mean, debug)
        """
        if seed is not None:
            random.seed(seed)

        n = len(pool)
        k = max(1, min(k, n))
        if k == 1:
            return [0], -1e9, {"reason": "k=1 silhouette undefined"}

        # build global point view once
        points, labels, cluster_points, Dist = self._build_point_view(
            pool, metric=metric
        )

        # if not enough clusters with points
        nonempty_clusters = [i for i in range(n) if len(cluster_points.get(i, [])) > 0]
        if len(nonempty_clusters) < 2:
            return (
                nonempty_clusters[:k],
                -1e9,
                {"reason": "not enough non-empty clusters"},
            )

        # ---- exact if feasible
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
                best_S = []
                best_v = -1e18
                for S in itertools.combinations(range(n), k):
                    v = self._subset_silhouette_mean(
                        list(S), cluster_points=cluster_points, Dist=Dist
                    )
                    if v > best_v:
                        best_v = v
                        best_S = list(S)
                return (
                    best_S,
                    float(best_v),
                    {"mode": "exact", "combs": combs, "P": len(points)},
                )

        # ---- approximate: greedy forward + local swap, with restarts
        all_clusters = list(range(n))

        def greedy_start(start_pair: Tuple[int, int]) -> List[int]:
            sel = [start_pair[0], start_pair[1]]
            remaining = [c for c in all_clusters if c not in sel]
            while len(sel) < k:
                best_c = None
                best_v = -1e18
                # try each candidate addition
                for c in remaining:
                    v = self._subset_silhouette_mean(
                        sel + [c], cluster_points=cluster_points, Dist=Dist
                    )
                    if v > best_v:
                        best_v = v
                        best_c = c
                sel.append(best_c)  # type: ignore[arg-type]
                remaining.remove(best_c)  # type: ignore[arg-type]
            return sel

        def local_swap(sel: List[int]) -> List[int]:
            cur = sel[:]
            cur_val = self._subset_silhouette_mean(
                cur, cluster_points=cluster_points, Dist=Dist
            )
            non_sel = [c for c in all_clusters if c not in cur]

            for _ in range(local_iters):
                improved = False
                # limited random trials per iter
                for _trial in range(min(60, len(cur) * max(1, len(non_sel)))):
                    out_c = random.choice(cur)
                    in_c = random.choice(non_sel)
                    new = cur[:]
                    new.remove(out_c)
                    new.append(in_c)
                    v = self._subset_silhouette_mean(
                        new, cluster_points=cluster_points, Dist=Dist
                    )
                    if v > cur_val + 1e-8:
                        non_sel.remove(in_c)
                        non_sel.append(out_c)
                        cur = new
                        cur_val = v
                        improved = True
                        break
                if not improved:
                    break
            return cur

        # pick restart seeds: choose some top inter-mece pairs + random pairs
        # (silhouette는 inter-mece와 별개지만, 좋은 후보에서 시작하면 대체로 성능 좋음)
        order = list(range(n))
        order.sort(key=lambda i: float(pool[i].mece.inter_mece), reverse=True)

        start_pairs: List[Tuple[int, int]] = []
        top_m = min(n, max(6, restarts))
        for i in range(top_m):
            for j in range(i + 1, top_m):
                start_pairs.append((order[i], order[j]))
                if len(start_pairs) >= restarts:
                    break
            if len(start_pairs) >= restarts:
                break
        while len(start_pairs) < restarts:
            a = random.randrange(n)
            b = random.randrange(n)
            if a != b:
                start_pairs.append((a, b))

        best_S: List[int] = []
        best_v = -1e18

        for sp in start_pairs:
            sel = greedy_start(sp)
            sel = local_swap(sel)
            v = self._subset_silhouette_mean(
                sel, cluster_points=cluster_points, Dist=Dist
            )
            if v > best_v:
                best_v = v
                best_S = sel

        return (
            best_S,
            float(best_v),
            {"mode": "approx", "restarts": restarts, "P": len(points)},
        )

    # ============================================================
    # Public API: pick K by silhouette-optimal subset,
    # then return sorted by inter-mece desc
    # ============================================================
    @torch.no_grad()
    def pick_topk_by_silhouette(
        self,
        *,
        task_text: str,
        task_decomposition_prompt: str,
        num_samples: int = 20,
        top_k: int = 5,
        alpha_inter: float = 0.5,
        # filtering
        min_subtasks: int = 2,
        max_subtasks: int = 10,
        dedup_raw: bool = True,
        seed: Optional[int] = None,
        # performance cap before silhouette selection (optional but recommended)
        pool_cap: int = 30,
        # sim scoring knobs forwarded to SimInterMeceEngine.score
        score_kwargs: Optional[Dict] = None,
        # silhouette selection knobs
        silhouette_metric: str = "cosine",
        exact_max_pool: int = 18,
        exact_max_comb: int = 200_000,
        restarts: int = 8,
        local_iters: int = 250,
    ) -> List[DecompCandidate]:
        if seed is not None:
            random.seed(seed)

        candidates = self._sample_candidates(
            task_text=task_text,
            task_decomposition_prompt=task_decomposition_prompt,
            num_samples=num_samples,
            alpha_inter=alpha_inter,
            min_subtasks=min_subtasks,
            max_subtasks=max_subtasks,
            dedup_raw=dedup_raw,
            score_kwargs=score_kwargs,
        )
        if not candidates:
            return []

        # pool cap for tractability (silhouette is O(P^2))
        if pool_cap and pool_cap > 0 and len(candidates) > pool_cap:
            candidates.sort(key=lambda c: float(c.mece.inter_mece), reverse=True)
            pool = candidates[:pool_cap]
        else:
            pool = candidates

        k = max(1, min(top_k, len(pool)))
        if k < 2:
            # silhouette undefined; just return top inter-mece
            pool.sort(key=lambda c: float(c.mece.inter_mece), reverse=True)
            return pool[:k]

        idxs, best_sil, dbg = self._select_subset_max_silhouette(
            pool=pool,
            k=k,
            metric=silhouette_metric,
            exact_max_pool=exact_max_pool,
            exact_max_comb=exact_max_comb,
            restarts=restarts,
            local_iters=local_iters,
            seed=seed,
        )

        selected = [pool[i] for i in idxs]

        # attach diagnostics
        for cand in selected:
            cand.details = {
                **cand.details,
                "selection_objective": "maximize_subset_silhouette",
                "subset_silhouette_mean": float(best_sil),  # [-1,1]
                "subset_silhouette_01": float(clamp01((best_sil + 1.0) * 0.5)),  # [0,1]
                "silhouette_metric": silhouette_metric,
                "silhouette_debug": dbg,
                "silhouette_pool_size": len(pool),
                "selected_k": k,
            }

        # 요구사항: 선택된 K개를 inter-MECE(sim) 내림차순으로 정렬해서 반환
        selected.sort(key=lambda c: float(c.mece.inter_mece), reverse=True)
        return selected
