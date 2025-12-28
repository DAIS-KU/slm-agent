from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from mece_utils import *
from mece_common import *


class InterMeceEngine:
    """
    Inter-MECE Engine
    - decomposition sampling
    - MECE scoring (loss-based)
    - best candidate selection
    """

    def __init__(
        self,
        tm,
        *,
        call_model_fn: Callable[..., str],
        call_model_kwargs: Dict[str, Any],
        max_length: int = 2048,
    ):
        self.tm = tm
        self.tok = tm.tokenizer
        self.hf = tm.model
        self.device = self.hf.device
        self.max_length = max_length

        self.call_model_fn = call_model_fn
        self.call_model_kwargs = call_model_kwargs

    # -------- loss-based core --------
    @torch.no_grad()
    def conditional_nll_per_token(
        self,
        context: str,
        target: str,
        *,
        max_length: Optional[int] = None,
    ) -> float:
        max_length = max_length or self.max_length

        ctx = _tokenize_no_special(self.tok, context)
        tgt = _tokenize_no_special(self.tok, target)

        ctx_ids = ctx["input_ids"]
        tgt_ids = tgt["input_ids"]

        if tgt_ids.numel() == 0:
            return float("inf")

        input_ids = torch.cat([ctx_ids, tgt_ids], dim=1).to(self.device)

        if max_length is not None and input_ids.shape[1] > max_length:
            tgt_len = tgt_ids.shape[1]
            keep = min(max_length, tgt_len + max(0, max_length - tgt_len))
            input_ids = input_ids[:, -keep:]
            new_ctx_len = max(0, input_ids.shape[1] - tgt_len)
        else:
            tgt_len = tgt_ids.shape[1]
            new_ctx_len = ctx_ids.shape[1]

        logits = self.hf(input_ids=input_ids).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        Lm1 = shift_labels.shape[1]

        start = 0 if new_ctx_len == 0 else (new_ctx_len - 1)
        end = min(start + tgt_len, Lm1)
        if start >= end:
            return float("inf")

        mask = torch.zeros((1, Lm1), dtype=torch.bool, device=self.device)
        mask[:, start:end] = True

        vocab = shift_logits.shape[-1]
        ce = F.cross_entropy(
            shift_logits.reshape(-1, vocab),
            shift_labels.reshape(-1),
            reduction="none",
        ).reshape(1, -1)

        return float(ce[mask].mean().item())

    # -------- MECE (loss-based) --------
    @torch.no_grad()
    def score_loss_mece(
        self,
        decomposition: List[str],
        task_text: str,
        *,
        alpha: float = 0.5,
    ) -> MeceScore:
        subtasks = [s.strip() for s in decomposition if s and s.strip()]
        if not subtasks:
            return MeceScore(0.0, 0.0, 1.0, 0.0, {"reason": "empty"})

        nll_task_base = self.conditional_nll_per_token("", task_text)
        nll_task_cond = self.conditional_nll_per_token(
            _subtasks_context(subtasks), task_text
        )

        if math.isfinite(nll_task_base) and nll_task_base > 1e-8:
            coverage = _clamp01((nll_task_base - nll_task_cond) / nll_task_base)
        else:
            coverage = 0.0

        base_nll = [self.conditional_nll_per_token("", s) for s in subtasks]

        overlaps = []
        for i, si in enumerate(subtasks):
            ctx_i = _single_subtask_context(si)
            for j, sj in enumerate(subtasks):
                if i == j:
                    continue
                nll_b = base_nll[j]
                nll_c = self.conditional_nll_per_token(ctx_i, sj)
                if math.isfinite(nll_b) and nll_b > 1e-8:
                    overlaps.append(_clamp01((nll_b - nll_c) / nll_b))

        redundancy = sum(overlaps) / max(1, len(overlaps))
        exclusivity = _clamp01(1.0 - redundancy)
        inter_mece = alpha * coverage + (1.0 - alpha) * exclusivity

        return MeceScore(
            coverage=coverage,
            redundancy=redundancy,
            exclusivity=exclusivity,
            inter_mece=inter_mece,
            details={
                "alpha": alpha,
                "nll_task_base": nll_task_base,
                "nll_task_cond": nll_task_cond,
                "pair_count": len(overlaps),
            },
        )

    # -------- unified entry --------
    @torch.no_grad()
    def score(
        self,
        mode: str,
        decomposition: List[str],
        task_text: str,
        *,
        alpha: float = 0.5,
    ) -> MeceScore:
        if mode == "loss":
            return self.score_loss_mece(decomposition, task_text, alpha=alpha)
        if mode == "sim":
            raise NotImplementedError("sim-based MECE not implemented yet")
        raise ValueError("mode must be 'loss' or 'sim'")

    # -------------------------
    # Decomposition picking
    # -------------------------
    @torch.no_grad()
    def pick_best(
        self,
        *,
        task_text: str,
        task_decomposition_prompt: str,
        mode: str = "loss",
        num_samples: int = 8,
        alpha: float = 0.5,
        min_subtasks: int = 2,
        max_subtasks: int = 20,
        dedup_raw: bool = True,
        seed: Optional[int] = None,
        return_topk: int = 1,
    ) -> List[DecompCandidate]:
        if seed is not None:
            random.seed(seed)

        seen_raw = set()
        candidates: List[DecompCandidate] = []

        for _ in range(num_samples):
            raw = self.call_model_fn(
                query=task_decomposition_prompt,
                **self.call_model_kwargs,
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

            mece = self.score(mode, subtasks, task_text, alpha=alpha)

            candidates.append(
                DecompCandidate(
                    subtasks=subtasks,
                    raw=raw,
                    score=mece.inter_mece,
                    score_mode=mode,
                    mece=mece,
                    details={"alpha": alpha},
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: max(1, return_topk)]
