from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from .mece_utils import *
from .mece_common import *


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

        ctx = tokenize_no_special(self.tok, context)
        tgt = tokenize_no_special(self.tok, target)

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
            subtasks_context(subtasks), task_text
        )

        if math.isfinite(nll_task_base) and nll_task_base > 1e-8:
            coverage = clamp01((nll_task_base - nll_task_cond) / nll_task_base)
        else:
            coverage = 0.0

        base_nll = [self.conditional_nll_per_token("", s) for s in subtasks]

        overlaps = []
        for i, si in enumerate(subtasks):
            ctx_i = single_subtask_context(si)
            for j, sj in enumerate(subtasks):
                if i == j:
                    continue
                nll_b = base_nll[j]
                nll_c = self.conditional_nll_per_token(ctx_i, sj)
                if math.isfinite(nll_b) and nll_b > 1e-8:
                    overlaps.append(clamp01((nll_b - nll_c) / nll_b))

        redundancy = sum(overlaps) / max(1, len(overlaps))
        exclusivity = clamp01(1.0 - redundancy)
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
        decomposition: List[str],
        task_text: str,
        *,
        alpha: float = 0.5,
    ) -> MeceScore:
        return self.score_loss_mece(decomposition, task_text, alpha=alpha)

    # -------------------------
    # Decomposition picking
    # -------------------------
    @torch.no_grad()
    def pick_best(
        self,
        *,
        task_text: str,
        task_decomposition_prompt: str,
        num_samples: int = 8,
        alpha: float = 0.5,
        min_subtasks: int = 2,
        max_subtasks: int = 10,
        dedup_raw: bool = True,
        seed: Optional[int] = None,
        return_topk: int = 1,
    ) -> List[DecompCandidate]:
        if seed is not None:
            random.seed(seed)

        seen_raw = set()
        candidates: List[DecompCandidate] = []

        for sample_num in range(num_samples):
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
            print(f"Generate {sample_num}th decomposition.:")
            print(subtasks)
            if not (min_subtasks <= len(subtasks) <= max_subtasks):
                continue

            mece = self.score(subtasks, task_text, alpha=alpha)
            print(f"Generate {sample_num}th subtasks.(score {mece})")

            candidates.append(
                DecompCandidate(
                    subtasks=subtasks,
                    raw=raw,
                    score=mece.inter_mece,
                    mece=mece,
                    details={"alpha": alpha},
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: max(1, return_topk)]


class SimInterMeceEngine:
    """
    Sim-based Inter-MECE Engine (separate class)
    - decomposition sampling
    - MECE scoring (embedding cosine, task-parallel/orthogonal decomposition)
    - best candidate selection

    Definitions:
      task embedding t, subtask embedding e_i
      t_hat = t / ||t||
      v_i  = (e_i · t_hat) t_hat         (horizontal / parallel)
      v_iT = e_i - v_i                    (vertical / orthogonal)

      coverage    = mean_i relu(cos(e_i, t))
      redundancy  = mean_{i<j} |cos(v_iT, v_jT)|
      exclusivity = 1 - redundancy
      inter_mece  = alpha*coverage + (1-alpha)*exclusivity
    """

    def __init__(
        self,
        tm,
        *,
        call_model_fn: Callable[..., str],
        call_model_kwargs: Dict[str, Any],
        embed_texts_fn: Optional[Callable[[List[str]], torch.Tensor]] = None,
        max_length: int = 2048,
        eps: float = 1e-12,
    ):
        self.tm = tm
        self.tok = tm.tokenizer
        self.hf = tm.model
        self.device = self.hf.device
        self.max_length = max_length
        self.eps = eps

        self.call_model_fn = call_model_fn
        self.call_model_kwargs = call_model_kwargs

        # If provided: List[str] -> Tensor [N, D]
        self.embed_texts_fn = embed_texts_fn

    # -------------------------
    # Embeddings
    # -------------------------
    @torch.no_grad()
    def _embed_fallback_lm(self, texts: List[str]) -> torch.Tensor:
        """
        Fallback embedding using underlying HF LM: mean pool last_hidden_state.
        Returns [N, D].
        """
        enc = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

        out = self.hf(input_ids=input_ids, attention_mask=attn)
        hs = out.last_hidden_state  # [N, L, D]

        attn_f = attn.unsqueeze(-1).to(hs.dtype)  # [N, L, 1]
        summed = (hs * attn_f).sum(dim=1)  # [N, D]
        denom = attn_f.sum(dim=1).clamp_min(self.eps)  # [N, 1]
        return summed / denom

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        E = (
            self.embed_texts_fn(texts)
            if self.embed_texts_fn is not None
            else self._embed_fallback_lm(texts)
        )
        if E.device != self.device:
            E = E.to(self.device)
        if E.ndim != 2 or E.shape[0] != len(texts):
            raise ValueError(f"embed_texts must return [N, D]. got {tuple(E.shape)}")
        return E

    # -------------------------
    # MECE scoring (sim-based)
    # -------------------------
    @torch.no_grad()
    def score(
        self,
        mode: str,  # kept for interface-compat; expect "sim"
        decomposition: List[str],
        task_text: str,
        *,
        alpha: float = 0.5,
        coverage_mode: str = "relu_cos_mean",  # or "proj_norm_mean"
        redundancy_mode: str = "abs_cos_mean",  # or "cos2_mean"
    ) -> MeceScore:
        if mode != "sim":
            raise ValueError("SimInterMeceEngine only supports mode='sim'")

        subtasks = [s.strip() for s in decomposition if s and s.strip()]
        if not subtasks:
            return MeceScore(0.0, 0.0, 1.0, 0.0, {"reason": "empty"})

        texts = [task_text] + subtasks
        E = self.embed_texts(texts)  # [1+N, D]
        task_e = E[0]  # [D]
        sub_e = E[1:]  # [N, D]
        N = sub_e.shape[0]

        eps = self.eps

        # task unit vector
        t_hat = task_e / torch.linalg.norm(task_e).clamp_min(eps)  # [D]

        # cos(e_i, t)
        sub_n = sub_e / torch.linalg.norm(sub_e, dim=-1, keepdim=True).clamp_min(eps)
        cos_et = (sub_n * t_hat.unsqueeze(0)).sum(dim=-1)  # [N]

        # decompose: parallel & orthogonal
        coeff = (sub_e * t_hat.unsqueeze(0)).sum(dim=-1)  # [N]
        v_h = coeff.unsqueeze(-1) * t_hat.unsqueeze(0)  # [N, D]
        v_v = sub_e - v_h  # [N, D]

        # ---- coverage ----
        if coverage_mode == "relu_cos_mean":
            coverage_t = torch.relu(cos_et).mean()
        elif coverage_mode == "proj_norm_mean":
            # |e_i·t_hat| / ||e_i||  (≈ |cos|)
            coverage_t = (
                coeff.abs() / torch.linalg.norm(sub_e, dim=-1).clamp_min(eps)
            ).mean()
        else:
            raise ValueError(
                "coverage_mode must be 'relu_cos_mean' or 'proj_norm_mean'"
            )

        coverage = float(clamp01(coverage_t.item()))

        # ---- redundancy (vertical components orthogonality) ----
        v_v_norm = torch.linalg.norm(v_v, dim=-1)  # [N]
        valid = v_v_norm > (10 * eps)
        idx = torch.where(valid)[0]

        if idx.numel() >= 2:
            vv = v_v[idx]  # [M, D]
            vv_n = vv / torch.linalg.norm(vv, dim=-1, keepdim=True).clamp_min(eps)
            C = vv_n @ vv_n.T  # [M, M]
            m = C.shape[0]
            triu = torch.triu(
                torch.ones((m, m), dtype=torch.bool, device=C.device), diagonal=1
            )
            vals = C[triu]  # [M*(M-1)/2]

            if redundancy_mode == "abs_cos_mean":
                redundancy_t = vals.abs().mean()
            elif redundancy_mode == "cos2_mean":
                redundancy_t = vals.pow(2).mean()
            else:
                raise ValueError(
                    "redundancy_mode must be 'abs_cos_mean' or 'cos2_mean'"
                )
        else:
            redundancy_t = torch.tensor(0.0, device=self.device)

        redundancy = float(clamp01(redundancy_t.item()))
        exclusivity = clamp01(1.0 - redundancy)
        inter_mece = clamp01(alpha * coverage + (1.0 - alpha) * exclusivity)

        return MeceScore(
            coverage=coverage,
            redundancy=redundancy,
            exclusivity=exclusivity,
            inter_mece=inter_mece,
            details={
                "alpha": alpha,
                "coverage_mode": coverage_mode,
                "redundancy_mode": redundancy_mode,
                "num_subtasks": N,
                "num_vertical_valid": int(valid.sum().item()),
                "pair_count": int(idx.numel() * (idx.numel() - 1) // 2)
                if idx.numel() >= 2
                else 0,
                "cos_task_sub_mean": float(cos_et.mean().item()),
                "cos_task_sub_min": float(cos_et.min().item()),
                "cos_task_sub_max": float(cos_et.max().item()),
            },
        )

    # -------------------------
    # Decomposition picking
    # -------------------------
    @torch.no_grad()
    def pick_best(
        self,
        *,
        task_text: str,
        task_decomposition_prompt: str,
        mode: str = "sim",
        num_samples: int = 8,
        alpha: float = 0.5,
        min_subtasks: int = 2,
        max_subtasks: int = 10,
        dedup_raw: bool = True,
        seed: Optional[int] = None,
        return_topk: int = 1,
        score_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[DecompCandidate]:
        if seed is not None:
            random.seed(seed)

        score_kwargs = score_kwargs or {}

        seen_raw = set()
        candidates: List[DecompCandidate] = []

        for sample_num in range(num_samples):
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
            print(f"Generate {sample_num}th decomposition.:")
            print(subtasks)

            if not (min_subtasks <= len(subtasks) <= max_subtasks):
                continue

            mece = self.score("sim", subtasks, task_text, alpha=alpha, **score_kwargs)
            print(f"Generate {sample_num}th subtasks.(score {mece})")

            candidates.append(
                DecompCandidate(
                    subtasks=subtasks,
                    raw=raw,
                    score=mece.inter_mece,
                    mece=mece,
                    details={"alpha": alpha, **score_kwargs},
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: max(1, return_topk)]
