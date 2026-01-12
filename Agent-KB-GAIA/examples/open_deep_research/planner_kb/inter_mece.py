from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from .mece_utils import *
from .mece_common import *

import logging

logger = logging.getLogger(__name__)


def format_execute_prompt(
    *,
    task_text: str,
    outline_text: Optional[str],
    subtask_text: str,
) -> str:
    """
    (t, o, s_i)를 주고 '실행하라'고 했을 때의 분포를 보고 싶으므로,
    "NEXT:" 형태로 모델이 다음 토큰을 생성하게 만드는 cue를 둡니다.
    """
    parts: List[str] = []
    parts.append("TASK:\n" + task_text.strip())

    if outline_text is not None and outline_text.strip():
        parts.append("OUTLINE:\n" + outline_text.strip())

    parts.append("SUBTASK:\n" + subtask_text.strip())
    parts.append("\nExecute the subtask. Produce the next step.\n\nNEXT:\n")
    return "\n\n".join(parts)


# -------------------------
# JS only
# -------------------------
class EntropyInterMeceEngine:
    """
    Entropy / Divergence-based Inter-MECE Engine (JS only)
    - decomposition sampling
    - for each subtask s_i: get next-token distribution p_i under prompt (t, o?, s_i, "execute")
    - score candidate set by maximizing mean pairwise JS divergence between {p_i}

    JS(p,q) = 0.5 KL(p||m) + 0.5 KL(q||m), m = 0.5(p+q)
    """

    def __init__(
        self,
        tm,
        *,
        call_model_fn: Callable[..., str],
        call_model_kwargs: Dict[str, Any],
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

    # -------------------------
    # Next-token distributions
    # -------------------------
    @torch.no_grad()
    def _next_token_logprobs_batch(self, prompts: List[str]) -> torch.Tensor:
        """
        Returns log-prob vectors for next token at end of each prompt: [B, V]
        """
        enc = self.tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(self.device)
        attn = enc.get("attention_mask", torch.ones_like(input_ids)).to(self.device)

        out = self.hf(input_ids=input_ids, attention_mask=attn)
        logits = out.logits  # [B, L, V]

        lengths = attn.sum(dim=1)  # [B]
        last_idx = (lengths - 1).clamp_min(0)  # [B]

        B, L, V = logits.shape
        batch_idx = torch.arange(B, device=logits.device)
        last_logits = logits[batch_idx, last_idx, :]  # [B, V]

        return F.log_softmax(last_logits, dim=-1)  # [B, V]

    # -------------------------
    # JS divergence
    # -------------------------
    @torch.no_grad()
    def _js_div_from_logp(self, logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
        """
        Jensen–Shannon divergence between two distributions given log-probs.
        Returns scalar tensor.
        """
        p = logp.exp()
        q = logq.exp()
        m = 0.5 * (p + q)

        logm = (m + self.eps).log()
        kl_pm = (p * (logp - logm)).sum()
        kl_qm = (q * (logq - logm)).sum()
        return 0.5 * (kl_pm + kl_qm)

    # -------------------------
    # Scoring: maximize pairwise JS
    # -------------------------
    @torch.no_grad()
    def score_js_diversity(
        self,
        decomposition: List[str],
        task_text: str,
        *,
        outline_text: Optional[str] = None,
    ) -> EntropyScore:
        subtasks = [s.strip() for s in decomposition if s and s.strip()]
        if len(subtasks) < 2:
            return EntropyScore(
                pairwise_js_mean=0.0,
                pairwise_js_min=0.0,
                pairwise_js_max=0.0,
                pair_count=0,
                details={"reason": "too_few_subtasks"},
            )

        prompts = [
            format_execute_prompt(
                task_text=task_text,
                outline_text=outline_text,
                subtask_text=s,
            )
            for s in subtasks
        ]

        logP = self._next_token_logprobs_batch(prompts)  # [N, V]
        N = logP.shape[0]

        divs: List[torch.Tensor] = []
        for i in range(N):
            for j in range(i + 1, N):
                divs.append(self._js_div_from_logp(logP[i], logP[j]))

        if not divs:
            return EntropyScore(0.0, 0.0, 0.0, 0, {"reason": "no_pairs"})

        div_t = torch.stack(divs)  # [pairs]
        return EntropyScore(
            pairwise_js_mean=float(div_t.mean().item()),
            pairwise_js_min=float(div_t.min().item()),
            pairwise_js_max=float(div_t.max().item()),
            pair_count=len(divs),
            details={
                "num_subtasks": N,
                "used_outline": bool(outline_text is not None and outline_text.strip()),
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
        outline_text: Optional[str] = None,
        num_samples: int = 8,
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
            subtasks = []
            while subtasks == []:
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
                if subtasks == []:
                    logger.info("Failed to generate subtasks.")
            logger.info(f"Generate {sample_num}th decomposition.")
            logger.info(subtasks)

            if not (min_subtasks <= len(subtasks) <= max_subtasks):
                continue

            ent = self.score_js_diversity(
                subtasks,
                task_text,
                outline_text=outline_text,
            )

            # maximize mean JS divergence
            score = ent.pairwise_js_mean

            candidates.append(
                DecompCandidate(
                    subtasks=subtasks,
                    raw=raw,
                    score=score,
                    entropy=ent,
                    details={"used_outline": ent.details.get("used_outline", False)},
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: max(1, return_topk)]


# -------------------------
# Surprise-based
# -------------------------
class SurpriseInterMeceEngine:
    """
    Surprise-based Engine
    - decomposition sampling
    - score by total surprise = sum_i -log P(si | prompt)
    - pick min or max total surprise candidate
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

    # -------- core: -log P(target | context) (sum over target tokens) --------
    @torch.no_grad()
    def conditional_surprise_sum(
        self,
        context: str,
        target: str,
        *,
        max_length: Optional[int] = None,
    ) -> float:
        """
        Returns: surprise = -log P(target | context) = sum_t CE_t  (teacher forcing)
        """
        max_length = max_length or self.max_length

        ctx = tokenize_no_special(self.tok, context)
        tgt = tokenize_no_special(self.tok, target)

        ctx_ids = ctx["input_ids"]
        tgt_ids = tgt["input_ids"]

        if tgt_ids.numel() == 0:
            return float("inf")

        input_ids = torch.cat([ctx_ids, tgt_ids], dim=1).to(self.device)

        # Truncation (keep the tail so that all target tokens remain if possible)
        if max_length is not None and input_ids.shape[1] > max_length:
            tgt_len = tgt_ids.shape[1]
            keep = min(max_length, tgt_len + max(0, max_length - tgt_len))
            input_ids = input_ids[:, -keep:]
            new_ctx_len = max(0, input_ids.shape[1] - tgt_len)
        else:
            tgt_len = tgt_ids.shape[1]
            new_ctx_len = ctx_ids.shape[1]

        logits = self.hf(input_ids=input_ids).logits  # [1, L, V]

        shift_logits = logits[:, :-1, :]  # predicts next token
        shift_labels = input_ids[:, 1:]  # next token labels
        Lm1 = shift_labels.shape[1]

        # target tokens in shift space:
        # If context length is new_ctx_len, the first target label position is:
        # start = (new_ctx_len - 1) if new_ctx_len>0 else 0
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
        ).reshape(
            1, -1
        )  # per-token NLL

        # sum over target tokens = -log P(target|context)
        return float(ce[mask].sum().item())

    # -------- scoring: total surprise across subtasks --------
    @torch.no_grad()
    def score_surprise(
        self,
        decomposition: List[str],
        task_text: str,
        *,
        outline_text: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> SurpriseScore:
        subtasks = [s.strip() for s in decomposition if s and s.strip()]
        if not subtasks:
            return SurpriseScore(
                total_surprise=float("inf"),
                per_subtask_surprise=[],
                details={"reason": "empty"},
            )

        per: List[float] = []
        prev: List[str] = []

        for i, si in enumerate(subtasks):
            prompt = format_prompt(
                task_text=task_text,
                outline_text=outline_text,
                prev_subtasks=prev,
            )
            # NOTE: 모델이 "NEXT SUBTASK:" 다음에 si를 생성한다고 가정하고,
            #       teacher forcing으로 si의 조건부 확률을 계산.
            s_i = self.conditional_surprise_sum(prompt, si, max_length=max_length)
            per.append(s_i)
            prev.append(si)

        total = float(sum(per))
        return SurpriseScore(
            total_surprise=total,
            per_subtask_surprise=per,
            details={
                "used_outline": bool(outline_text is not None and outline_text.strip()),
                "num_subtasks": len(subtasks),
            },
        )

    # -------- pick best candidate by min/max total surprise --------
    @torch.no_grad()
    def pick_best(
        self,
        *,
        task_text: str,
        task_decomposition_prompt: str,
        outline_text: Optional[str] = None,
        num_samples: int = 8,
        min_subtasks: int = 2,
        max_subtasks: int = 10,
        dedup_raw: bool = True,
        seed: Optional[int] = None,
        return_topk: int = 1,
        objective: Literal[
            "min", "max"
        ] = "min",  # "min" => 최소 surprise, "max" => 최대 surprise
        max_length: Optional[int] = None,
    ) -> List[DecompCandidate]:
        if seed is not None:
            random.seed(seed)

        seen_raw = set()
        candidates: List[DecompCandidate] = []

        for sample_num in range(num_samples):
            subtasks = []
            while subtasks == []:
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
                if subtasks == []:
                    logger.info("Failed to generate subtasks.")
            logger.info(f"Generate {sample_num}th decomposition.:")
            logger.info(subtasks)

            if not (min_subtasks <= len(subtasks) <= max_subtasks):
                continue

            sscore = self.score_surprise(
                subtasks,
                task_text,
                outline_text=outline_text,
                max_length=max_length,
            )

            # selection score:
            # - objective=min: smaller total_surprise is better => sort ascending by total_surprise
            # - objective=max: larger total_surprise is better => sort descending by total_surprise
            # 여기서는 일관되게 "score가 클수록 좋다"로 맞추기 위해,
            # min이면 score = -total_surprise, max이면 score = +total_surprise로 둡니다.
            if objective == "min":
                score = -sscore.total_surprise
            elif objective == "max":
                score = sscore.total_surprise
            else:
                raise ValueError("objective must be 'min' or 'max'")

            logger.info(
                f"Candidate {sample_num}: total_surprise={sscore.total_surprise:.4f}, score={score:.4f}"
            )

            candidates.append(
                DecompCandidate(
                    subtasks=subtasks,
                    raw=raw,
                    score=float(score),
                    surprise=sscore,
                    details={
                        "objective": objective,
                        "used_outline": sscore.details.get("used_outline", False),
                    },
                )
            )

        # score 큰 순으로 정렬(위에서 min/max를 score 변환으로 통일했기 때문)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: max(1, return_topk)]


# -------------------------
# redundancy-only, optional outline-aware vertical
# -------------------------
class SimInterMeceEngine:
    """
    Redundancy-only Inter-MECE Engine
    - decomposition sampling
    - redundancy scoring (vertical / orthogonal decomposition)
    - best candidate selection

    Definitions:
      task embedding t, subtask embedding e_i
      t_hat = t / ||t||

      (optional) outline embedding o
      o_hat = o / ||o||

    Vertical component:
      - If outline_text is None:
          v_v = e_i - proj_{t_hat}(e_i)
      - If outline_text is provided:
          Remove components parallel to BOTH task and outline directions
          using a 2D orthonormal basis (Gram-Schmidt):
            b1 = t_hat
            b2 = normalize(o_hat - (o_hat·b1)b1)   (if non-degenerate)
          then:
            v_v = e_i - proj_{b1}(e_i) - proj_{b2}(e_i)

    redundancy  = mean_{i<j} |cos(v_v_i, v_v_j)|   (or mean cos^2)
    exclusivity = 1 - redundancy
    inter_mece  = exclusivity
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

        out = self.hf(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,  # embedding 용도면 끄는 게 보통 이득
        )
        # CausalLMOutputWithPast 에서는 hidden_states[-1]가 정답
        hs = out.hidden_states[-1]  # [N, L, D]

        attn_f = attn.unsqueeze(-1).to(hs.dtype)  # [N, L, 1]
        summed = (hs * attn_f).sum(dim=1)  # [N, D]
        denom = attn_f.sum(dim=1).clamp_min(self.eps)
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
    # Redundancy-only scoring
    # -------------------------
    @torch.no_grad()
    def score_redundancy_only(
        self,
        decomposition: List[str],
        task_text: str,
        *,
        outline_text: Optional[str] = None,
        redundancy_mode: str = "abs_cos_mean",  # or "cos2_mean"
    ) -> SimScore:
        subtasks = [s.strip() for s in decomposition if s and s.strip()]
        if not subtasks:
            return SimScore(
                redundancy=0.0,
                exclusivity=1.0,
                inter_mece=1.0,
                details={"reason": "empty"},
            )

        eps = self.eps

        # Embed:
        # - always need task to define vertical
        # - optionally outline adds another "to-be-removed" parallel direction
        if outline_text is not None and outline_text.strip():
            texts = [task_text, outline_text] + subtasks
            E = self.embed_texts(texts)  # [2+N, D]
            task_e = E[0]
            outline_e = E[1]
            sub_e = E[2:]
            used_outline = True
        else:
            texts = [task_text] + subtasks
            E = self.embed_texts(texts)  # [1+N, D]
            task_e = E[0]
            outline_e = None
            sub_e = E[1:]
            used_outline = False

        N = sub_e.shape[0]

        # ---- Build orthonormal basis to remove parallel components ----
        # b1 = normalized task
        b1 = task_e / torch.linalg.norm(task_e).clamp_min(eps)  # [D]

        # v_v = e - proj_b1(e) [- proj_b2(e) if outline provided]
        coeff1 = (sub_e * b1.unsqueeze(0)).sum(dim=-1)  # [N]
        proj1 = coeff1.unsqueeze(-1) * b1.unsqueeze(0)  # [N, D]
        v_v = sub_e - proj1  # [N, D]

        b2 = None
        if used_outline and outline_e is not None:
            o_hat = outline_e / torch.linalg.norm(outline_e).clamp_min(eps)  # [D]
            # Gram-Schmidt: make outline orthogonal to b1
            o_ortho = o_hat - (o_hat * b1).sum() * b1  # [D]
            o_ortho_norm = torch.linalg.norm(o_ortho)

            # If outline is not almost colinear with task, use it as b2
            if o_ortho_norm > (10 * eps):
                b2 = o_ortho / o_ortho_norm.clamp_min(eps)  # [D]
                coeff2 = (v_v * b2.unsqueeze(0)).sum(dim=-1)  # [N]
                proj2 = coeff2.unsqueeze(-1) * b2.unsqueeze(0)  # [N, D]
                v_v = v_v - proj2

        # ---- redundancy on (task & outline)-vertical components ----
        v_v_norm = torch.linalg.norm(v_v, dim=-1)  # [N]
        valid = v_v_norm > (10 * eps)
        idx = torch.where(valid)[0]

        if idx.numel() >= 2:
            vv = v_v[idx]  # [M, D]
            vv_n = vv / torch.linalg.norm(vv, dim=-1, keepdim=True).clamp_min(eps)
            C = vv_n @ vv_n.T  # [M, M]
            m = C.shape[0]
            triu = torch.triu(
                torch.ones((m, m), dtype=torch.bool, device=C.device),
                diagonal=1,
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

        redundancy = clamp01(float(redundancy_t.item()))
        exclusivity = clamp01(1.0 - redundancy)
        inter_mece = exclusivity  # objective: minimize redundancy

        return SimScore(
            redundancy=redundancy,
            exclusivity=exclusivity,
            inter_mece=inter_mece,
            details={
                "objective": "redundancy_only",
                "redundancy_mode": redundancy_mode,
                "num_subtasks": N,
                "used_outline": bool(used_outline and (b2 is not None)),
                "num_vertical_valid": int(valid.sum().item()),
                "pair_count": (
                    int(idx.numel() * (idx.numel() - 1) // 2) if idx.numel() >= 2 else 0
                ),
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
        outline_text: Optional[str] = None,
        num_samples: int = 8,
        min_subtasks: int = 2,
        max_subtasks: int = 10,
        dedup_raw: bool = True,
        seed: Optional[int] = None,
        return_topk: int = 1,
        redundancy_mode: str = "abs_cos_mean",
    ) -> List[DecompCandidate]:
        if seed is not None:
            random.seed(seed)

        seen_raw = set()
        candidates: List[DecompCandidate] = []

        for sample_num in range(num_samples):
            subtasks = []
            while subtasks == []:
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
                if subtasks == []:
                    logger.info("Failed to generate subtasks.")
            logger.info(f"Generate {sample_num}th decomposition.")
            logger.info(subtasks)

            if not (min_subtasks <= len(subtasks) <= max_subtasks):
                continue

            mece = self.score_redundancy_only(
                subtasks,
                task_text,
                outline_text=outline_text,
                redundancy_mode=redundancy_mode,
            )
            logger.info(f"Generate {sample_num}th subtasks. (score {mece.inter_mece})")

            candidates.append(
                DecompCandidate(
                    subtasks=subtasks,
                    raw=raw,
                    score=mece.inter_mece,  # == 1 - redundancy
                    mece=mece,
                    details={
                        "redundancy_mode": redundancy_mode,
                        "outline_used": mece.details.get("used_outline", False),
                    },
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: max(1, return_topk)]
