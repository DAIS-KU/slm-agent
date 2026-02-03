from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Literal, Tuple
import itertools
import random

import torch
import torch.nn.functional as F
from .mece_common import *

import logging

logger = logging.getLogger(__name__)

Mode = Literal["direction_only", "sim", "surprise", "entropy"]


def format_generate_outline_prompt(task_text: str, existing: List[str]) -> str:
    return f"""
Describe, in one or two sentences, a new approach to solving the task below that does not overlap with existing approaches.
Return only the newly generated approach, excluding the task description and any other plans.

Task: {task_text}
Approaches: {existing}
"""


def format_surprise_prompt(task_text: str, given_outline: str) -> str:
    return f"""
You are given a task and ONE existing approach. Produce a NEW approach that does not overlap with the existing one.
Return only the new approach in one or two sentences.

Task: {task_text}
Existing approach: {given_outline}
New approach:
"""


def format_execute_prompt(task_text: str, outline_text: str) -> str:
    return f"""
You are given a task and an approach. Execute the approach by outlining concrete steps.
Be concise but actionable.

Task: {task_text}
Approach: {outline_text}
Steps:
"""


class OutlineMeceEngine:
    """
    Mode-based Outline Selection Engine (brute-force)
    - sample n outlines via call_model
    - modes:
      sample: just return k from sampled outlines (no scoring)
      sim: maximize sum pairwise cosine distance of embeddings
      surprise: maximize mean pairwise symmetrized NLL where prompt(t,o_i)->target(o_j)
      entropy: maximize mean pairwise JS divergence between next-token distributions under execute prompt(t,o_i)
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

        self.call_model_fn = call_model_fn
        self.call_model_kwargs = call_model_kwargs

        self.max_length = max_length
        self.eps = eps

        self._embed_cache: Dict[str, torch.Tensor] = {}
        self._next_logp_cache: Dict[str, torch.Tensor] = {}
        self._nll_cache: Dict[Tuple[str, str], float] = {}

    # -------------------------
    # Sampling
    # -------------------------
    def sample_outlines(
        self, *, task_text: str, n: int, seed: Optional[int] = None, dedup: bool = True
    ) -> List[str]:
        if seed is not None:
            random.seed(seed)

        outlines: List[str] = []
        seen = set()
        attempts = 1

        while len(outlines) < n:
            logger.info(f"Try to generate {attempts}-th outline.")
            attempts += 1
            prompt = format_generate_outline_prompt(task_text, outlines)
            o = self.call_model_fn(query=prompt, **self.call_model_kwargs)
            o = "New approach: " + str(o)
            logger.info(f"Generated raw outline. :{o}")

            # if not isinstance(o, str):
            #     continue
            # o = o.strip()
            # if not o:
            #     continue

            # if dedup:
            #     key = " ".join(o.split())
            #     if key in seen:
            #         continue
            #     seen.add(key)

            outlines.append(o)
        return outlines

    # -------------------------
    # Embeddings (sim mode)
    # -------------------------
    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
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
            input_ids=input_ids, attention_mask=attn, output_hidden_states=True
        )
        hs = out.hidden_states[-1]  # [B, L, H]

        attn_f = attn.unsqueeze(-1).float()
        emb = (hs * attn_f).sum(dim=1) / attn_f.sum(dim=1).clamp_min(1.0)
        emb = F.normalize(emb, dim=-1)
        return emb

    @torch.no_grad()
    def _get_embed(self, text: str) -> torch.Tensor:
        if text not in self._embed_cache:
            self._embed_cache[text] = self._encode_texts([text])[0].detach()
        return self._embed_cache[text]

    # -------------------------
    # Next-token (entropy mode)
    # -------------------------
    @torch.no_grad()
    def _next_token_logprobs_batch(self, prompts: List[str]) -> torch.Tensor:
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

        lengths = attn.sum(dim=1)
        last_idx = (lengths - 1).clamp_min(0)

        B, L, V = logits.shape
        batch_idx = torch.arange(B, device=logits.device)
        last_logits = logits[batch_idx, last_idx, :]
        return F.log_softmax(last_logits, dim=-1)

    @torch.no_grad()
    def _get_next_logp(self, prompt: str) -> torch.Tensor:
        if prompt not in self._next_logp_cache:
            self._next_logp_cache[prompt] = self._next_token_logprobs_batch([prompt])[
                0
            ].detach()
        return self._next_logp_cache[prompt]

    # -------------------------
    # Surprise: NLL(target | prompt)
    # -------------------------
    @torch.no_grad()
    def _nll_target_given_prompt(self, prompt: str, target: str) -> float:
        key = (prompt, target)
        if key in self._nll_cache:
            return self._nll_cache[key]

        prompt_ids = self.tok(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"].to(self.device)

        full = prompt + target
        full_ids = self.tok(
            full,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"].to(self.device)

        if full_ids.shape[1] <= prompt_ids.shape[1]:
            self._nll_cache[key] = float("inf")
            return self._nll_cache[key]

        out = self.hf(input_ids=full_ids)
        logits = out.logits
        logp = F.log_softmax(logits, dim=-1)

        L = full_ids.shape[1]
        prompt_len = prompt_ids.shape[1]
        nll = 0.0
        for pos in range(prompt_len, L):
            if pos == 0:
                continue
            tok_id = int(full_ids[0, pos].item())
            nll -= float(logp[0, pos - 1, tok_id].item())

        self._nll_cache[key] = nll
        return nll

    # -------------------------
    # JS divergence
    # -------------------------
    @torch.no_grad()
    def _js_div_from_logp(self, logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
        p = logp.exp()
        q = logq.exp()
        m = 0.5 * (p + q)

        logm = (m + self.eps).log()
        kl_pm = (p * (logp - logm)).sum()
        kl_qm = (q * (logq - logm)).sum()
        return 0.5 * (kl_pm + kl_qm)

    # -------------------------
    # Picking
    # -------------------------
    @torch.no_grad()
    def pick_k(
        self,
        *,
        task_text: str,
        outlines: List[str],
        k: int,
        mode: Mode,
        sample_strategy: Literal["first", "random"] = "first",
        seed: Optional[int] = None,
    ) -> OutlinePick:
        if k <= 0:
            return OutlinePick([], [], 0.0, {"reason": "k<=0"})
        if k > len(outlines):
            raise ValueError(f"k({k}) > len(outlines)({len(outlines)})")

        n = len(outlines)

        # ---- NEW: sample mode ----
        if mode == "direction_only":
            if sample_strategy == "random":
                if seed is not None:
                    random.seed(seed)
                idxs = random.sample(range(n), k)
            else:
                idxs = list(range(k))

            return OutlinePick(
                outlines=[outlines[i] for i in idxs],
                indices=idxs,
                score=0.0,
                details={"mode": "sample", "strategy": sample_strategy},
            )

        combos = itertools.combinations(range(n), k)

        if mode == "sim":
            return self._pick_sim(outlines, combos)
        if mode == "surprise":
            return self._pick_surprise(task_text, outlines, combos)
        if mode == "entropy":
            return self._pick_entropy(task_text, outlines, combos)

        raise ValueError(f"Unknown mode: {mode}")

    # ---------- sim ----------
    @torch.no_grad()
    def _pick_sim(self, outlines: List[str], combos) -> OutlinePick:
        embs = [self._get_embed(o) for o in outlines]
        n = len(outlines)
        dist = torch.zeros((n, n), device=self.device)

        for i in range(n):
            for j in range(i + 1, n):
                d = 1.0 - float(torch.dot(embs[i], embs[j]).item())
                dist[i, j] = d
                dist[j, i] = d

        best_score = -1e18
        best = None

        for combo in combos:
            s = 0.0
            for a in range(len(combo)):
                for b in range(a + 1, len(combo)):
                    s += float(dist[combo[a], combo[b]].item())
            if s > best_score:
                best_score = s
                best = combo

        idxs = list(best)
        return OutlinePick(
            outlines=[outlines[i] for i in idxs],
            indices=idxs,
            score=float(best_score),
            details={"mode": "sim", "objective": "sum_pairwise_cosine_distance"},
        )

    # ------- surprise --------
    @torch.no_grad()
    def _pick_surprise(
        self, task_text: str, outlines: List[str], combos
    ) -> OutlinePick:
        n = len(outlines)
        nll = [[0.0] * n for _ in range(n)]

        for i in range(n):
            p = format_surprise_prompt(task_text, outlines[i])
            for j in range(n):
                if i == j:
                    nll[i][j] = 0.0
                else:
                    nll[i][j] = self._nll_target_given_prompt(p, outlines[j])

        best_score = -1e18
        best = None

        for combo in combos:
            vals = []
            for a in range(len(combo)):
                for b in range(a + 1, len(combo)):
                    i, j = combo[a], combo[b]
                    vals.append(0.5 * (nll[i][j] + nll[j][i]))
            score = float(sum(vals) / (len(vals) + 1e-12))
            if score > best_score:
                best_score = score
                best = combo

        idxs = list(best)
        return OutlinePick(
            outlines=[outlines[i] for i in idxs],
            indices=idxs,
            score=float(best_score),
            details={"mode": "surprise", "objective": "mean_pairwise_symmetrized_NLL"},
        )

    # ------- entropy ---------
    @torch.no_grad()
    def _pick_entropy(self, task_text: str, outlines: List[str], combos) -> OutlinePick:
        prompts = [format_execute_prompt(task_text, o) for o in outlines]
        logPs = [self._get_next_logp(p) for p in prompts]
        n = len(outlines)

        js = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = float(self._js_div_from_logp(logPs[i], logPs[j]).item())
                js[i][j] = d
                js[j][i] = d

        best_score = -1e18
        best = None

        for combo in combos:
            vals = []
            for a in range(len(combo)):
                for b in range(a + 1, len(combo)):
                    vals.append(js[combo[a]][combo[b]])
            score = float(sum(vals) / (len(vals) + 1e-12))
            if score > best_score:
                best_score = score
                best = combo

        idxs = list(best)
        return OutlinePick(
            outlines=[outlines[i] for i in idxs],
            indices=idxs,
            score=float(best_score),
            details={"mode": "entropy", "objective": "mean_pairwise_JS_next_token"},
        )

    # -------------------------
    # Convenience
    # -------------------------
    @torch.no_grad()
    def sample_and_pick(
        self,
        *,
        task_text: str,
        n: int,
        k: int,
        mode: Mode,
        seed: Optional[int] = None,
        dedup: bool = True,
        sample_strategy: Literal["first", "random"] = "first",
    ) -> OutlinePick:
        outs = self.sample_outlines(task_text=task_text, n=n, seed=seed, dedup=dedup)
        return self.pick_k(
            task_text=task_text,
            outlines=outs,
            k=k,
            mode=mode,
            sample_strategy=sample_strategy,
            seed=seed,
        )
