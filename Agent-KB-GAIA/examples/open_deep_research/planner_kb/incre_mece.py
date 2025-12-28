class IncreMECE:
    """
    5. IncreMECE:
    기존 추론 집합(existing_reasonings)에 새로운 추론(new_reasoning)이 추가될 때,
    Novelty와 Information gain의 가중 점수를 반환한다.

    (information-based 구현)
    - novelty: 새 추론이 기존 추론들과 얼마나 다른가?
        novelty = min_i JSD(new, old_i)  (old가 없으면 1)
    - information_gain: task 관점에서 coverage 개선량
        coverage(texts) = 1 - JSD(task, union(texts))
        info_gain = max(0, coverage_after - coverage_before)

    total = normalized( wN*novelty + wG*information_gain )
    """

    def __init__(
        self,
        task: Optional[str] = None,
        existing_reasonings: Optional[Sequence[str]] = None,
        new_reasoning: Union[str, Sequence[str], None] = None,
        *,
        tokenizer: Tokenizer = default_tokenize,
        novelty_weight: float = 0.5,
        info_gain_weight: float = 0.5,
    ):
        self.task = task or ""
        self.existing = [
            s for s in (_clean_subtask(x) for x in (existing_reasonings or [])) if s
        ]
        if new_reasoning is None:
            self.new = []
        elif isinstance(new_reasoning, str):
            nr = _clean_subtask(new_reasoning)
            self.new = [nr] if nr else []
        else:
            self.new = [s for s in (_clean_subtask(x) for x in new_reasoning) if s]

        self.tokenizer = tokenizer
        self.novelty_weight = float(novelty_weight)
        self.info_gain_weight = float(info_gain_weight)

    def calc(
        self,
        mode: str = "information-based",
        *,
        task: Optional[str] = None,
        existing_reasonings: Optional[Sequence[str]] = None,
        new_reasoning: Union[str, Sequence[str], None] = None,
    ) -> Dict[str, float]:
        if mode != "information-based":
            raise NotImplementedError(f"{mode} IncreMECE not implemented.")

        if task is not None:
            self.task = task
        if existing_reasonings is not None:
            self.existing = [
                s for s in (_clean_subtask(x) for x in existing_reasonings) if s
            ]
        if new_reasoning is not None:
            if isinstance(new_reasoning, str):
                nr = _clean_subtask(new_reasoning)
                self.new = [nr] if nr else []
            else:
                self.new = [s for s in (_clean_subtask(x) for x in new_reasoning) if s]

        task_tokens = self.tokenizer(self.task)
        task_dist = _normalize_counter(Counter(task_tokens)) if task_tokens else {}

        def union_dist(texts: Sequence[str]) -> Dict[str, float]:
            toks: List[str] = []
            for t in texts:
                toks.extend(self.tokenizer(t))
            if not toks:
                return {}
            return _normalize_counter(Counter(toks))

        def coverage(u: Dict[str, float]) -> float:
            if not task_dist or not u:
                return 0.0
            return _clamp01(1.0 - _js_divergence(task_dist, u, base=2.0))

        before_union = union_dist(self.existing)
        after_union = union_dist(list(self.existing) + list(self.new))

        cov_before = coverage(before_union)
        cov_after = coverage(after_union)
        info_gain = _clamp01(max(0.0, cov_after - cov_before))

        new_text = " ".join(self.new).strip()
        if not new_text:
            novelty = 0.0
        elif not self.existing:
            novelty = 1.0
        else:
            new_dist = _normalize_counter(Counter(self.tokenizer(new_text)))
            novelty = _clamp01(
                min(
                    _js_divergence(
                        new_dist,
                        _normalize_counter(Counter(self.tokenizer(old))),
                        base=2.0,
                    )
                    for old in self.existing
                )
            )

        w_sum = self.novelty_weight + self.info_gain_weight
        if w_sum <= 1e-12:
            w_n = 0.5
            w_g = 0.5
        else:
            w_n = self.novelty_weight / w_sum
            w_g = self.info_gain_weight / w_sum

        total = _clamp01(w_n * novelty + w_g * info_gain)
        return {
            "novelty": novelty,
            "information_gain": info_gain,
            "total": total,
            "coverage_before": cov_before,
            "coverage_after": cov_after,
        }
