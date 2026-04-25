"""agent_kb_retrieval_bu.py

Bu-taxonomy 기반 KB 조회 모듈.
- kb_bu_augmented.json 을 로드
- taxonomy_bu_tree.json 의 _lookup 을 이용해 query → (task_type path, domain path) 분류
- 계층적 필터링(minor → intermediate → major) 후 hybrid(TF-IDF + semantic) 리랭킹
- mapping_table_bu.json 을 이용해 dynamic_proposal_planning 수행
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from agent_kb_utils import call_model

import logging
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BUTaxonomyPath:
    """bu_taxonomy_path 구조 (task_type 축 + domain 축)."""

    task_type_major: str = ""
    task_type_major_label: str = ""
    task_type_inter: str = ""
    task_type_inter_label: str = ""
    task_type_minor: str = ""
    task_type_minor_label: str = ""
    domain_major: str = ""
    domain_major_label: str = ""
    domain_inter: str = ""
    domain_inter_label: str = ""
    domain_minor: str = ""
    domain_minor_label: str = ""
    task_type_keyword: str = ""
    domain_keyword: str = ""

    @staticmethod
    def from_dict(bp: Dict) -> "BUTaxonomyPath":
        tt  = bp.get("task_type") or {}
        dom = bp.get("domain") or {}
        return BUTaxonomyPath(
            task_type_major       = tt.get("major", ""),
            task_type_major_label = tt.get("major_label", ""),
            task_type_inter       = tt.get("intermediate", ""),
            task_type_inter_label = tt.get("inter_label", ""),
            task_type_minor       = tt.get("minor", ""),
            task_type_minor_label = tt.get("minor_label", ""),
            domain_major          = dom.get("major", ""),
            domain_major_label    = dom.get("major_label", ""),
            domain_inter          = dom.get("intermediate", ""),
            domain_inter_label    = dom.get("inter_label", ""),
            domain_minor          = dom.get("minor", ""),
            domain_minor_label    = dom.get("minor_label", ""),
            task_type_keyword     = bp.get("task_type_keyword", ""),
            domain_keyword        = bp.get("domain_keyword", ""),
        )

    def to_dict(self) -> Dict:
        return {
            "task_type": {
                "major":       self.task_type_major,
                "major_label": self.task_type_major_label,
                "intermediate": self.task_type_inter,
                "inter_label": self.task_type_inter_label,
                "minor":       self.task_type_minor,
                "minor_label": self.task_type_minor_label,
            },
            "domain": {
                "major":       self.domain_major,
                "major_label": self.domain_major_label,
                "intermediate": self.domain_inter,
                "inter_label": self.domain_inter_label,
                "minor":       self.domain_minor,
                "minor_label": self.domain_minor_label,
            },
            "task_type_keyword": self.task_type_keyword,
            "domain_keyword":    self.domain_keyword,
        }


@dataclass
class BUTaskInstance:
    """kb_bu_augmented.json 레코드 한 건."""

    task_id:             str
    task:                str
    true_answer:         str
    source:              str = ""
    task_analysis:       Optional[Dict[str, Any]] = None  # knowledge, plan, subtasks, bu_taxonomy_path
    plan_subtask_action: Optional[List[Dict]]      = None  # [{step, step_action, subtasks:[{subtask, subtask_action}]}]
    taxonomy_path:       Optional[BUTaxonomyPath]  = None
    task_embedding:      Optional[np.ndarray]      = None


# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy Classifier  (LLM-based hierarchical, 3 calls × 2 axes = 6 calls)
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalTaxonomyClassifier:
    """taxonomy_bu_tree.json 의 _lookup 을 이용해 query → 3계층 경로 분류.

    각 축(task_type / domain)마다 3번의 LLM 호출:
      1. major  목록 중 선택
      2. 선택된 major 의 intermediate 목록 중 선택
      3. 선택된 intermediate 의 minor 목록 중 선택
    파싱 실패 시 해당 레벨의 첫 번째 항목으로 fallback (항상 반환 보장).
    """

    def __init__(
        self,
        taxonomy_tree: Dict,
        model_name: str,
        key: str,
        url: str,
        model: Any = None,
        slm: bool = False,
    ):
        self.model_name = model_name
        self.key        = key
        self.url        = url
        self.model      = model
        self.slm        = slm

        lookup  = taxonomy_tree.get("_lookup") or {}
        tt_lk   = lookup.get("task_type") or {}
        dom_lk  = lookup.get("domain")    or {}

        self.tt_minor2path:  Dict[str, Dict] = tt_lk.get("by_minor_id")  or {}
        self.dom_minor2path: Dict[str, Dict] = dom_lk.get("by_minor_id") or {}

        # 계층 트리: {major_id: {label, children: {inter_id: {label, children: {minor_id: label}}}}}
        self.tt_tree  = self._build_tree(self.tt_minor2path)
        self.dom_tree = self._build_tree(self.dom_minor2path)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_tree(by_minor_id: Dict) -> Dict:
        tree: Dict[str, Any] = {}
        for minor_id, path in by_minor_id.items():
            major      = path.get("major", "")
            major_lbl  = path.get("major_label", "")
            inter      = path.get("intermediate", "")
            inter_lbl  = path.get("inter_label", "")
            minor_lbl  = path.get("minor_label", "")
            if not major:
                continue
            if major not in tree:
                tree[major] = {"label": major_lbl, "children": {}}
            if inter and inter not in tree[major]["children"]:
                tree[major]["children"][inter] = {"label": inter_lbl, "children": {}}
            if inter and minor_id:
                tree[major]["children"][inter]["children"][minor_id] = minor_lbl
        return tree

    # ------------------------------------------------------------------
    def _select_one_level(
        self,
        query:      str,
        options:    List[Tuple[str, str]],   # [(id, label), ...]
        axis_name:  str,
        level_name: str,
    ) -> int:
        """LLM 에게 번호로 선택하게 함. 파싱 실패 시 0 반환."""
        options_text = "\n".join(
            f"{i + 1}. {label}" for i, (_, label) in enumerate(options)
        )
        prompt = (
            f"You are classifying a task into a taxonomy.\n"
            f"Axis: {axis_name}  |  Level: {level_name}\n\n"
            f"Task: {query}\n\n"
            f"Options:\n{options_text}\n\n"
            f"Choose the most appropriate option number (1-{len(options)}). "
            f"Respond with only the number."
        )
        try:
            response = call_model(
                query      = prompt,
                model_name = self.model_name,
                key        = self.key,
                url        = self.url,
                model      = self.model,
                slm        = self.slm,
            )
            match = re.search(r"\d+", (response or "").strip())
            if match:
                idx = int(match.group()) - 1
                if 0 <= idx < len(options):
                    return idx
        except Exception as e:
            logger.warning(f"[HierarchicalTaxonomyClassifier] LLM call failed: {e}")
        return 0  # fallback: 첫 번째 항목

    # ------------------------------------------------------------------
    def classify_axis(
        self,
        query:      str,
        tree:       Dict,
        minor2path: Dict,
        axis_name:  str,
    ) -> Dict:
        """3-level LLM 분류. minor_id 의 전체 경로 dict 반환."""
        # ── Level 1: major ──────────────────────────────────────────────
        majors = [(mid, data["label"]) for mid, data in tree.items()]
        if not majors:
            return {}
        idx       = self._select_one_level(query, majors, axis_name, "major")
        major_id  = majors[idx][0]
        logger.info(f"[Classifier/{axis_name}] major → {majors[idx][1]}")

        # ── Level 2: intermediate ────────────────────────────────────────
        inter_dict = tree[major_id]["children"]
        inters     = [(iid, data["label"]) for iid, data in inter_dict.items()]
        if not inters:
            # major에 intermediate가 없으면 major 경로만 반환
            return {"major": major_id, "major_label": majors[idx][1]}
        idx      = self._select_one_level(query, inters, axis_name, "intermediate")
        inter_id = inters[idx][0]
        logger.info(f"[Classifier/{axis_name}] intermediate → {inters[idx][1]}")

        # ── Level 3: minor ───────────────────────────────────────────────
        minor_dict = inter_dict[inter_id]["children"]
        minors     = [(mid, label) for mid, label in minor_dict.items()]
        if not minors:
            return {"major": major_id, "major_label": majors[idx][1],
                    "intermediate": inter_id, "inter_label": inters[idx][1]}
        idx      = self._select_one_level(query, minors, axis_name, "minor")
        minor_id = minors[idx][0]
        logger.info(f"[Classifier/{axis_name}] minor → {minors[idx][1]}")

        return minor2path.get(minor_id) or {
            "major": major_id, "major_label": majors[idx][1],
            "intermediate": inter_id, "inter_label": inters[idx][1],
            "minor": minor_id, "minor_label": minors[idx][1],
        }

    # ------------------------------------------------------------------
    def classify(self, query: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """(task_type_path_dict, domain_path_dict) 반환. 항상 non-None."""
        tt_path  = self.classify_axis(query, self.tt_tree,  self.tt_minor2path,  "task_type") or None
        dom_path = self.classify_axis(query, self.dom_tree, self.dom_minor2path, "domain")    or None
        return tt_path, dom_path


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Proposal Planner
# ─────────────────────────────────────────────────────────────────────────────

class DynamicProposalPlanner:
    """mapping_table_bu.json 을 이용해 KB 조회 결과의 제공 내용을 동적으로 결정.

    mapping table 구조:
      - major       : "TaskTypeMajorLabel|DomainMajorLabel"
      - intermediate: "TaskTypeInterLabel|DomainInterLabel"
      - minor       : "TaskTypeMinorLabel|DomainMinorLabel"

    각 entry 에는:
      - recommended_knowledge : bool
      - recommended_depth     : "plan_only" | "plan_subtask" | "full"
      - field_combo           : "knowledge+plan+subtasks[+actions]"
      - confidence            : "high" | "medium" | "low"
    """

    _FALLBACK: Dict[str, Any] = {
        "recommended_knowledge": True,
        "recommended_depth":     "plan_subtask",
        "field_combo":           "knowledge+plan+subtasks",
        "confidence":            "low",
        "matched_level":         None,
        "matched_key":           None,
    }

    def __init__(self, mapping_table: Dict):
        self.major_table: Dict = mapping_table.get("major")        or {}
        self.inter_table: Dict = mapping_table.get("intermediate") or {}
        self.minor_table: Dict = mapping_table.get("minor")        or {}

    # ------------------------------------------------------------------
    def get_config(
        self,
        tt_path:  Optional[Dict],
        dom_path: Optional[Dict],
    ) -> Dict:
        """분류된 경로에 대해 가장 세밀한 mapping config 를 조회."""
        if tt_path and dom_path:
            checks = [
                ("minor",        self.minor_table, "minor_label",  "minor_label"),
                ("intermediate", self.inter_table, "inter_label",  "inter_label"),
                ("major",        self.major_table, "major_label",  "major_label"),
            ]
            for level, table, tt_key, dom_key in checks:
                tt_label  = tt_path.get(tt_key,  "")
                dom_label = dom_path.get(dom_key, "")
                if tt_label and dom_label:
                    key = f"{tt_label}|{dom_label}"
                    if key in table:
                        return {**table[key], "matched_level": level, "matched_key": key}
        return dict(self._FALLBACK)

    # ------------------------------------------------------------------
    def build_proposal(self, task_record: Dict, config: Dict) -> Dict:
        """config 에 따라 task_record 에서 제공할 내용 서브셋 추출.

        depth 별 포함 범위:
          plan_only    : plan
          plan_subtask : plan + subtasks (subtask 설명만)
          full         : plan + subtasks + actions (plan_subtask_action 전체)
        """
        ta     = task_record.get("task_analysis") or {}
        psa    = task_record.get("plan_subtask_action") or []
        fields = set((config.get("field_combo") or "").split("+"))
        depth  = config.get("recommended_depth", "plan_subtask")

        proposal: Dict[str, Any] = {}

        if config.get("recommended_knowledge", True) and "knowledge" in fields:
            proposal["knowledge"] = ta.get("knowledge")

        if "plan" in fields:
            proposal["plan"] = ta.get("plan")

        if depth in ("plan_subtask", "full") and "subtasks" in fields:
            subs = ta.get("subtasks") or []
            proposal["subtasks"] = [
                s.get("subtask") if isinstance(s, dict) else s
                for s in subs
            ]

        if depth == "full" and "actions" in fields:
            proposal["actions"] = psa

        proposal["_config"] = {
            "recommended_knowledge": config.get("recommended_knowledge", True),
            "recommended_depth":     depth,
            "field_combo":           config.get("field_combo"),
            "confidence":            config.get("confidence"),
            "matched_level":         config.get("matched_level"),
            "matched_key":           config.get("matched_key"),
        }
        return proposal


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────

class BUAgenticKnowledgeBase:
    """kb_bu_augmented.json 기반 지식베이스.

    인덱스:
    - TF-IDF  (task text)
    - Sentence embedding (task text)
    - Taxonomy inverted index (major/inter/minor × task_type/domain)
    """

    def __init__(self, json_file_paths: Optional[List[str]] = None):
        self.tasks: Dict[str, BUTaskInstance] = {}
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.field_components: Dict[str, Any] = {
            "task": {
                "vectorizer": TfidfVectorizer(stop_words="english"),
                "matrix":     None,
                "task_ids":   [],
            }
        }
        # 분류 축별 계층 역색인
        self.tt_major_index:  Dict[str, List[str]] = defaultdict(list)
        self.tt_inter_index:  Dict[str, List[str]] = defaultdict(list)
        self.tt_minor_index:  Dict[str, List[str]] = defaultdict(list)
        self.dom_major_index: Dict[str, List[str]] = defaultdict(list)
        self.dom_inter_index: Dict[str, List[str]] = defaultdict(list)
        self.dom_minor_index: Dict[str, List[str]] = defaultdict(list)

        if json_file_paths:
            self._load(json_file_paths)
            self._finalize()

    # ------------------------------------------------------------------
    def _load(self, paths: List[str]):
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"KB file not found: {p}")
            self._parse(p)

    def _parse(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = (
                    [json.loads(ln) for ln in f if ln.strip()]
                    if path.endswith(".jsonl")
                    else json.load(f)
                )
            if isinstance(data, dict):
                data = [data]

            base = os.path.basename(path)
            for idx, item in enumerate(data):
                try:
                    task_id = str(
                        item.get("_key") or item.get("task_id") or f"{base}_{idx}"
                    )
                    task_text           = item.get("task") or item.get("question", "")
                    true_answer         = item.get("true_answer", "")
                    source              = item.get("source", "")
                    task_analysis       = dict(item.get("task_analysis") or {})
                    plan_subtask_action = item.get("plan_subtask_action") or []

                    # odyseuss_db.jsonl: bu_taxonomy_path is top-level, not inside task_analysis
                    bp = (
                        task_analysis.get("bu_taxonomy_path")
                        or item.get("bu_taxonomy_path")
                        or {}
                    )
                    taxonomy_path = BUTaxonomyPath.from_dict(bp) if bp else None

                    # Normalize odyseuss_db task_analysis: task_type/domain are dicts
                    ta_type = task_analysis.get("task_type")
                    ta_domain = task_analysis.get("domain")
                    if isinstance(ta_type, dict):
                        task_analysis["task_type"] = [
                            v for v in [ta_type.get("raw"), ta_type.get("normalized")] if v
                        ]
                    if isinstance(ta_domain, dict):
                        task_analysis["domain"] = [
                            v for v in [ta_domain.get("raw"), ta_domain.get("normalized")] if v
                        ]
                    # Inject knowledge from decision_augmentation if missing
                    if not task_analysis.get("knowledge"):
                        da = item.get("decision_augmentation") or {}
                        fr = da.get("final_reference") or {}
                        if fr.get("knowledge"):
                            task_analysis["knowledge"] = fr["knowledge"]
                    # Inject plan from agent_planning if missing
                    agent_planning = item.get("agent_planning") or ""
                    if not task_analysis.get("plan") and agent_planning:
                        task_analysis["plan"] = [
                            ln.strip().lstrip("- ").lstrip("* ")
                            for ln in agent_planning.split("\n")
                            if ln.strip()
                        ]

                    self.tasks[task_id] = BUTaskInstance(
                        task_id             = task_id,
                        task                = task_text,
                        true_answer         = true_answer,
                        source              = source,
                        task_analysis       = task_analysis,
                        plan_subtask_action = plan_subtask_action,
                        taxonomy_path       = taxonomy_path,
                    )
                except Exception as e:
                    print(f"[BU-KB] Skipping item {idx}: {e}")
        except Exception as e:
            print(f"[BU-KB] Error parsing {path}: {e}")

    # ------------------------------------------------------------------
    def _finalize(self):
        print("[BU-KB] Building indices...")
        self._build_taxonomy_indices()
        self._build_tfidf()
        self._build_embeddings()

    def _build_taxonomy_indices(self):
        for tid, t in self.tasks.items():
            p = t.taxonomy_path
            if p is None:
                continue
            if p.task_type_major: self.tt_major_index[p.task_type_major].append(tid)
            if p.task_type_inter:  self.tt_inter_index[p.task_type_inter].append(tid)
            if p.task_type_minor:  self.tt_minor_index[p.task_type_minor].append(tid)
            if p.domain_major:    self.dom_major_index[p.domain_major].append(tid)
            if p.domain_inter:    self.dom_inter_index[p.domain_inter].append(tid)
            if p.domain_minor:    self.dom_minor_index[p.domain_minor].append(tid)

    def _build_tfidf(self):
        if not self.tasks:
            return
        ids, texts = zip(*[(tid, t.task) for tid, t in self.tasks.items()])
        self.field_components["task"]["matrix"]   = (
            self.field_components["task"]["vectorizer"].fit_transform(list(texts))
        )
        self.field_components["task"]["task_ids"] = list(ids)

    def _build_embeddings(self):
        print("[BU-KB] Generating embeddings...")
        ts = list(self.tasks.values())
        if not ts:
            return
        embs = self.embedding_model.encode(
            [t.task for t in ts],
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        for i, t in enumerate(ts):
            t.task_embedding = embs[i]

    # ------------------------------------------------------------------
    # Core retrieval: taxonomy-filtered hybrid search
    # ------------------------------------------------------------------

    def taxonomy_filtered_hybrid_search(
        self,
        query:         str,
        tt_path:       Optional[Dict],
        dom_path:      Optional[Dict],
        top_k:         int  = 3,
        min_pool_size: int  = 20,
        weights:       Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """계층 필터링 → hybrid 리랭킹 → top_k 반환.

        필터링 전략:
          1. minor × minor 교집합이 min_pool_size 이상이면 사용
          2. 아니면 한쪽 축을 한 단계 넓히거나, 양쪽 모두 넓히는 방향으로 반복
          3. 여전히 부족하면 major 합집합 또는 전체 사용
        """
        weights = weights or {"text": 0.5, "semantic": 0.5}

        # ── 분류 경로별 역색인 매핑 ──────────────────────────────────────
        _tt_idx  = {"minor": self.tt_minor_index,  "intermediate": self.tt_inter_index,  "major": self.tt_major_index}
        _dom_idx = {"minor": self.dom_minor_index, "intermediate": self.dom_inter_index, "major": self.dom_major_index}

        def _tt_set(level: str) -> Optional[set]:
            if tt_path is None:
                return None
            key = tt_path.get(level)
            return set(_tt_idx[level].get(key, [])) if key else None

        def _dom_set(level: str) -> Optional[set]:
            if dom_path is None:
                return None
            key = dom_path.get(level)
            return set(_dom_idx[level].get(key, [])) if key else None

        def _intersect(tt_l: str, dom_l: str) -> Optional[set]:
            tt_s  = _tt_set(tt_l)
            dom_s = _dom_set(dom_l)
            if tt_s is None and dom_s is None:
                return None
            if tt_s  is None: return dom_s
            if dom_s is None: return tt_s
            return tt_s & dom_s

        # ── 가장 좁은 교집합부터 점진적으로 확대 ────────────────────────
        _level_pairs = [
            ("minor",        "minor"),
            ("minor",        "intermediate"),
            ("intermediate", "minor"),
            ("intermediate", "intermediate"),
            ("minor",        "major"),
            ("major",        "minor"),
            ("intermediate", "major"),
            ("major",        "intermediate"),
            ("major",        "major"),
        ]

        candidates: Optional[set] = None
        for tt_l, dom_l in _level_pairs:
            pool = _intersect(tt_l, dom_l)
            if pool is not None and len(pool) >= min_pool_size:
                candidates = pool
                break

        if not candidates:
            # 합집합 fallback
            tt_all  = _tt_set("major")  or set()
            dom_all = _dom_set("major") or set()
            candidates = (tt_all | dom_all) if (tt_all or dom_all) else set(self.tasks.keys())

        # ── TF-IDF 점수 ──────────────────────────────────────────────────
        cands = list(candidates)
        comp  = self.field_components["task"]
        text_scores: Dict[str, float] = {}
        if comp["matrix"] is not None and comp["task_ids"]:
            qvec      = comp["vectorizer"].transform([query])
            id_to_pos = {tid: pos for pos, tid in enumerate(comp["task_ids"])}
            for tid in cands:
                pos = id_to_pos.get(tid)
                text_scores[tid] = (
                    float((qvec * comp["matrix"][pos].T).toarray()[0][0])
                    if pos is not None else 0.0
                )
        else:
            text_scores = {tid: 0.0 for tid in cands}

        # ── Semantic 점수 ─────────────────────────────────────────────────
        q_emb = self.embedding_model.encode(query, convert_to_numpy=True)
        sem_scores: Dict[str, float] = {}
        for tid in cands:
            emb = self.tasks[tid].task_embedding
            sem_scores[tid] = (
                float(cosine_similarity([q_emb], [emb])[0][0])
                if emb is not None else 0.0
            )

        # ── 최종 점수 & 정렬 ──────────────────────────────────────────────
        w_t = weights.get("text",     0.5)
        w_s = weights.get("semantic", 0.5)
        scored = sorted(
            [(w_t * text_scores[tid] + w_s * sem_scores[tid], tid) for tid in cands],
            reverse=True,
        )
        return [
            {
                "task_id":       tid,
                "score":         score,
                "text_score":    text_scores[tid],
                "semantic_score": sem_scores[tid],
            }
            for score, tid in scored[:top_k]
        ]

    # ------------------------------------------------------------------
    # General-purpose helpers
    # ------------------------------------------------------------------

    def field_text_search(self, query: str, top_k: int = 3) -> List[Dict]:
        comp = self.field_components["task"]
        if comp["matrix"] is None or not comp["task_ids"]:
            return []
        qvec  = comp["vectorizer"].transform([query])
        sims  = cosine_similarity(qvec, comp["matrix"]).flatten()
        top_i = sims.argsort()[-top_k:][::-1]
        return [
            {"task_id": comp["task_ids"][i], "score": float(sims[i])}
            for i in top_i
        ]

    def field_semantic_search(self, query: str, top_k: int = 3) -> List[Dict]:
        q_emb = self.embedding_model.encode(query, convert_to_numpy=True)
        items = [
            (t.task_id, t.task_embedding)
            for t in self.tasks.values()
            if t.task_embedding is not None
        ]
        if not items:
            return []
        ids, embs = zip(*items)
        sims  = cosine_similarity([q_emb], list(embs))[0]
        top_i = sims.argsort()[-top_k:][::-1]
        return [{"task_id": ids[i], "score": float(sims[i])} for i in top_i]


# ─────────────────────────────────────────────────────────────────────────────
# Manager  (서비스 레이어에서 사용하는 공개 API)
# ─────────────────────────────────────────────────────────────────────────────

class BUAKB_Manager:
    def __init__(self, json_file_paths: Optional[List[str]] = None):
        self.kb = BUAgenticKnowledgeBase(json_file_paths=json_file_paths)

    # ------------------------------------------------------------------
    def _record(self, task_id: str, score: float = 0.0, extra: Optional[Dict] = None) -> Dict:
        t = self.kb.tasks.get(task_id)
        if t is None:
            return {}
        return {
            "task_id":             task_id,
            "total_score":         score,
            "task":                t.task,
            "true_answer":         t.true_answer,
            "source":              t.source,
            "task_analysis":       t.task_analysis,
            "plan_subtask_action": t.plan_subtask_action,
            "bu_taxonomy_path":    t.taxonomy_path.to_dict() if t.taxonomy_path else None,
            **(extra or {}),
        }

    # ------------------------------------------------------------------
    def taxonomy_search(
        self,
        query:         str,
        tt_path:       Optional[Dict] = None,
        dom_path:      Optional[Dict] = None,
        top_k:         int  = 3,
        min_pool_size: int  = 20,
        weights:       Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """taxonomy-filtered hybrid search. 전체 레코드 반환."""
        raw = self.kb.taxonomy_filtered_hybrid_search(
            query, tt_path, dom_path, top_k, min_pool_size, weights
        )
        results = []
        for r in raw:
            rec = self._record(r["task_id"], r["score"])
            rec["text_score"]     = r.get("text_score",     0.0)
            rec["semantic_score"] = r.get("semantic_score", 0.0)
            results.append(rec)
        return results

    def hybrid_search(
        self,
        query:   str,
        top_k:   int  = 5,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """일반 hybrid search (taxonomy 필터 없음)."""
        weights = weights or {"text": 0.5, "semantic": 0.5}
        board: Dict[str, float] = defaultdict(float)
        for r in self.kb.field_text_search(query, top_k * 2):
            board[r["task_id"]] += weights["text"] * r["score"]
        for r in self.kb.field_semantic_search(query, top_k * 2):
            board[r["task_id"]] += weights["semantic"] * r["score"]
        ranked = sorted(board.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self._record(tid, score) for tid, score in ranked]

    def get_task(self, task_id: str) -> Optional[BUTaskInstance]:
        return self.kb.tasks.get(task_id)
