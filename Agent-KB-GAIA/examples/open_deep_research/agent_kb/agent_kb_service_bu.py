"""agent_kb_service_bu.py

Bu-taxonomy 기반 KB 검색 서비스 (port 8006).

주요 엔드포인트:
  POST /search/taxonomy   — 핵심: 계층 분류 → 필터링 → hybrid 리랭킹 → dynamic_proposal
  POST /search/hybrid     — taxonomy 없이 전체 hybrid 검색
  POST /search/text       — TF-IDF 검색
  POST /search/semantic   — Sentence-embedding 검색
  GET  /performance       — 통계

실행:
  cd agent_kb
  python agent_kb_service_bu.py
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent_kb_retrieval_bu import (
    BUAKB_Manager,
    DynamicProposalPlanner,
    HierarchicalTaxonomyClassifier,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_SEARCHES", 10))
CACHE_TTL      = int(os.getenv("CACHE_TTL", 60))
PORT           = int(os.getenv("BU_KB_PORT", 8006))

_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Agent KB Service (bu-taxonomy)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# 기동 시 자원 로드
# ─────────────────────────────────────────────────────────────────────────────
manager = BUAKB_Manager(
    json_file_paths=[os.path.join(_DIR, "kb_bu_augmented.json")]
)

with open(os.path.join(_DIR, "taxonomy_bu_tree.json"), "r", encoding="utf-8") as _f:
    _taxonomy_tree = json.load(_f)

with open(os.path.join(_DIR, "mapping_table_bu.json"), "r", encoding="utf-8") as _f:
    _mapping_table = json.load(_f)

_classifier_model_name = os.getenv("RETRIEVAL_MODEL_NAME", "gpt-4.1")
_classifier_key        = os.getenv("OPENAI_API_KEY", "")
_classifier_url        = os.getenv("OPENAI_BASE_URL", None)

taxonomy_classifier = HierarchicalTaxonomyClassifier(
    taxonomy_tree = _taxonomy_tree,
    model_name    = _classifier_model_name,
    key           = _classifier_key,
    url           = _classifier_url,
)
proposal_planner = DynamicProposalPlanner(_mapping_table)

# ─────────────────────────────────────────────────────────────────────────────
# 성능 통계 & 캐시
# ─────────────────────────────────────────────────────────────────────────────
_perf: Dict[str, Any] = {
    "total_requests":    0,
    "avg_response_time": 0.0,
    "last_updated":      time.time(),
}
_cache: Dict[str, Dict[str, Any]] = {}


def _update_perf(elapsed: float):
    total = _perf["avg_response_time"] * _perf["total_requests"]
    _perf["total_requests"]    += 1
    _perf["avg_response_time"]  = (total + elapsed) / _perf["total_requests"]
    _perf["last_updated"]       = time.time()


def _cache_get(key: str) -> Optional[Any]:
    entry = _cache.get(key)
    if entry and time.time() - entry["timestamp"] < CACHE_TTL:
        return entry["data"]
    return None


def _cache_set(key: str, data: Any):
    _cache[key] = {"timestamp": time.time(), "data": data}


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────

class TaxonomySearchRequest(BaseModel):
    """taxonomy 기반 검색 요청.

    tt_path / dom_path 를 넘기지 않으면 query 로 자동 분류.
    """
    query:            str
    top_k:            int = 3
    tt_path:          Optional[Dict[str, str]] = Field(
        default=None,
        description="task_type 경로 {major, intermediate, minor}. 미입력 시 자동 분류.",
    )
    dom_path:         Optional[Dict[str, str]] = Field(
        default=None,
        description="domain 경로 {major, intermediate, minor}. 미입력 시 자동 분류.",
    )
    min_pool_size:    int = 20
    hybrid_weights:   Optional[Dict[str, float]] = Field(
        default_factory=lambda: {"text": 0.5, "semantic": 0.5}
    )
    include_proposal: bool = Field(
        default=True,
        description="dynamic_proposal_planning 결과를 응답에 포함할지 여부.",
    )


class SearchRequest(BaseModel):
    query:   str
    top_k:   int = 3
    weights: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {"text": 0.5, "semantic": 0.5}
    )


class TaskResponse(BaseModel):
    task_id:             str
    task:                str
    true_answer:         Optional[str] = None
    source:              Optional[str] = None
    task_analysis:       Optional[Any] = None
    plan_subtask_action: Optional[Any] = None
    bu_taxonomy_path:    Optional[Any] = None
    total_score:         Optional[float] = None


class TaxonomyTaskResponse(TaskResponse):
    text_score:       Optional[float] = None
    semantic_score:   Optional[float] = None
    classified_tt:    Optional[Any]   = None   # 자동 분류된 task_type 경로
    classified_dom:   Optional[Any]   = None   # 자동 분류된 domain 경로
    dynamic_proposal: Optional[Any]   = None   # build_proposal() 결과


class PerformanceStats(BaseModel):
    total_requests:    int
    avg_response_time: float
    cache_hit_rate:    float


# ─────────────────────────────────────────────────────────────────────────────
# 변환 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _to_base(item: Dict) -> Dict:
    return {
        "task_id":             str(item.get("task_id", "")),
        "task":                item.get("task", ""),
        "true_answer":         item.get("true_answer"),
        "source":              item.get("source"),
        "task_analysis":       item.get("task_analysis"),
        "plan_subtask_action": item.get("plan_subtask_action"),
        "bu_taxonomy_path":    item.get("bu_taxonomy_path"),
        "total_score":         item.get("total_score") or item.get("score"),
    }


def _to_taxonomy_response(
    item:             Dict,
    tt_path:          Optional[Dict],
    dom_path:         Optional[Dict],
    include_proposal: bool,
) -> TaxonomyTaskResponse:
    proposal = None
    if include_proposal:
        config   = proposal_planner.get_config(tt_path, dom_path)
        proposal = proposal_planner.build_proposal(item, config)

    return TaxonomyTaskResponse(
        **_to_base(item),
        text_score       = item.get("text_score"),
        semantic_score   = item.get("semantic_score"),
        classified_tt    = tt_path,
        classified_dom   = dom_path,
        dynamic_proposal = proposal,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/search/taxonomy", response_model=List[TaxonomyTaskResponse])
async def taxonomy_search(request: TaxonomySearchRequest):
    """bu-taxonomy 기반 4단계 검색 파이프라인.

    1. query → (task_type path, domain path) 자동 분류 (미입력 시)
    2. 계층 필터링: minor×minor 교집합 → 풀 부족 시 점진적 확대
    3. hybrid 리랭킹: TF-IDF + Sentence-embedding
    4. dynamic_proposal_planning: mapping_table_bu 조회 → 제공 내용 서브셋 결정
    """
    t0 = time.time()
    cache_key = (
        f"tax|{request.query}|{request.top_k}|{request.tt_path}|"
        f"{request.dom_path}|{request.min_pool_size}|"
        f"{request.hybrid_weights}|{request.include_proposal}"
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        # ── Step 1: 분류 ──────────────────────────────────────────────
        tt_path  = request.tt_path
        dom_path = request.dom_path
        if tt_path is None or dom_path is None:
            auto_tt, auto_dom = taxonomy_classifier.classify(request.query)
            tt_path  = tt_path  or auto_tt
            dom_path = dom_path or auto_dom

        # ── Step 2+3: 필터링 + 리랭킹 ─────────────────────────────────
        results = manager.taxonomy_search(
            query         = request.query,
            tt_path       = tt_path,
            dom_path      = dom_path,
            top_k         = request.top_k,
            min_pool_size = request.min_pool_size,
            weights       = request.hybrid_weights,
        )

        # ── Step 4: dynamic proposal ──────────────────────────────────
        response_data = [
            _to_taxonomy_response(item, tt_path, dom_path, request.include_proposal)
            for item in results
        ]

        _cache_set(cache_key, response_data)
        _update_perf(time.time() - t0)
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Taxonomy search failed: {e}")


@app.post("/search/hybrid", response_model=List[TaskResponse])
async def hybrid_search(request: SearchRequest):
    """taxonomy 없이 전체 KB 대상 hybrid 검색."""
    t0 = time.time()
    cache_key = f"hybrid|{request.query}|{request.top_k}|{request.weights}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        results       = manager.hybrid_search(request.query, request.top_k, request.weights)
        response_data = [TaskResponse(**_to_base(item)) for item in results]
        _cache_set(cache_key, response_data)
        _update_perf(time.time() - t0)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {e}")


@app.post("/search/text", response_model=List[TaskResponse])
async def text_search(request: SearchRequest):
    """TF-IDF 검색."""
    t0 = time.time()
    cache_key = f"text|{request.query}|{request.top_k}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        raw = manager.kb.field_text_search(request.query, request.top_k)
        response_data = [
            TaskResponse(**_to_base(manager._record(r["task_id"], r["score"])))
            for r in raw
        ]
        _cache_set(cache_key, response_data)
        _update_perf(time.time() - t0)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text search failed: {e}")


@app.post("/search/semantic", response_model=List[TaskResponse])
async def semantic_search(request: SearchRequest):
    """Sentence-embedding 검색."""
    t0 = time.time()
    cache_key = f"sem|{request.query}|{request.top_k}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        raw = manager.kb.field_semantic_search(request.query, request.top_k)
        response_data = [
            TaskResponse(**_to_base(manager._record(r["task_id"], r["score"])))
            for r in raw
        ]
        _cache_set(cache_key, response_data)
        _update_perf(time.time() - t0)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {e}")


@app.get("/performance", response_model=PerformanceStats)
async def get_performance():
    now    = time.time()
    active = sum(1 for v in _cache.values() if now - v["timestamp"] < CACHE_TTL)
    return {
        "total_requests":    _perf["total_requests"],
        "avg_response_time": _perf["avg_response_time"],
        "cache_hit_rate":    active / len(_cache) if _cache else 0.0,
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        workers=int(os.getenv("UVICORN_WORKERS", 1)),
        limit_concurrency=MAX_CONCURRENT,
    )
