from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from agent_kb_retrieval import AKB_Manager
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import time
import os

MAX_CONCURRENT_SEARCHES = int(os.getenv("MAX_CONCURRENT_SEARCHES", 10))
CACHE_TTL = int(os.getenv("CACHE_TTL", 60))

app = FastAPI(title="Optimized Task KB Retrieval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 변환 후 엔티티 DB를 로드한다고 가정
# unified_database.json 내부 각 레코드는 task_id/task/subtasks를 포함해야 함
manager = AKB_Manager(json_file_paths=["./augmented_akb.json"])

performance_stats = {
    "total_requests": 0,
    "avg_response_time": 0.0,
    "last_updated": time.time(),
}

response_cache: Dict[str, Dict[str, Any]] = {}


# -----------------------------
# Pydantic models (변환 후 엔티티 기준)
# -----------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 1
    weights: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {"text": 0.5, "semantic": 0.5}
    )


class TypeDomainSearchRequest(BaseModel):
    query: str
    task_types: List[str]
    domains: List[str]
    top_k: int = 3
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Weighted sum coefficients: {'type': 0.3, 'domain': 0.3, 'text': 0.4}",
    )


class TaskResponse(BaseModel):
    task_id: str
    task: str
    true_answer: Optional[str] = None
    agent_planning: Optional[Any] = None
    agent_experience: Optional[Any] = None
    task_analysis: Optional[Any] = None
    plan_subtask_action: Optional[Any] = None
    total_score: Optional[float] = None


class PerformanceStats(BaseModel):
    total_requests: int
    avg_response_time: float
    cache_hit_rate: float


def update_performance_stats(response_time: float):
    total_time = (
        performance_stats["avg_response_time"] * performance_stats["total_requests"]
    )
    performance_stats["total_requests"] += 1
    performance_stats["avg_response_time"] = (
        total_time + response_time
    ) / performance_stats["total_requests"]
    performance_stats["last_updated"] = time.time()


def _get_cached(cache_key: str):
    cached = response_cache.get(cache_key)
    if not cached:
        return None
    if time.time() - cached["timestamp"] < CACHE_TTL:
        return cached["data"]
    return None


def _set_cached(cache_key: str, data: Any):
    response_cache[cache_key] = {"timestamp": time.time(), "data": data}


def _extract_task_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": str(item["task_id"]),
        "task": item["task"],
        "true_answer": item.get("true_answer", None),
        "agent_planning": item.get("agent_planning", None),
        "agent_experience": item.get("agent_experience", None),
        "task_analysis": item.get("task_analysis", None),
        "plan_subtask_action": item.get("plan_subtask_action", None),
    }


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/search/hybrid", response_model=List[TaskResponse])
async def hybrid_search(request: SearchRequest):
    start_time = time.time()
    cache_key = f"hybrid_{request.query}_{request.top_k}_{request.weights}"

    try:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

        # AKB_Manager.hybrid_search가 변환 후 엔티티를 반환한다고 가정
        results = manager.hybrid_search(
            query=request.query, top_k=request.top_k, weights=request.weights
        )

        response_data: List[TaskResponse] = []
        for item in results:
            core = _extract_task_fields(item)
            response_data.append(
                TaskResponse(
                    task_id=str(core["task_id"]),
                    task=core["task"],
                    true_answer=core["true_answer"],
                    agent_planning=core["agent_planning"],
                    agent_experience=core["agent_experience"],
                    task_analysis=core["task_analysis"],
                    plan_subtask_action=core["plan_subtask_action"],
                    total_score=item.get("total_score"),
                )
            )

        _set_cached(cache_key, response_data)
        update_performance_stats(time.time() - start_time)
        return response_data

    except Exception as e:
        print(f"Hybrid search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


@app.post("/search/text", response_model=List[TaskResponse])
async def text_search(request: SearchRequest):
    start_time = time.time()
    cache_key = f"text_{request.query}_{request.top_k}"

    try:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

        # 기존 코드의 "query" 필드 대신 "task" 필드를 대상으로 검색
        raw_results = manager.search_by_text(request.query, "task", request.top_k)

        response_data: List[TaskResponse] = []
        for item in raw_results:
            # item: {"score":..., "content": {...}} 형태를 가정
            core = _extract_task_fields(item)
            response_data.append(
                TaskResponse(
                    task_id=str(core["task_id"]),
                    task=core["task"],
                    true_answer=core["true_answer"],
                    agent_planning=core["agent_planning"],
                    agent_experience=core["agent_experience"],
                    task_analysis=core["task_analysis"],
                    plan_subtask_action=core["plan_subtask_action"],
                    total_score=item.get("score"),
                )
            )

        _set_cached(cache_key, response_data)
        update_performance_stats(time.time() - start_time)
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text search failed: {str(e)}")


@app.post("/search/semantic", response_model=List[TaskResponse])
async def semantic_search(request: SearchRequest):
    start_time = time.time()
    cache_key = f"semantic_{request.query}_{request.top_k}"

    try:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

        # 기존 코드의 "query" 필드 대신 "task" 필드를 대상으로 검색
        raw_results = manager.search_by_semantic(request.query, "task", request.top_k)

        response_data: List[TaskResponse] = []
        for item in raw_results:
            core = _extract_task_fields(item)
            response_data.append(
                TaskResponse(
                    task_id=str(core["task_id"]),
                    task=core["task"],
                    true_answer=core["true_answer"],
                    agent_planning=core["agent_planning"],
                    agent_experience=core["agent_experience"],
                    task_analysis=core["task_analysis"],
                    plan_subtask_action=core["plan_subtask_action"],
                    total_score=item.get("score"),
                )
            )

        _set_cached(cache_key, response_data)
        update_performance_stats(time.time() - start_time)
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@app.post("/search/type_domain_text", response_model=List[TaskResponse])
async def type_domain_text_search(request: TypeDomainSearchRequest):
    start_time = time.time()
    cache_key = f"type_domain_{request.query}_{request.task_types}_{request.domains}_{request.top_k}_{request.weights}"

    try:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

        results = manager.type_domain_text_search(
            query=request.query,
            task_types=request.task_types,
            domains=request.domains,
            top_k=request.top_k,
            weights=request.weights,
        )

        response_data: List[TaskResponse] = []
        for item in results:
            core = _extract_task_fields(item)
            response_data.append(
                TaskResponse(
                    task_id=str(core["task_id"]),
                    task=core["task"],
                    true_answer=core["true_answer"],
                    agent_planning=core["agent_planning"],
                    agent_experience=core["agent_experience"],
                    task_analysis=core["task_analysis"],
                    plan_subtask_action=core["plan_subtask_action"],
                    total_score=item.get("score"),
                )
            )

        _set_cached(cache_key, response_data)
        update_performance_stats(time.time() - start_time)
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Type-domain text search failed: {str(e)}")


@app.get("/performance", response_model=PerformanceStats)
async def get_performance():
    # TTL 내 캐시 엔트리 비율을 “cache hit rate”로 단순 추정
    cache_hit_rate = (
        sum(
            1
            for v in response_cache.values()
            if time.time() - v["timestamp"] < CACHE_TTL
        )
        / len(response_cache)
        if response_cache
        else 0.0
    )

    return {
        "total_requests": performance_stats["total_requests"],
        "avg_response_time": performance_stats["avg_response_time"],
        "cache_hit_rate": cache_hit_rate,
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
        workers=int(os.getenv("UVICORN_WORKERS", 1)),
        limit_concurrency=MAX_CONCURRENT_SEARCHES,
    )
