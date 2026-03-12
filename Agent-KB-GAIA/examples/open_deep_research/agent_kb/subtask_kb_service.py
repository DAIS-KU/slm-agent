from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from subtask_kb_retrieval import SubAKB_Manager
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import time
import os

MAX_CONCURRENT_SEARCHES = int(os.getenv("MAX_CONCURRENT_SEARCHES", 10))
CACHE_TTL = int(os.getenv("CACHE_TTL", 60))

app = FastAPI(title="Optimized Subtask KB Retrieval API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = SubAKB_Manager(json_file_paths=["./subtask_akb.json"])

performance_stats = {
    "total_requests": 0,
    "avg_response_time": 0.0,
    "last_updated": time.time(),
}

response_cache: Dict[str, Dict[str, Any]] = {}


class SearchRequest(BaseModel):
    query: str
    top_k: int = 1
    weights: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {"text": 0.5, "semantic": 0.5}
    )


class SubTaskResponse(BaseModel):
    subtask_id: str
    task_id: str
    subtask: str
    inputs: Optional[Any] = None
    procedure: Optional[Any] = None
    do_raw: Optional[Any] = None
    do_sum: Optional[Any] = None
    expected_output: Optional[Any] = None
    actual_output: Optional[Any] = None
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


def _unwrap_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    manager가 반환하는 형태가
      1) {"content": {...}, "score": 0.12}
      2) {..., "total_score": 0.12}
    등 다양할 수 있어 content 래핑을 풀어준다.
    """
    if (
        isinstance(item, dict)
        and "content" in item
        and isinstance(item["content"], dict)
    ):
        return item["content"]
    return item


def _extract_subtask_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    core = _unwrap_item(item)
    return {
        "subtask_id": str(core.get("subtask_id", "")),
        "task_id": str(core.get("task_id", "")),
        "subtask": core.get("subtask", ""),
        "inputs": core.get("inputs", ""),
        "procedure": core.get("procedure", ""),
        "do_raw": core.get("do_raw", None),
        "do_sum": core.get("do_sum", None),
        "expected_output": core.get("expected_output", None),
        "actual_output": core.get("actual_output", None),
    }


# -----------------------------
# Endpoints
# -----------------------------
@app.post("/search/hybrid", response_model=List[SubTaskResponse])
async def hybrid_search(request: SearchRequest):
    start_time = time.time()
    cache_key = f"hybrid_{request.query}_{request.top_k}_{request.weights}"

    try:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

        results = manager.hybrid_search(
            query=request.query, top_k=request.top_k, weights=request.weights
        )

        response_data: List[SubTaskResponse] = []
        for item in results:
            core = _extract_subtask_fields(item)
            response_data.append(
                SubTaskResponse(
                    subtask_id=core["subtask_id"],
                    task_id=core["task_id"],
                    subtask=core["subtask"],
                    do_raw=core["do_raw"],
                    do_sum=core["do_sum"],
                    inputs=core["inputs"],
                    procedure=core["procedure"],
                    expected_output=core["expected_output"],
                    actual_output=core["actual_output"],
                    total_score=item.get(
                        "total_score"
                    ),  # hybrid은 total_score를 쓴다고 가정
                )
            )

        _set_cached(cache_key, response_data)
        update_performance_stats(time.time() - start_time)
        return response_data

    except Exception as e:
        print(f"Hybrid search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


@app.post("/search/text", response_model=List[SubTaskResponse])
async def text_search(request: SearchRequest):
    start_time = time.time()
    cache_key = f"text_{request.query}_{request.top_k}"

    try:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

        # "subtask" 필드를 대상으로 텍스트 검색(원하시면 "task" 등으로 변경)
        raw_results = manager.search_by_text(request.query, "subtask", request.top_k)

        response_data: List[SubTaskResponse] = []
        for item in raw_results:
            core = _extract_subtask_fields(item)
            response_data.append(
                SubTaskResponse(
                    subtask_id=core["subtask_id"],
                    task_id=core["task_id"],
                    subtask=core["subtask"],
                    do_raw=core["do_raw"],
                    do_sum=core["do_sum"],
                    inputs=core["inputs"],
                    procedure=core["procedure"],
                    expected_output=core["expected_output"],
                    actual_output=core["actual_output"],
                    total_score=item.get("score"),  # text 검색은 score -> total_score
                )
            )

        _set_cached(cache_key, response_data)
        update_performance_stats(time.time() - start_time)
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text search failed: {str(e)}")


@app.post("/search/semantic", response_model=List[SubTaskResponse])
async def semantic_search(request: SearchRequest):
    start_time = time.time()
    cache_key = f"semantic_{request.query}_{request.top_k}"

    try:
        cached = _get_cached(cache_key)
        if cached is not None:
            return cached

        # "subtask" 필드를 대상으로 semantic 검색(원하시면 변경)
        raw_results = manager.search_by_semantic(
            request.query, "subtask", request.top_k
        )

        response_data: List[SubTaskResponse] = []
        for item in raw_results:
            core = _extract_subtask_fields(item)
            response_data.append(
                SubTaskResponse(
                    subtask_id=core["subtask_id"],
                    task_id=core["task_id"],
                    subtask=core["subtask"],
                    do_raw=core["do_raw"],
                    do_sum=core["do_sum"],
                    inputs=core["inputs"],
                    procedure=core["procedure"],
                    expected_output=core["expected_output"],
                    actual_output=core["actual_output"],
                    total_score=item.get(
                        "score"
                    ),  # semantic 검색은 score -> total_score
                )
            )

        _set_cached(cache_key, response_data)
        update_performance_stats(time.time() - start_time)
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@app.get("/performance", response_model=PerformanceStats)
async def get_performance():
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
        port=8001,
        workers=int(os.getenv("UVICORN_WORKERS", 1)),
        limit_concurrency=MAX_CONCURRENT_SEARCHES,
    )
