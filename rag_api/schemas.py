from pydantic import BaseModel, Field
from typing import Literal

class IngestFigmaRequest(BaseModel):
    project: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    figma_json: str | None = None
    # Optional pre-normalized canonical UI IR (e.g. produced by the gateway).
    ui_ir: dict | None = None
    # Optional {screen_name: purpose} hints (e.g. LLM-classified feature areas).
    purpose_hints: dict[str, str] | None = None


class RetrieveRequest(BaseModel):
    project: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    top_k: int = 5
    include_history: bool = True


class BriefContextRequest(BaseModel):
    project: str = Field(..., min_length=1)
    recent_limit: int = 12


class IngestSRSRequest(BaseModel):
    project: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    srs_text: str | None = None
    chunk_chars: int = 700
    srs_summary: str | None = None
    # Structured entity-graph payload from ingestion.extractor (requirements/entities/rules).
    extraction: dict | None = None


class LogTestRequest(BaseModel):
    project: str = Field(..., min_length=1)
    test_case_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    verdict: Literal["pass", "failed"]
    notes: str = ""
    area: str = "general"
    # Original requirement refs (e.g. ["FR-5","FR-7"]) this test covers -> COVERS edges.
    requirement_ids: list[str] = Field(default_factory=list)


class ResetProjectRequest(BaseModel):
    project: str = Field(..., min_length=1)
    delete_tests: bool = True
    delete_srs: bool = True
    delete_figma: bool = True


class GraphSubgraphRequest(BaseModel):
    project: str = Field(..., min_length=1)
    max_nodes: int = 300
    max_rels: int = 800


class SeedDemoTestsRequest(BaseModel):
    project: str = Field(..., min_length=1)
    area: str = "general"
    count: int = 6
    verdict_pattern: Literal["alternating", "all_pass", "all_failed"] = "alternating"


