from pydantic import BaseModel, Field
from typing import Literal

class DimensionMixin(BaseModel):
    """Optional WP6 partitioning dimensions (profile/platform/application)."""
    profile: str = ""
    platform: str = ""
    application: str = ""


class IngestFigmaRequest(DimensionMixin):
    project: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    figma_json: str | None = None
    # Optional pre-normalized canonical UI IR (e.g. produced by the gateway).
    ui_ir: dict | None = None
    # Optional {screen_name: purpose} hints (e.g. LLM-classified feature areas).
    purpose_hints: dict[str, str] | None = None


class RetrieveRequest(DimensionMixin):
    project: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    top_k: int = 5
    include_history: bool = True


class BriefContextRequest(BaseModel):
    project: str = Field(..., min_length=1)
    recent_limit: int = 12


class IngestSRSRequest(DimensionMixin):
    project: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    srs_text: str | None = None
    chunk_chars: int = 700
    srs_summary: str | None = None
    # Structured entity-graph payload from ingestion.extractor (requirements/entities/rules).
    extraction: dict | None = None


class LogTestRequest(DimensionMixin):
    project: str = Field(..., min_length=1)
    test_case_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    verdict: Literal["pass", "failed"]
    notes: str = ""
    area: str = "general"
    # Original requirement refs (e.g. ["FR-5","FR-7"]) this test covers -> COVERS edges.
    requirement_ids: list[str] = Field(default_factory=list)
    # Exploratory heuristic used (boundary/negative/state_transition/...) -> StrategyMemory.
    test_type: str = ""


class ResetProjectRequest(BaseModel):
    project: str = Field(..., min_length=1)
    delete_tests: bool = True
    delete_srs: bool = True
    delete_figma: bool = True


class GraphSubgraphRequest(BaseModel):
    project: str = Field(..., min_length=1)
    max_nodes: int = 300
    max_rels: int = 800


class ObserveStateRequest(BaseModel):
    """One observed UI state for the Live App Model (WP1)."""
    project: str = Field(..., min_length=1)
    # Normalized observation: {phone_state:{package,activity}, nodes:[...]}
    # (mobilerun.macro.state.normalize_ui_state output, or equivalent).
    normalized: dict
    screenshot_b64: str | None = None
    from_state_id: str | None = None
    action: str = ""
    element: str = ""


class ExecutionLogRequest(BaseModel):
    """One test execution's full record + the path it walked (WP3)."""
    project: str = Field(..., min_length=1)
    test_case_id: str = ""
    title: str = ""
    verdict: str = "failed"
    duration_ms: int = 0
    planned_steps: int = 0
    device_steps: int = 0
    states_visited: int = 0
    error_type: str = ""       # WP7: classified failure category (NAVIGATION_FAILURE, ...)
    error_message: str = ""
    recovery_action: str = ""  # WP7: self-healing recovery attempted + its outcome
    device: str = ""
    os_version: str = ""
    app_package: str = ""
    path: list[str] = Field(default_factory=list)         # ordered UIState ids visited
    path_labels: list[str] = Field(default_factory=list)  # ordered state labels


class IngestDefectsRequest(DimensionMixin):
    """Defect-history ingestion (ETA-REQ-301.1). Accepts a file path, inline text
    (JSON/CSV), or a pre-parsed list of defect dicts."""
    project: str = Field(..., min_length=1)
    source_path: str | None = None
    raw_text: str | None = None
    defects: list[dict] | None = None


class DimensionsRegisterRequest(DimensionMixin):
    """Register the dimensions a project targets, without ingesting content (WP6)."""
    project: str = Field(..., min_length=1)


class NavRecordPathRequest(BaseModel):
    """Record a navigation path through the app model (ETA-REQ-302.2)."""
    project: str = Field(..., min_length=1)
    test_case_id: str = ""
    title: str = ""
    verdict: str = "pass"
    path: list[str] = Field(default_factory=list)         # ordered UIState ids
    path_labels: list[str] = Field(default_factory=list)  # ordered state labels
    actions: list[str] = Field(default_factory=list)      # action taken to reach each step


class SessionStartRequest(BaseModel):
    """Start an exploratory testing session (ETA-REQ-303.5)."""
    project: str = Field(..., min_length=1)
    focus_area: str = ""
    strategy: str = ""


class SessionEndRequest(BaseModel):
    project: str = Field(..., min_length=1)
    session_id: str = ""


class SeedDemoTestsRequest(BaseModel):
    project: str = Field(..., min_length=1)
    area: str = "general"
    count: int = 6
    verdict_pattern: Literal["alternating", "all_pass", "all_failed"] = "alternating"


class DedupCheckRequest(BaseModel):
    """Semantic duplicate check for a candidate test (ETA-REQ-307.3)."""
    project: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    threshold: float = 0.9


class AnomaliesDetectRequest(BaseModel):
    """Trigger anomaly detection over execution logs (ETA-REQ-308.1)."""
    project: str = Field(..., min_length=1)


