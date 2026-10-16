"""Response models of API v1 (Part 5; spec 9.2, 9.3; contract 4.6).

Every JSON response is an `Envelope` (spec 9.2, plus explicit `status`, `notice`,
`provisional`, `run`, `sim_t_s` and `error`): a message type adds its fields next to the
envelope ones, as in the spec's `bus_status` example. Errors are envelopes too, with
`error: {code, message}` (404) or `error.code = "internal"` (503).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Status = Literal["ok", "degraded", "unavailable"]
STALE_NOTICE = "dernière valeur connue (donnée ancienne)"

# Error codes of 404 responses (contract 4.6); `not_found` is an unknown path
ERROR_CODES = (
    "unknown_run", "unknown_crossing", "unknown_bus", "unknown_trajet", "bad_speed",
    "bad_t", "bad_deadline", "bad_request", "not_found",
)  # fmt: skip


class ApiError(Exception):
    """A client error: unknown identifier or bad parameter (404 with an envelope)."""

    def __init__(self, code: str, message: str, status_code: int = 404) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class ErrorInfo(BaseModel):
    code: str
    message: str


class RunRef(BaseModel):
    run_id: str
    scenario: str
    seed: int
    controller: str | None = None
    code_version: str | None = None
    current: bool | None = Field(
        None, description="run.json config_hash and demand_hash equal the current config's"
    )


class Envelope(BaseModel):
    """Spec 9.2 common message format (ENF-03, ENF-04, ENF-05)."""

    data_source: Literal["simulation"] = "simulation"
    calibrated_on: date | None = Field(
        None, description="last sensor day used by the demand calibration (ENF-05)"
    )
    timestamp: datetime = Field(..., description="instant the data describes (Europe/Paris)")
    data_age_s: float | None = Field(
        None,
        description="replay data: query time minus the time of the data used; aggregates: "
        "wall-clock age of the file",
    )
    stale: bool = Field(False, description="data_age_s above the message type's limit")
    type: str
    status: Status = "ok"
    notice: str | None = None
    provisional: bool = Field(
        True, description="signal timing or queues not yet measured in the field"
    )
    run: RunRef | None = None
    sim_t_s: float | None = Field(None, description="replay time, s since the scenario start")
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _stale_notice(self) -> Envelope:
        if self.stale and self.notice is None:
            self.notice = STALE_NOTICE
        return self


# -- /health ---------------------------------------------------------------------------
class HealthResponse(Envelope):
    type: str = "health"
    runs_indexed: int = 0
    eta_model_version: str | None = None
    default_run: RunRef | None = None


# -- /scenarios ------------------------------------------------------------------------
class ScenarioRun(BaseModel):
    run_id: str
    seed: int
    current: bool | None = None
    has_eta: bool = False
    dir: str


class ScenarioGroup(BaseModel):
    scenario: str
    label: str | None = Field(None, description="evaluation label (S0, S1, S2, S2_noP)")
    family: str | None = None
    controller: str | None = None
    runs: list[ScenarioRun] = []


class ScenarioListResponse(Envelope):
    type: str = "scenario_list"
    scenarios: list[ScenarioGroup] = []


# -- /kpi ------------------------------------------------------------------------------
class KpiRow(BaseModel):
    user: str
    kpi: str
    mean: float | None = None
    std: float | None = None
    n: int
    rel_change_vs_s0_pct: float | None = None
    p_value: float | None = None
    worse_flag: bool | None = None


class KpiTableResponse(Envelope):
    type: str = "kpi_table"
    scenario: str | None = None
    label: str | None = None
    family: str | None = None
    source: Literal["evaluation", "runs"] | None = None
    rows: list[KpiRow] = []


# -- /replay ---------------------------------------------------------------------------
class Position(BaseModel):
    entity: str
    id: str
    x: float | None = None
    y: float | None = None
    speed: float | None = None


class ReplayStateResponse(Envelope):
    type: str = "replay_state"
    phase: int | None = None
    stage: int | None = None
    stage_name: str | None = None
    in_clearance: bool | None = None
    queue_total: float | None = None
    queues: dict[str, int] = {}
    counts: dict[str, int] = {}
    positions: list[Position] | None = None


# -- /crossing -------------------------------------------------------------------------
class CrossingStatusResponse(Envelope):
    type: str = "crossing_status"
    crossing: str
    signal: Literal["green", "red", "clearance"] | None = None
    green_remaining_s: float | None = None
    remaining_basis: Literal["program", "replay", "unknown"] = "unknown"
    next_green_in_s: float | None = None
    length_m: float | None = None
    speed_mps: float
    walk_time_s: float | None = None
    can_cross: bool | None = Field(None, description="walk time <= green remaining")
    obstructed: bool | None = Field(None, description="not logged in v1")


# -- /bus ------------------------------------------------------------------------------
class BusStatusResponse(Envelope):
    type: str = "bus_status"
    bus_id: str
    line: str
    flow: str | None = None
    vehicle_id: str | None = None
    delay_s: float | None = None
    delay_basis: Literal["junction_time_loss"] = "junction_time_loss"
    headway_s: float | None = None
    scheduled_headway_s: float | None = None
    priority_available: bool = False
    priority_active: bool | None = None
    message: str | None = None


# -- /ambulance ------------------------------------------------------------------------
class Intersection(BaseModel):
    id: str
    name: str
    modelled: bool
    status: Literal["clear", "green_wave", "blocked", "unknown"]
    distance_m: float | None = Field(
        None, description="straight-line distance of the ambulance to the junction centre"
    )
    passed: bool | None = Field(
        None, description="the ambulance is past the stop line of its approach (null: unknown)"
    )
    reason: str | None = None


class AmbulanceRouteResponse(Envelope):
    type: str = "ambulance_route"
    vehicle_id: str | None = None
    approach: str | None = None
    intersections: list[Intersection] = []
    eta_destination_s: float | None = Field(None, description="not modelled in v1")


# -- /eta ------------------------------------------------------------------------------
class EtaAdvice(BaseModel):
    code: str
    message: str
    arrival_median: datetime
    arrival_high: datetime
    margin_s: float


class EtaResponse(Envelope):
    type: str = "eta"
    trajet: str
    eta_median_s: float | None = None
    eta_low_s: float | None = None
    eta_high_s: float | None = None
    approximate: bool = False
    model_version: str | None = None
    queue_beyond_zone: bool | None = None
    rest_of_trip_s: float = 0.0
    advice: EtaAdvice | None = None
