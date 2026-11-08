"""Versioned HTTP interface and packaged workbench assets."""

from dataclasses import asdict, dataclass
from importlib.resources import files
from threading import Lock
from typing import Annotated, Literal

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from ._charts import render_chart
from ._scenarios import compare_scenarios, read_production_csv
from ._service import SolveOutcome, execute_model
from ._version import __version__


BoundPair = Annotated[list[FiniteFloat | None], Field(min_length=2, max_length=2)]
MAX_MODEL_BYTES = 65_536
_CHART_LOCK = Lock()


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    c: list[FiniteFloat] = Field(min_length=1, max_length=12)
    A: list[list[FiniteFloat]] = Field(max_length=30)
    b: list[FiniteFloat] = Field(max_length=30)
    constraint_senses: list[Literal["<=", ">=", "="]] | None = None
    sense: str = "min"
    bounds: list[BoundPair] | None = None
    max_iterations: int = Field(default=10_000, ge=0, le=10_000)


@dataclass(frozen=True, slots=True)
class ModelResponse(SolveOutcome):
    chart: str | None = None
    chart_message: str | None = None


class ProductionCsv(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    csv: str = Field(max_length=32_768)


class ScenarioRequest(ProductionCsv):
    baseline: dict[str, FiniteFloat] = Field(max_length=30)
    alternative: dict[str, FiniteFloat] = Field(max_length=30)


app = FastAPI(title="Simplex Workbench API", version=__version__)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    first = exc.errors()[0]
    field = ".".join(str(part) for part in first["loc"] if part != "body")
    return JSONResponse(
        {"status": "invalid_model", "error": f"{field or 'model'}: {first['msg']}"},
        status_code=422,
    )


class ModelSizeLimit:
    """Limit actual request bytes, including chunked requests."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or not scope["path"].startswith("/api/v1/"):
            await self.app(scope, receive, send)
            return
        chunks = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.extend(message.get("body", b""))
            if len(chunks) > MAX_MODEL_BYTES:
                response = JSONResponse(
                    {"status": "invalid_model", "error": "The model exceeds 65,536 bytes."},
                    status_code=413,
                )
                await response(scope, receive, send)
                return
            if not message.get("more_body"):
                break

        replay = {"type": "http.request", "body": bytes(chunks), "more_body": False}

        async def model_body():
            nonlocal replay
            if replay is not None:
                message, replay = replay, None
                return message
            return await receive()

        await self.app(scope, model_body, send)


app.add_middleware(ModelSizeLimit)


@app.post("/api/v1/solve", response_model=ModelResponse, response_model_exclude_none=True,
          responses={422: {"model": SolveOutcome}, 413: {"model": SolveOutcome}})
def solve_request(model: ModelRequest, response: Response, trace: bool = False, chart: bool = False):
    outcome = execute_model(model.model_dump(), trace=trace or chart)
    payload = asdict(outcome)
    if outcome.status == "invalid_model":
        response.status_code = 422
    if outcome.status == "optimal" and chart and len(model.c) == 2:
        relations = model.constraint_senses if model.constraint_senses is not None else ["<="] * len(model.b)
        bounds = model.bounds if model.bounds is not None else [(0, None)] * len(model.c)
        try:
            with _CHART_LOCK:
                payload["chart"] = render_chart(model.A, model.b, relations, bounds, outcome.steps)
        except ImportError:
            payload["chart_message"] = "Install the viz extra to show the 2D chart."
    if not trace:
        payload.pop("steps")
    return payload


@app.post("/api/v1/scenarios/import")
def import_production(data: ProductionCsv):
    try:
        products, resources, _, _ = read_production_csv(data.csv)
    except ValueError as exc:
        return JSONResponse({"status": "invalid_model", "error": str(exc)}, status_code=422)
    return {"products": products, "resources": resources}


@app.post("/api/v1/scenarios/compare")
def compare_production(data: ScenarioRequest):
    try:
        return compare_scenarios(data.csv, data.baseline, data.alternative)
    except ValueError as exc:
        return JSONResponse({"status": "invalid_model", "error": str(exc)}, status_code=422)


_ASSETS = {
    "/": ("index.html", "text/html"),
    "/app.css": ("app.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
    "/scenarios.js": ("scenarios.js", "text/javascript"),
    "/capacity_example.csv": ("capacity_example.csv", "text/csv"),
}


@app.get("/{asset:path}", include_in_schema=False)
def workbench_asset(asset: str) -> Response:
    path = "/" + asset
    if path not in _ASSETS:
        return Response("Not found", status_code=404, media_type="text/plain")
    filename, content_type = _ASSETS[path]
    content = files("simplex_codes").joinpath("_web", filename).read_bytes()
    return Response(content, media_type=content_type, headers={"X-Content-Type-Options": "nosniff"})
