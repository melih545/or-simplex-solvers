"""HTTP and CLI contract tests for the installed workbench."""

import asyncio
from base64 import b64decode
import json
import subprocess
import sys

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from simplex_codes._api import app


def request(method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


MODEL = {"c": [8, 5], "A": [[1, 1], [9, 5]], "b": [6, 45], "sense": "max"}


def test_workbench_reports_solution_trace_and_optional_chart():
    pytest.importorskip("matplotlib")
    response = request("POST", "/api/v1/solve?trace=true&chart=true", json=MODEL)
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "optimal"
    assert result["solution"] == pytest.approx([3.75, 2.25])
    assert result["objective_value"] == pytest.approx(41.25)
    assert [step["iteration"] for step in result["steps"]] == [0, 1, 2]
    assert b64decode(result["chart"]).startswith(b"\x89PNG\r\n\x1a\n")


def test_workbench_supports_free_and_fixed_variables_without_a_chart():
    response = request("POST", "/api/v1/solve?chart=true", json={
        "c": [1, 2, 3],
        "A": [[1, 0, 0], [0, 1, 0]],
        "b": [-2, 1],
        "constraint_senses": ["=", ">="],
        "bounds": [[None, None], [0, None], [2, 2]],
    })
    result = response.json()
    assert result["solution"] == pytest.approx([-2, 1, 2])
    assert result["objective_value"] == pytest.approx(6)
    assert "chart" not in result
    assert "steps" not in result


@pytest.mark.parametrize(
    ("model", "status", "http_code"),
    [
        (MODEL, "optimal", 200),
        ({"c": [1], "A": [[1]], "b": [-1]}, "infeasible", 200),
        ({"c": [-1], "A": [], "b": []}, "unbounded", 200),
        (MODEL | {"max_iterations": 0}, "iteration_limit", 200),
        ({"c": [1e308], "A": [[1]], "b": [2], "sense": "max"}, "numerical_error", 200),
        (MODEL | {"b": []}, "invalid_model", 422),
    ],
)
def test_cli_and_http_agree_on_model_outcome(model, status, http_code):
    response = request("POST", "/api/v1/solve", json=model)
    cli = subprocess.run(
        [sys.executable, "-m", "simplex_codes", "--structured"],
        input=json.dumps(model), capture_output=True, text=True, check=False,
    )
    assert response.status_code == http_code
    assert response.json()["status"] == status
    assert json.loads(cli.stdout) == response.json()
    assert cli.returncode == (0 if status == "optimal" else 1)


def test_workbench_exposes_schema_and_limits_requests():
    assert request("GET", "/").status_code == 200
    assert request("GET", "/app.js").status_code == 200
    assert "/api/v1/solve" in request("GET", "/openapi.json").json()["paths"]
    assert request("POST", "/api/v1/solve", json=MODEL | {"unexpected": True}).status_code == 422
    assert request("POST", "/api/v1/solve", content=b" " * 65_537).status_code == 413


def test_capacity_comparison_reports_binding_constraints_and_profit_change():
    source = request("GET", "/capacity_example.csv").text
    imported = request("POST", "/api/v1/scenarios/import", json={"csv": source})
    assert imported.json() == {"products": ["Masa", "Sandalye"], "resources": ["Depo", "İşçilik"]}
    baseline = {"Depo": 6, "İşçilik": 45}
    alternative = {"Depo": 7, "İşçilik": 54}
    response = request("POST", "/api/v1/scenarios/compare", json={
        "csv": source, "baseline": baseline, "alternative": alternative,
    })
    assert response.status_code == 200
    report = response.json()
    assert report["baseline"]["profit"] == pytest.approx(41.25)
    assert report["alternative"]["products"] == pytest.approx({"Masa": 4.75, "Sandalye": 2.25})
    assert report["profit_delta"] == pytest.approx(8)
    assert all(resource["active"] for resource in report["baseline"]["resources"])


def test_capacity_import_reports_invalid_row_number():
    response = request("POST", "/api/v1/scenarios/import", json={"csv": "product,profit,steel\na,2,bad\n"})
    assert response.status_code == 422
    assert "line 2" in response.json()["error"]


def test_capacity_comparison_reports_unused_resources_and_unbounded_profit():
    response = request("POST", "/api/v1/scenarios/compare", json={
        "csv": "product,profit,steel,labor\na,2,1,0\n",
        "baseline": {"steel": 4, "labor": 10},
        "alternative": {"steel": 8, "labor": 20},
    })
    report = response.json()
    assert report["baseline"]["resources"][1]["slack"] == 10
    assert not report["baseline"]["resources"][1]["active"]
    unbounded = request("POST", "/api/v1/scenarios/compare", json={
        "csv": "product,profit,steel\na,2,0\n",
        "baseline": {"steel": 4},
        "alternative": {"steel": 8},
    }).json()
    assert unbounded["baseline"]["status"] == "unbounded"
    assert "profit_delta" not in unbounded


def test_workbench_exposes_health_and_records_outcomes(caplog):
    from prometheus_client.parser import text_string_to_metric_families

    def attempt_counts():
        scrape = request("GET", "/metrics")
        assert scrape.status_code == 200
        return {
            sample.labels["status"]: sample.value
            for family in text_string_to_metric_families(scrape.text)
            for sample in family.samples
            if sample.name == "simplex_solve_attempts_total" and sample.labels["solver"] == "two_phase"
        }

    assert request("GET", "/healthz").json() == {"status": "ok"}
    before = attempt_counts()
    with caplog.at_level("INFO", logger="simplex.solve"):
        for model in (MODEL, {"c": [1], "A": [[1]], "b": [-1]},
                      {"c": [1e308], "A": [[1]], "b": [2], "sense": "max"}):
            request("POST", "/api/v1/solve", json=model)
    after = attempt_counts()
    statuses = ["optimal", "infeasible", "numerical_error"]
    assert [after[status] - before[status] for status in statuses] == [1, 1, 1]
    records = [json.loads(record.message) for record in caplog.records if record.name == "simplex.solve"]
    assert [record["status"] for record in records] == statuses
    assert all(record["duration_seconds"] >= 0 and "model" not in record for record in records)
    assert "simplex_solve_duration_seconds_bucket" in request("GET", "/metrics").text
