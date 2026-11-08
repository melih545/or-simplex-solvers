"""Verify the running Docker stack with its public loopback endpoints."""

import json
from os import environ
from time import monotonic, sleep
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def read_json(url, model=None):
    data = json.dumps(model).encode() if model is not None else None
    request = Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def wait_for(check, description):
    deadline = monotonic() + 60
    while monotonic() < deadline:
        try:
            if check():
                return
        except (URLError, TimeoutError):
            pass
        sleep(1)
    raise RuntimeError(f"Timed out waiting for {description}.")


def main():
    app = f"http://127.0.0.1:{environ.get('SIMPLEX_PORT', '8765')}"
    prometheus = f"http://127.0.0.1:{environ.get('PROMETHEUS_PORT', '9090')}"
    grafana = f"http://127.0.0.1:{environ.get('GRAFANA_PORT', '3000')}"
    wait_for(lambda: read_json(app + "/healthz") == {"status": "ok"}, "workbench health")
    endpoint = app + "/api/v1/solve"
    assert read_json(endpoint, {"c": [1], "A": [[1]], "b": [2], "sense": "max"})["solution"] == [2]
    assert read_json(endpoint, {"c": [1], "A": [[1]], "b": [-1]})["status"] == "infeasible"

    def scraped(status):
        query = urlencode({"query": f'simplex_solve_attempts_total{{status="{status}"}}'})
        response = read_json(prometheus + "/api/v1/query?" + query)
        return any(float(sample["value"][1]) >= 1 for sample in response["data"]["result"])

    wait_for(lambda: scraped("optimal") and scraped("infeasible"), "Prometheus solve metrics")
    wait_for(
        lambda: read_json(grafana + "/api/dashboards/uid/simplex-overview")["dashboard"]["uid"] == "simplex-overview",
        "provisioned Grafana dashboard",
    )
    print("Workbench, Prometheus, and Grafana passed the stack smoke test.")


if __name__ == "__main__":
    main()
