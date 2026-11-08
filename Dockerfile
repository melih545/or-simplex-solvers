FROM prom/prometheus:v3.16.0 AS prometheus
COPY monitoring/prometheus/prometheus.yml /etc/prometheus/prometheus.yml

FROM grafana/grafana:13.2.2 AS grafana
COPY monitoring/grafana/datasource.yml /etc/grafana/provisioning/datasources/simplex.yml
COPY monitoring/grafana/provider.yml /etc/grafana/provisioning/dashboards/simplex.yml
COPY monitoring/grafana/dashboard.json /etc/grafana/dashboards/simplex.json

FROM python:3.12-slim AS wheel
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheel .

FROM python:3.12-slim AS workbench
ENV PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
RUN --mount=type=bind,from=wheel,source=/wheel,target=/wheel \
    set -- /wheel/simplex_codes-*.whl; pip install --no-cache-dir "$1[api,viz]"
USER 10001:10001
EXPOSE 8765
CMD ["python", "-m", "simplex_codes._app", "--host", "0.0.0.0"]
