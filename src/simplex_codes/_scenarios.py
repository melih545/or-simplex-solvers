"""Import production data and compare two capacity plans."""

from __future__ import annotations

import csv
from io import StringIO
from math import isfinite

from ._numerics import FEASIBILITY_TOLERANCE
from ._service import execute_model


def read_production_csv(source: str) -> tuple[list[str], list[str], list[float], list[list[float]]]:
    """Read product, profit, and per-unit usage of named resources."""
    reader = csv.reader(StringIO(source.removeprefix("\ufeff")), strict=True)
    try:
        header = [field.strip() for field in next(reader, [])]
        if header[:2] != ["product", "profit"] or not 3 <= len(header) <= 32:
            raise ValueError("CSV header must start with product,profit and name 1–30 resources.")
        resources = header[2:]
        if any(not name for name in resources) or len(set(resources)) != len(resources):
            raise ValueError("CSV resource names must be nonempty and unique.")

        products: list[str] = []
        profits: list[float] = []
        usage: list[list[float]] = []
        for row in reader:
            if not row:
                continue
            product = row[0].strip()
            if len(row) != len(header) or not product or product in products:
                raise ValueError(f"CSV line {reader.line_num}: expected a unique product and {len(header)} fields.")
            try:
                profit, *quantities = map(float, row[1:])
            except ValueError as exc:
                raise ValueError(f"CSV line {reader.line_num}: profit and usage must be numeric.") from exc
            if not isfinite(profit) or any(not isfinite(value) or value < 0 for value in quantities):
                raise ValueError(f"CSV line {reader.line_num}: use finite profit and finite nonnegative usage.")
            products.append(product)
            profits.append(profit)
            usage.append(quantities)
            if len(products) > 12:
                raise ValueError("CSV supports up to 12 products.")
    except csv.Error as exc:
        raise ValueError(f"CSV line {reader.line_num}: {exc}.") from exc
    if not products:
        raise ValueError("CSV must contain at least one product.")
    return products, resources, profits, usage


def compare_scenarios(source: str, baseline: dict[str, float], alternative: dict[str, float]) -> dict:
    products, resources, profits, usage = read_production_csv(source)
    A = [list(column) for column in zip(*usage)]

    for label, capacities in (("Baseline", baseline), ("Alternative", alternative)):
        if capacities.keys() != set(resources):
            raise ValueError(f"{label}: provide a capacity for every CSV resource.")
        if any(not isfinite(value) or value < 0 for value in capacities.values()):
            raise ValueError(f"{label}: capacities must be finite and nonnegative.")

    reports = []
    for capacities in (baseline, alternative):
        limits = [capacities[resource] for resource in resources]
        outcome = execute_model({"c": profits, "A": A, "b": limits, "sense": "max"})
        report = {"status": outcome.status}
        if outcome.solution is None:
            report["error"] = outcome.error
        else:
            report["profit"] = outcome.objective_value
            report["products"] = dict(zip(products, outcome.solution))
            report["resources"] = []
            for name, limit, coefficients in zip(resources, limits, A):
                used = sum(coefficient * amount for coefficient, amount in zip(coefficients, outcome.solution))
                slack = limit - used
                active = abs(slack) <= FEASIBILITY_TOLERANCE * max(abs(limit), abs(used))
                report["resources"].append({
                    "name": name, "limit": limit, "used": used, "slack": 0.0 if active else slack, "active": active,
                })
        reports.append(report)

    first, second = reports
    comparison = {"baseline": first, "alternative": second}
    if "profit" in first and "profit" in second:
        comparison["profit_delta"] = second["profit"] - first["profit"]
    return comparison
