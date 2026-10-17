"""Track and estimate OpenRouter cost for one program invocation."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any


PRICES_PER_MILLION_USD = {
    "anthropic/claude-haiku-5.5": {"input": 0.10, "output": 0.50},
    "anthropic/claude-sonnet-5.5": {"input": 2.00, "output": 10.00},
    "anthropic/claude-opus-5.5": {"input": 4.00, "output": 20.00},
    "openai/gpt-6-luna": {"input": 0.10, "output": 0.50},
    "openai/gpt-6-astra-pro": {"input": 10.00, "output": 50.00},
    "nvidia/nemotron-3.5-super-vl:free": {"input": 0.0, "output": 0.0},
    "thinkingmachines/inkling:free": {"input": 0.0, "output": 0.0},
    "google/gemma-4-26b-a4b-it:free": {"input": 0.0, "output": 0.0},
    "google/gemma-4-31b-it:free": {"input": 0.0, "output": 0.0},
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free": {
        "input": 0.0,
        "output": 0.0,
    },
    "nvidia/nemotron-3-super-120b-a12b:free": {"input": 0.0, "output": 0.0},
    "openrouter/free": {"input": 0.0, "output": 0.0},
}
PRICES_AS_OF = "2026-10-17"


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _prices() -> dict[str, dict[str, float]]:
    """Return built-in prices plus optional JSON overrides from the environment."""
    prices = {model: dict(values) for model, values in PRICES_PER_MILLION_USD.items()}
    raw = os.getenv("OPENROUTER_PRICE_OVERRIDES", "").strip()
    if not raw:
        return prices
    try:
        overrides = json.loads(raw)
        for model, values in overrides.items():
            prices[str(model)] = {
                "input": float(values["input"]),
                "output": float(values["output"]),
            }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "OPENROUTER_PRICE_OVERRIDES must be JSON like "
            "{\"provider/model\": {\"input\": 1.0, \"output\": 2.0}}"
        ) from exc
    return prices


def record_usage(registry: dict[str, dict], model: str, usage: Mapping[str, Any]) -> None:
    """Accumulate one successful API response in a per-model registry."""
    row = registry.setdefault(
        model,
        {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reported_cost_usd": 0.0,
            "reported_cost_available": False,
        },
    )
    row["calls"] += 1
    row["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
    row["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
    reported_cost = _number(usage.get("cost"))
    if reported_cost is not None:
        row["reported_cost_usd"] += reported_cost
        row["reported_cost_available"] = True


def _estimated_cost(model: str, row: Mapping[str, Any], prices: Mapping) -> float | None:
    if model.endswith(":free") or model == "openrouter/free":
        return 0.0
    price = prices.get(model)
    if price is None:
        return None
    return (
        int(row.get("prompt_tokens", 0)) * float(price["input"])
        + int(row.get("completion_tokens", 0)) * float(price["output"])
    ) / 1_000_000


def build_cost_report(
    scopes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    command: list[str] | None = None,
) -> dict:
    """Build a serializable cost report for OCR and translation usage registries."""
    prices = _prices()
    scope_reports = {}
    combined: dict[str, dict] = {}

    for scope_name, registry in scopes.items():
        calls = prompt_tokens = completion_tokens = 0
        reported_cost = estimated_cost = 0.0
        has_reported_cost = False
        unknown_models = []
        missing_reported_cost_models = []
        for model, row in registry.items():
            calls += int(row.get("calls", 0))
            prompt_tokens += int(row.get("prompt_tokens", 0))
            completion_tokens += int(row.get("completion_tokens", 0))
            if row.get("reported_cost_available"):
                reported_cost += float(row.get("reported_cost_usd", 0.0))
                has_reported_cost = True
            elif int(row.get("calls", 0)):
                missing_reported_cost_models.append(model)
            estimate = _estimated_cost(model, row, prices)
            if estimate is None:
                unknown_models.append(model)
            else:
                estimated_cost += estimate

            merged = combined.setdefault(
                model,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reported_cost_usd": 0.0,
                    "reported_cost_available": False,
                },
            )
            for key in ("calls", "prompt_tokens", "completion_tokens"):
                merged[key] += int(row.get(key, 0))
            if row.get("reported_cost_available"):
                merged["reported_cost_usd"] += float(row.get("reported_cost_usd", 0.0))
                merged["reported_cost_available"] = True

        scope_reports[scope_name] = {
            "calls": calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reported_cost_usd": reported_cost if has_reported_cost else None,
            "reported_cost_is_partial": bool(missing_reported_cost_models),
            "missing_reported_cost_models": sorted(missing_reported_cost_models),
            "estimated_cost_usd": estimated_cost,
            "estimate_is_partial": bool(unknown_models),
            "unknown_price_models": sorted(unknown_models),
        }

    model_reports = {}
    for model, row in sorted(combined.items()):
        estimate = _estimated_cost(model, row, prices)
        model_reports[model] = {
            **row,
            "estimated_cost_usd": estimate,
            "price_per_million_usd": prices.get(model),
        }

    total_calls = sum(row["calls"] for row in scope_reports.values())
    total_prompt = sum(row["prompt_tokens"] for row in scope_reports.values())
    total_completion = sum(row["completion_tokens"] for row in scope_reports.values())
    total_estimate = sum(row["estimated_cost_usd"] for row in scope_reports.values())
    reported_rows = [
        row["reported_cost_usd"]
        for row in scope_reports.values()
        if row["reported_cost_usd"] is not None
    ]
    unknown_models = sorted({
        model
        for row in scope_reports.values()
        for model in row["unknown_price_models"]
    })
    missing_reported_models = sorted({
        model
        for row in scope_reports.values()
        for model in row["missing_reported_cost_models"]
    })
    return {
        "command": command,
        "currency": "USD",
        "prices_as_of": PRICES_AS_OF,
        "scopes": scope_reports,
        "models": model_reports,
        "total": {
            "calls": total_calls,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "reported_cost_usd": sum(reported_rows) if reported_rows else None,
            "reported_cost_is_partial": bool(missing_reported_models),
            "missing_reported_cost_models": missing_reported_models,
            "estimated_cost_usd": total_estimate,
            "estimate_is_partial": bool(unknown_models),
            "unknown_price_models": unknown_models,
        },
    }


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.6f}"


def print_cost_report(report: Mapping[str, Any]) -> None:
    """Print one summary after every input in the command has completed."""
    print("\nAPI cost summary (entire command)")
    print("scope        calls   input tokens   output tokens   reported      estimated")
    for name, row in report["scopes"].items():
        estimate = _money(row["estimated_cost_usd"])
        if row["estimate_is_partial"]:
            estimate += "*"
        reported = _money(row["reported_cost_usd"])
        if row["reported_cost_is_partial"]:
            reported += "*"
        print(
            f"{name:<12} {row['calls']:>5} {row['prompt_tokens']:>14,} "
            f"{row['completion_tokens']:>15,} {reported:>12} "
            f"{estimate:>14}"
        )
    total = report["total"]
    total_estimate = _money(total["estimated_cost_usd"])
    if total["estimate_is_partial"]:
        total_estimate += "*"
    total_reported = _money(total["reported_cost_usd"])
    if total["reported_cost_is_partial"]:
        total_reported += "*"
    print(
        f"{'TOTAL':<12} {total['calls']:>5} {total['prompt_tokens']:>14,} "
        f"{total['completion_tokens']:>15,} {total_reported:>12} "
        f"{total_estimate:>14}"
    )
    for model, row in report["models"].items():
        print(
            f"  - {model}: {row['calls']} call(s), "
            f"estimated {_money(row['estimated_cost_usd'])}"
        )
    if total["unknown_price_models"]:
        print("* Partial estimate; missing price for: " + ", ".join(total["unknown_price_models"]))
    if total["missing_reported_cost_models"]:
        print(
            "* Partial reported cost; API omitted cost for: "
            + ", ".join(total["missing_reported_cost_models"])
        )
    print(f"Listed prices dated {report['prices_as_of']}; reported cost wins for billing.")

