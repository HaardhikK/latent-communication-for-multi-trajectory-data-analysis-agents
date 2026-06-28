from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class ScoreResult:
    passed: bool
    score: float
    message: str
    details: dict[str, object]


@dataclass(frozen=True)
class ToyTask:
    task_id: str
    name: str
    prompt: str
    setup: Callable[[Path], None]
    score: Callable[[Path], ScoreResult]
    task_family: str = ""
    horizon_level: str = ""
    horizon_stages: int = 0
    stage_specs: tuple[str, ...] = ()
    reference_script: str = ""
    reference_output_tokens: int = 0
    reference_budget_ratio: float = 0.0


def _clean_summary_setup(work_dir: Path) -> None:
    rows = [
        {"city": "Berlin", "revenue": "100.0", "units": 10},
        {"city": "Paris", "revenue": "200.0", "units": 20},
        {"city": "Berlin", "revenue": "100.0", "units": 10},
        {"city": "Madrid", "revenue": "", "units": 5},
        {"city": "Paris", "revenue": "50.0", "units": 5},
    ]
    pd.DataFrame(rows).to_csv(work_dir / "input.csv", index=False)


def _clean_summary_score(work_dir: Path) -> ScoreResult:
    path = work_dir / "summary.json"
    if not path.exists():
        return ScoreResult(False, 0.0, "summary.json was not created", {})
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ScoreResult(False, 0.0, f"summary.json is not valid JSON: {exc}", {})
    if not isinstance(data, dict):
        return ScoreResult(False, 0.0, "summary.json must be a JSON object", {"type": type(data).__name__})

    expected = {
        "rows_after_cleaning": 3,
        "total_revenue": 350.0,
        "mean_units": 35.0 / 3.0,
        "top_city": "Paris",
    }
    checks = {
        "rows_after_cleaning": data.get("rows_after_cleaning") == expected["rows_after_cleaning"],
        "total_revenue": np.isclose(float(data.get("total_revenue", -1)), expected["total_revenue"], atol=1e-6),
        "mean_units": np.isclose(float(data.get("mean_units", -1)), expected["mean_units"], atol=1e-6),
        "top_city": str(data.get("top_city", "")).lower() == expected["top_city"].lower(),
    }
    score = sum(checks.values()) / len(checks)
    return ScoreResult(bool(score == 1.0), float(score), "clean summary checks completed", {"checks": checks, "data": data})


def _linear_regression_setup(work_dir: Path) -> None:
    pd.DataFrame({"x": [0, 1, 2, 3, 4], "y": [1, 3, 5, 7, 9]}).to_csv(work_dir / "train.csv", index=False)
    pd.DataFrame({"id": [10, 11, 12], "x": [5, 6, 7]}).to_csv(work_dir / "test.csv", index=False)


def _linear_regression_score(work_dir: Path) -> ScoreResult:
    path = work_dir / "predictions.csv"
    if not path.exists():
        return ScoreResult(False, 0.0, "predictions.csv was not created", {})
    try:
        data = pd.read_csv(path)
    except Exception as exc:
        return ScoreResult(False, 0.0, f"predictions.csv could not be read: {exc}", {})
    if "id" not in data.columns or "y_pred" not in data.columns:
        return ScoreResult(False, 0.0, "predictions.csv must contain id and y_pred columns", {"columns": list(data.columns)})
    data = data.sort_values("id").reset_index(drop=True)
    expected = pd.DataFrame({"id": [10, 11, 12], "y_pred": [11.0, 13.0, 15.0]})
    id_ok = data["id"].tolist() == expected["id"].tolist()
    pred_ok = np.allclose(data["y_pred"].astype(float).to_numpy(), expected["y_pred"].to_numpy(), atol=1e-4)
    score = (int(id_ok) + int(pred_ok)) / 2
    return ScoreResult(bool(id_ok and pred_ok), float(score), "linear regression checks completed", {"rows": data.to_dict("records")})


def _grouped_sales_setup(work_dir: Path) -> None:
    rows = [
        {"region": "north", "product": "a", "revenue": 10},
        {"region": "south", "product": "a", "revenue": 7},
        {"region": "north", "product": "b", "revenue": 5},
        {"region": "south", "product": "b", "revenue": 3},
        {"region": "west", "product": "a", "revenue": 8},
    ]
    pd.DataFrame(rows).to_csv(work_dir / "sales.csv", index=False)


def _grouped_sales_score(work_dir: Path) -> ScoreResult:
    path = work_dir / "region_summary.csv"
    if not path.exists():
        return ScoreResult(False, 0.0, "region_summary.csv was not created", {})
    try:
        data = pd.read_csv(path)
    except Exception as exc:
        return ScoreResult(False, 0.0, f"region_summary.csv could not be read: {exc}", {})
    required = {"region", "total_revenue", "row_count"}
    if not required.issubset(data.columns):
        return ScoreResult(False, 0.0, "region_summary.csv missing required columns", {"columns": list(data.columns)})
    got = data.sort_values("region").reset_index(drop=True)
    expected = pd.DataFrame(
        {
            "region": ["north", "south", "west"],
            "total_revenue": [15, 10, 8],
            "row_count": [2, 2, 1],
        }
    )
    region_ok = got["region"].astype(str).str.lower().tolist() == expected["region"].tolist()
    revenue_ok = np.allclose(got["total_revenue"].astype(float), expected["total_revenue"], atol=1e-6)
    count_ok = got["row_count"].astype(int).tolist() == expected["row_count"].tolist()
    score = (int(region_ok) + int(revenue_ok) + int(count_ok)) / 3
    return ScoreResult(bool(region_ok and revenue_ok and count_ok), float(score), "grouped sales checks completed", {"rows": got.to_dict("records")})


TASKS: dict[str, ToyTask] = {
    "clean_summary": ToyTask(
        task_id="clean_summary",
        name="CSV cleaning and summary statistics",
        setup=_clean_summary_setup,
        score=_clean_summary_score,
        prompt=(
            "Files in the current directory: input.csv with columns city, revenue, units.\n"
            "Write a Python script that reads input.csv, drops duplicate rows, drops rows with missing revenue, "
            "converts revenue and units to numbers, and writes summary.json.\n"
            "Use exact duplicate-row removal such as df.drop_duplicates(); do not drop duplicates by city only.\n"
            "summary.json must contain exactly these keys: rows_after_cleaning, total_revenue, mean_units, top_city.\n"
            "top_city is the city with the largest total revenue after cleaning."
        ),
    ),
    "linear_regression": ToyTask(
        task_id="linear_regression",
        name="Simple linear regression",
        setup=_linear_regression_setup,
        score=_linear_regression_score,
        prompt=(
            "Files in the current directory: train.csv with columns x,y and test.csv with columns id,x.\n"
            "Write a Python script that fits y = a*x + b from train.csv and predicts y for rows in test.csv.\n"
            "Use sklearn LinearRegression or numpy.polyfit/least-squares; do not use column means as the slope/intercept.\n"
            "Write predictions.csv with columns id and y_pred, preserving the ids from test.csv."
        ),
    ),
    "grouped_sales": ToyTask(
        task_id="grouped_sales",
        name="Grouped aggregation",
        setup=_grouped_sales_setup,
        score=_grouped_sales_score,
        prompt=(
            "File in the current directory: sales.csv with columns region, product, revenue. There is no index column.\n"
            "Write a Python script that groups rows by region and writes region_summary.csv.\n"
            "The output CSV must contain columns region, total_revenue, and row_count, sorted alphabetically by region. "
            "row_count is the number of input rows in each region; compute it with group size, not the DataFrame index. "
            "A safe pandas pattern is groupby('region').agg(total_revenue=('revenue','sum'), row_count=('revenue','size')).reset_index(). "
            "The output must not contain a column named revenue."
        ),
    ),
}


def selected_tasks(task_ids: list[str] | None = None) -> list[ToyTask]:
    if not task_ids:
        return list(TASKS.values())
    missing = [task_id for task_id in task_ids if task_id not in TASKS]
    if missing:
        raise KeyError(f"Unknown task id(s): {', '.join(missing)}")
    return [TASKS[task_id] for task_id in task_ids]
