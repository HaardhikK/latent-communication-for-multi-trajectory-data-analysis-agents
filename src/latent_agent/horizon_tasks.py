from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .tasks import ScoreResult, ToyTask


HORIZON_ORDER = {"short": 3, "medium": 5, "long": 7, "xlong": 9, "xxlong": 11}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _close(a: object, b: float, atol: float = 1e-4) -> bool:
    try:
        return bool(np.isclose(float(a), float(b), atol=atol))
    except Exception:
        return False


def _allclose_values(a: object, b: object, atol: float = 1e-4) -> bool:
    try:
        left = np.asarray(list(a), dtype=float)
        right = np.asarray(list(b), dtype=float)
        return left.shape == right.shape and bool(np.allclose(left, right, atol=atol))
    except Exception:
        return False


def _same_id(a: object, b: object) -> bool:
    try:
        return int(float(a)) == int(float(b))
    except Exception:
        return str(a).strip().lower() == str(b).strip().lower()


def _same_channel_values(observed: list[object], expected: list[object]) -> bool:
    if len(observed) != len(expected):
        return False
    return all(_same_id(a, b) for a, b in zip(observed, expected)) or [
        str(item).strip().lower() for item in observed
    ] == [str(item).strip().lower() for item in expected]


def _field_from_object(value: object, field: str) -> object:
    if isinstance(value, list) and value:
        return _field_from_object(value[0], field)
    if isinstance(value, dict):
        return value.get(field)
    return value


def _load_json(path: Path) -> tuple[dict[str, object] | None, str]:
    if not path.exists():
        return None, f"{path.name} was not created"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"{path.name} is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return None, f"{path.name} must contain a JSON object"
    return data, "ok"


def _orders_setup(work_dir: Path) -> None:
    _write_csv(
        work_dir / "orders.csv",
        [
            {"order_id": 1, "customer_id": 1, "units": 2, "unit_price": 10, "price": 10, "unit_cost": 0, "discount_rate": 0.00, "discount_amount": 0, "order_count": 1, "status": "complete"},
            {"order_id": 2, "customer_id": 2, "units": 1, "unit_price": 20, "price": 20, "unit_cost": 0, "discount_rate": 0.00, "discount_amount": 0, "order_count": 1, "status": "complete"},
            {"order_id": 3, "customer_id": 3, "units": 4, "unit_price": 5, "price": 5, "unit_cost": 0, "discount_rate": 0.00, "discount_amount": 0, "order_count": 1, "status": "complete"},
            {"order_id": 4, "customer_id": 1, "units": 3, "unit_price": 8, "price": 8, "unit_cost": 0, "discount_rate": 0.00, "discount_amount": 0, "order_count": 1, "status": "complete"},
            {"order_id": 5, "customer_id": 4, "units": 1, "unit_price": 7, "price": 7, "unit_cost": 0, "discount_rate": 0.00, "discount_amount": 0, "order_count": 1, "status": "complete"},
            {"order_id": 6, "customer_id": 5, "units": 5, "unit_price": 6, "price": 6, "unit_cost": 0, "discount_rate": 0.00, "discount_amount": 0, "order_count": 1, "status": "complete"},
        ],
    )
    _write_csv(
        work_dir / "customers.csv",
        [
            {"customer_id": 1, "region": "north", "customer_region": "north", "tier": "gold"},
            {"customer_id": 2, "region": "south", "customer_region": "south", "tier": "silver"},
            {"customer_id": 3, "region": "north", "customer_region": "north", "tier": "bronze"},
            {"customer_id": 4, "region": "west", "customer_region": "west", "tier": "silver"},
            {"customer_id": 5, "region": "south", "customer_region": "south", "tier": "gold"},
        ],
    )


def _orders_cleaned(work_dir: Path) -> pd.DataFrame:
    orders = pd.read_csv(work_dir / "orders.csv")
    customers = pd.read_csv(work_dir / "customers.csv")
    for col in ["units", "unit_price", "unit_cost", "discount_rate"]:
        orders[col] = pd.to_numeric(orders[col], errors="coerce")
    orders = orders[(orders["status"] == "complete") & orders["units"].notna()].copy()
    orders["gross_revenue"] = orders["units"] * orders["unit_price"]
    orders["net_revenue"] = orders["gross_revenue"] * (1 - orders["discount_rate"])
    orders["margin"] = orders["net_revenue"] - orders["units"] * orders["unit_cost"]
    return orders.merge(customers, on="customer_id", how="left")


def _orders_short_score(work_dir: Path) -> ScoreResult:
    data, message = _load_json(work_dir / "orders_short_report.json")
    if data is None:
        return ScoreResult(False, 0.0, message, {})
    expected = _orders_cleaned(work_dir)
    top_region = expected.groupby("region")["net_revenue"].sum().idxmax()
    observed_rows = data.get("rows_clean")
    rows_ok = len(observed_rows) == len(expected) if isinstance(observed_rows, list) else observed_rows == len(expected)
    checks = {
        "rows_clean": rows_ok,
        "total_net_revenue": _close(data.get("total_net_revenue"), expected["net_revenue"].sum()),
        "top_region": str(data.get("top_region", "")).lower() == str(top_region).lower(),
    }
    return _score(checks, "orders short checks completed", data)


def _orders_medium_score(work_dir: Path) -> ScoreResult:
    path = work_dir / "orders_medium_region_summary.csv"
    if not path.exists():
        return ScoreResult(False, 0.0, "orders_medium_region_summary.csv was not created", {})
    try:
        got = pd.read_csv(path).sort_values("region").reset_index(drop=True)
    except Exception as exc:
        return ScoreResult(False, 0.0, f"orders_medium_region_summary.csv could not be read: {exc}", {})
    expected = (
        _orders_cleaned(work_dir)
        .groupby("region")
        .agg(net_revenue=("net_revenue", "sum"), margin=("margin", "sum"), order_count=("order_id", "size"))
        .reset_index()
        .sort_values("region")
        .reset_index(drop=True)
    )
    checks = {
        "columns": {"region", "net_revenue", "margin", "order_count"}.issubset(got.columns),
        "regions": got.get("region", pd.Series(dtype=str)).astype(str).str.lower().tolist() == expected["region"].str.lower().tolist(),
        "net_revenue": "net_revenue" in got and _allclose_values(got["net_revenue"], expected["net_revenue"]),
        "margin": "margin" in got,
        "order_count": "order_count" in got and got["order_count"].astype(int).tolist() == expected["order_count"].astype(int).tolist(),
    }
    return _score(checks, "orders medium checks completed", got.to_dict("records"))


def _orders_long_score(work_dir: Path) -> ScoreResult:
    from sklearn.linear_model import LinearRegression

    path = work_dir / "orders_long_report.csv"
    if not path.exists():
        return ScoreResult(False, 0.0, "orders_long_report.csv was not created", {})
    try:
        data = pd.read_csv(path).iloc[0].to_dict()
    except Exception as exc:
        return ScoreResult(False, 0.0, f"orders_long_report.csv could not be read: {exc}", {})
    clean = _orders_cleaned(work_dir)
    customer = clean.groupby("customer_id")["net_revenue"].sum()
    top_region = clean.groupby("region")["net_revenue"].sum().idxmax()
    model = LinearRegression().fit(clean[["units", "unit_price", "discount_rate"]], clean["net_revenue"])
    expected_mae = float(abs(model.predict(clean[["units", "unit_price", "discount_rate"]]) - clean["net_revenue"]).mean())
    high_value_observed = data.get("high_value_customers")
    if isinstance(high_value_observed, str) and high_value_observed.startswith("["):
        try:
            high_value_observed = json.loads(high_value_observed)
        except Exception:
            pass
    if isinstance(high_value_observed, list):
        high_value_ok = len(high_value_observed) == int((customer >= 30).sum())
    else:
        try:
            high_value_ok = int(high_value_observed) == int((customer >= 30).sum())
        except Exception:
            high_value_ok = False
    checks = {
        "rows_clean": "rows_clean" in data,
        "total_net_revenue": _close(data.get("total_net_revenue"), clean["net_revenue"].sum()),
        "total_margin": "total_margin" in data,
        "top_region": str(data.get("top_region", "")).lower() == str(top_region).lower(),
        "high_value_customers": high_value_ok,
        "model_mae": "model_mae" in data or _close(data.get("model_mae"), expected_mae, atol=1e-3),
    }
    return _score(checks, "orders long checks completed", data)


def _orders_xlong_score(work_dir: Path) -> ScoreResult:
    from sklearn.linear_model import LinearRegression

    data, message = _load_json(work_dir / "orders_xlong_report.json")
    if data is None:
        return ScoreResult(False, 0.0, message, {})
    clean = _orders_cleaned(work_dir)
    customer = clean.groupby("customer_id")["net_revenue"].sum()
    top_region = clean.groupby("region")["net_revenue"].sum().idxmax()
    model = LinearRegression().fit(clean[["units", "unit_price", "discount_rate"]], clean["net_revenue"])
    expected_mae = float(abs(model.predict(clean[["units", "unit_price", "discount_rate"]]) - clean["net_revenue"]).mean())
    best_tier = clean.groupby("tier")["margin"].sum().idxmax()
    checks = {
        "rows_clean": data.get("rows_clean") == len(clean),
        "total_net_revenue": _close(data.get("total_net_revenue"), clean["net_revenue"].sum()),
        "total_margin": _close(data.get("total_margin"), clean["margin"].sum()),
        "top_region": str(data.get("top_region", "")).lower() == str(top_region).lower(),
        "high_value_customers": int(data.get("high_value_customers", -1)) == int((customer >= 30).sum()),
        "model_mae": _close(data.get("model_mae"), expected_mae, atol=1e-3),
        "best_tier_by_margin": str(data.get("best_tier_by_margin", "")).lower() == str(best_tier).lower(),
        "margin_per_unit": _close(data.get("margin_per_unit"), clean["margin"].sum() / clean["units"].sum()),
    }
    return _score(checks, "orders xlong checks completed", data)


def _orders_xxlong_score(work_dir: Path) -> ScoreResult:
    from sklearn.linear_model import LinearRegression

    data, message = _load_json(work_dir / "orders_xxlong_report.json")
    if data is None:
        return ScoreResult(False, 0.0, message, {})
    clean = _orders_cleaned(work_dir)
    customer = clean.groupby("customer_id")["net_revenue"].sum()
    top_region = clean.groupby("region")["net_revenue"].sum().idxmax()
    best_tier = clean.groupby("tier")["margin"].sum().idxmax()
    top_customer = customer.idxmax()
    model = LinearRegression().fit(clean[["units", "unit_price", "discount_rate"]], clean["net_revenue"])
    train_pred = model.predict(clean[["units", "unit_price", "discount_rate"]])
    expected_mae = float(abs(train_pred - clean["net_revenue"]).mean())
    expected_prediction = float(model.predict(pd.DataFrame([{"units": 6, "unit_price": 9, "discount_rate": 0.0}]))[0])
    checks = {
        "rows_clean": data.get("rows_clean") == len(clean),
        "total_net_revenue": _close(data.get("total_net_revenue"), clean["net_revenue"].sum()),
        "total_margin": _close(data.get("total_margin"), clean["margin"].sum()),
        "top_region": str(data.get("top_region", "")).lower() == str(top_region).lower(),
        "high_value_customers": int(data.get("high_value_customers", -1)) == int((customer >= 30).sum()),
        "model_mae": _close(data.get("model_mae"), expected_mae, atol=1e-3),
        "best_tier_by_margin": str(data.get("best_tier_by_margin", "")).lower() == str(best_tier).lower(),
        "margin_per_unit": _close(data.get("margin_per_unit"), clean["margin"].sum() / clean["units"].sum()),
        "top_customer_id_by_net_revenue": _same_id(data.get("top_customer_id_by_net_revenue"), top_customer),
        "predicted_net_revenue_for_units_6_price_9_discount_0": _close(
            data.get("predicted_net_revenue_for_units_6_price_9_discount_0"),
            expected_prediction,
            atol=1e-2,
        ),
    }
    return _score(checks, "orders xxlong checks completed", data)


ORDERS_SHORT_REFERENCE = """import json
import pandas as pd
orders=pd.read_csv('orders.csv')
customers=pd.read_csv('customers.csv')
for c in ['units','unit_price','unit_cost','discount_rate']:
    orders[c]=pd.to_numeric(orders[c],errors='coerce')
orders=orders[(orders['status']=='complete') & orders['units'].notna()].copy()
orders['gross_revenue']=orders['units']*orders['unit_price']
orders['net_revenue']=orders['gross_revenue']*(1-orders['discount_rate'])
df=orders.merge(customers,on='customer_id',how='left')
top=df.groupby('region')['net_revenue'].sum().idxmax()
out={'rows_clean':int(len(df)),'total_net_revenue':float(df['net_revenue'].sum()),'top_region':str(top)}
with open('orders_short_report.json','w',encoding='utf-8') as f:
    json.dump(out,f)
"""


ORDERS_MEDIUM_REFERENCE = """import pandas as pd
orders=pd.read_csv('orders.csv')
customers=pd.read_csv('customers.csv')
for c in ['units','unit_price','unit_cost','discount_rate']:
    orders[c]=pd.to_numeric(orders[c],errors='coerce')
orders=orders[(orders['status']=='complete') & orders['units'].notna()].copy()
orders['gross_revenue']=orders['units']*orders['unit_price']
orders['net_revenue']=orders['gross_revenue']*(1-orders['discount_rate'])
orders['margin']=orders['net_revenue']-orders['units']*orders['unit_cost']
df=orders.merge(customers,on='customer_id',how='left')
summary=df.groupby('region').agg(net_revenue=('net_revenue','sum'),margin=('margin','sum'),order_count=('order_id','size')).reset_index()
summary=summary.sort_values('region')
summary.to_csv('orders_medium_region_summary.csv',index=False)
"""


ORDERS_LONG_REFERENCE = """import json
import pandas as pd
from sklearn.linear_model import LinearRegression
orders=pd.read_csv('orders.csv')
customers=pd.read_csv('customers.csv')
for c in ['units','unit_price','unit_cost','discount_rate']:
    orders[c]=pd.to_numeric(orders[c],errors='coerce')
orders=orders[(orders['status']=='complete') & orders['units'].notna()].copy()
orders['gross_revenue']=orders['units']*orders['unit_price']
orders['net_revenue']=orders['gross_revenue']*(1-orders['discount_rate'])
orders['margin']=orders['net_revenue']-orders['units']*orders['unit_cost']
df=orders.merge(customers,on='customer_id',how='left')
customer=df.groupby('customer_id')['net_revenue'].sum()
top_region=df.groupby('region')['net_revenue'].sum().idxmax()
X=df[['units','unit_price','discount_rate']]
y=df['net_revenue']
model=LinearRegression().fit(X,y)
pred=model.predict(X)
mae=float(abs(pred-y).mean())
out=pd.DataFrame([{'rows_clean':int(len(df)),'total_net_revenue':float(df['net_revenue'].sum()),'total_margin':float(df['margin'].sum()),'top_region':str(top_region),'high_value_customers':int((customer>=30).sum()),'model_mae':mae}])
out.to_csv('orders_long_report.csv',index=False)
"""


ORDERS_XLONG_REFERENCE = """import json
import pandas as pd
from sklearn.linear_model import LinearRegression
orders=pd.read_csv('orders.csv')
customers=pd.read_csv('customers.csv')
for c in ['units','unit_price','unit_cost','discount_rate']:
    orders[c]=pd.to_numeric(orders[c],errors='coerce')
orders=orders[(orders['status']=='complete') & orders['units'].notna()].copy()
orders['gross_revenue']=orders['units']*orders['unit_price']
orders['net_revenue']=orders['gross_revenue']*(1-orders['discount_rate'])
orders['margin']=orders['net_revenue']-orders['units']*orders['unit_cost']
df=orders.merge(customers,on='customer_id',how='left')
customer=df.groupby('customer_id')['net_revenue'].sum()
top_region=df.groupby('region')['net_revenue'].sum().idxmax()
model=LinearRegression().fit(df[['units','unit_price','discount_rate']],df['net_revenue'])
mae=float(abs(model.predict(df[['units','unit_price','discount_rate']])-df['net_revenue']).mean())
best_tier=df.groupby('tier')['margin'].sum().idxmax()
out={'rows_clean':int(len(df)),'total_net_revenue':float(df['net_revenue'].sum()),'total_margin':float(df['margin'].sum()),'top_region':str(top_region),'high_value_customers':int((customer>=30).sum()),'model_mae':mae,'best_tier_by_margin':str(best_tier),'margin_per_unit':float(df['margin'].sum()/df['units'].sum())}
with open('orders_xlong_report.json','w',encoding='utf-8') as f:
    json.dump(out,f)
"""


ORDERS_XXLONG_REFERENCE = """import json
import pandas as pd
from sklearn.linear_model import LinearRegression
orders=pd.read_csv('orders.csv')
customers=pd.read_csv('customers.csv')
for c in ['units','unit_price','unit_cost','discount_rate']:
    orders[c]=pd.to_numeric(orders[c],errors='coerce')
orders=orders[(orders['status']=='complete') & orders['units'].notna()].copy()
orders['gross_revenue']=orders['units']*orders['unit_price']
orders['net_revenue']=orders['gross_revenue']*(1-orders['discount_rate'])
orders['margin']=orders['net_revenue']-orders['units']*orders['unit_cost']
df=orders.merge(customers,on='customer_id',how='left')
customer=df.groupby('customer_id')['net_revenue'].sum()
top_region=df.groupby('region')['net_revenue'].sum().idxmax()
model=LinearRegression().fit(df[['units','unit_price','discount_rate']],df['net_revenue'])
mae=float(abs(model.predict(df[['units','unit_price','discount_rate']])-df['net_revenue']).mean())
best_tier=df.groupby('tier')['margin'].sum().idxmax()
future=pd.DataFrame([{'units':6,'unit_price':9,'discount_rate':0.0}])
out={'rows_clean':int(len(df)),'total_net_revenue':float(df['net_revenue'].sum()),'total_margin':float(df['margin'].sum()),'top_region':str(top_region),'high_value_customers':int((customer>=30).sum()),'model_mae':mae,'best_tier_by_margin':str(best_tier),'margin_per_unit':float(df['margin'].sum()/df['units'].sum()),'top_customer_id_by_net_revenue':int(customer.idxmax()),'predicted_net_revenue_for_units_6_price_9_discount_0':float(model.predict(future)[0])}
with open('orders_xxlong_report.json','w',encoding='utf-8') as f:
    json.dump(out,f)
"""


def _sensor_setup(work_dir: Path) -> None:
    _write_csv(
        work_dir / "readings.csv",
        [
            {"timestamp": "2026-01-01 08:00", "sensor_id": "S1", "raw_temp": 70, "raw_pressure": 100, "status": "ok"},
            {"timestamp": "2026-01-01 08:30", "sensor_id": "S2", "raw_temp": 78, "raw_pressure": 102, "status": "ok"},
            {"timestamp": "2026-01-01 09:00", "sensor_id": "S1", "raw_temp": 71, "raw_pressure": 99, "status": "ok"},
            {"timestamp": "2026-01-01 09:30", "sensor_id": "S3", "raw_temp": 82, "raw_pressure": 108, "status": "ok"},
            {"timestamp": "2026-01-01 10:00", "sensor_id": "S2", "raw_temp": 73, "raw_pressure": 104, "status": "ok"},
            {"timestamp": "2026-01-01 10:30", "sensor_id": "S3", "raw_temp": 76, "raw_pressure": 106, "status": "ok"},
        ],
    )
    _write_csv(
        work_dir / "sensor_meta.csv",
        [
            {"sensor_id": "S1", "site": "alpha", "site_id": "alpha", "temp_offset": 1.5, "calibration_temp": 1.5, "pressure_scale": 1.00, "calibration_pressure": 0},
            {"sensor_id": "S2", "site": "beta", "site_id": "beta", "temp_offset": -2.0, "calibration_temp": -2.0, "pressure_scale": 1.02, "calibration_pressure": 0},
            {"sensor_id": "S3", "site": "beta", "site_id": "beta", "temp_offset": 0.5, "calibration_temp": 0.5, "pressure_scale": 0.98, "calibration_pressure": 0},
        ],
    )


def _sensor_cleaned(work_dir: Path) -> pd.DataFrame:
    readings = pd.read_csv(work_dir / "readings.csv")
    meta = pd.read_csv(work_dir / "sensor_meta.csv")
    readings["raw_temp"] = pd.to_numeric(readings["raw_temp"], errors="coerce")
    readings["raw_pressure"] = pd.to_numeric(readings["raw_pressure"], errors="coerce")
    readings = readings[(readings["status"] == "ok") & readings["raw_temp"].notna()].copy()
    df = readings.merge(meta, on="sensor_id", how="left")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["adjusted_temp"] = df["raw_temp"] + df["temp_offset"]
    df["adjusted_pressure"] = df["raw_pressure"] * df["pressure_scale"]
    df["alert"] = (df["adjusted_temp"] > 76) | (df["adjusted_pressure"] > 105)
    return df


def _sensor_short_score(work_dir: Path) -> ScoreResult:
    data, message = _load_json(work_dir / "sensor_short_report.json")
    if data is None:
        path = work_dir / "sensor_short_report.json"
        if not path.exists():
            return ScoreResult(False, 0.0, message, {})
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return ScoreResult(False, 0.0, f"sensor_short_report.json is not valid JSON: {exc}", {})
        if isinstance(rows, list) and rows:
            expected = _sensor_cleaned(work_dir)
            checks = {
                "rows_present": len(rows) == len(expected),
                "adjusted_temp_present": "adjusted_temp" in rows[0],
                "alert_present": "alert" in rows[0],
            }
            return _score(checks, "sensor short row-list checks completed", rows[:2])
        return ScoreResult(False, 0.0, message, {})
    df = _sensor_cleaned(work_dir)
    checks = {
        "rows_clean": data.get("rows_clean") == len(df),
        "mean_adjusted_temp": _close(data.get("mean_adjusted_temp"), df["adjusted_temp"].mean()),
        "alert_count": int(data.get("alert_count", -1)) == int(df["alert"].sum()),
    }
    return _score(checks, "sensor short checks completed", data)


def _sensor_medium_score(work_dir: Path) -> ScoreResult:
    path = work_dir / "sensor_medium_site_summary.csv"
    if not path.exists():
        return ScoreResult(False, 0.0, "sensor_medium_site_summary.csv was not created", {})
    got = pd.read_csv(path).sort_values("site").reset_index(drop=True)
    expected = (
        _sensor_cleaned(work_dir)
        .groupby("site")
        .agg(mean_adjusted_temp=("adjusted_temp", "mean"), alert_count=("alert", "sum"), reading_count=("sensor_id", "size"))
        .reset_index()
        .sort_values("site")
        .reset_index(drop=True)
    )
    raw_expected = (
        _sensor_cleaned(work_dir)
        .groupby("site")
        .agg(mean_adjusted_temp=("raw_temp", "mean"))
        .reset_index()
        .sort_values("site")
        .reset_index(drop=True)
    )
    checks = {
        "columns": {"site", "mean_adjusted_temp", "alert_count", "reading_count"}.issubset(got.columns),
        "sites": got.get("site", pd.Series(dtype=str)).astype(str).str.lower().tolist() == expected["site"].str.lower().tolist(),
        "mean_adjusted_temp": "mean_adjusted_temp" in got
        and (
            _allclose_values(got["mean_adjusted_temp"], expected["mean_adjusted_temp"])
            or _allclose_values(got["mean_adjusted_temp"], raw_expected["mean_adjusted_temp"])
        ),
        "alert_count": "alert_count" in got and got["alert_count"].astype(int).tolist() == expected["alert_count"].astype(int).tolist(),
        "reading_count": "reading_count" in got and got["reading_count"].astype(int).tolist() == expected["reading_count"].astype(int).tolist(),
    }
    return _score(checks, "sensor medium checks completed", got.to_dict("records"))


def _sensor_long_score(work_dir: Path) -> ScoreResult:
    report, message = _load_json(work_dir / "sensor_long_report.json")
    if report is None:
        return ScoreResult(False, 0.0, message, {})
    hourly_path = work_dir / "sensor_long_hourly_summary.csv"
    if not hourly_path.exists():
        return ScoreResult(False, 0.0, "sensor_long_hourly_summary.csv was not created", {})
    hourly = pd.read_csv(hourly_path)
    df = _sensor_cleaned(work_dir)
    expected_hourly = df.groupby(["site", "hour"]).agg(alert_count=("alert", "sum"), mean_adjusted_temp=("adjusted_temp", "mean")).reset_index()
    worst_site = df.groupby("site")["alert"].sum().idxmax()
    peak_hour = int(df.groupby("hour")["alert"].sum().idxmax())
    checks = {
        "rows_clean": report.get("rows_clean") == len(df),
        "total_alerts": _close(report.get("total_alerts", -1), int(df["alert"].sum()), atol=0.0),
        "worst_site": str(report.get("worst_site", "")).lower() == str(worst_site).lower(),
        "peak_hour": "peak_hour" in report,
        "hourly_rows": len(hourly) == len(expected_hourly),
    }
    return _score(checks, "sensor long checks completed", {"report": report, "hourly": hourly.to_dict("records")})


def _sensor_xlong_score(work_dir: Path) -> ScoreResult:
    report, message = _load_json(work_dir / "sensor_xlong_report.json")
    if report is None:
        return ScoreResult(False, 0.0, message, {})
    site_path = work_dir / "sensor_xlong_site_summary.csv"
    if not site_path.exists():
        return ScoreResult(False, 0.0, "sensor_xlong_site_summary.csv was not created", {})
    site_summary = pd.read_csv(site_path)
    df = _sensor_cleaned(work_dir)
    worst_site = df.groupby("site")["alert"].sum().idxmax()
    peak_hour = int(df.groupby("hour")["alert"].sum().idxmax())
    expected_site = (
        df.groupby("site")
        .agg(temp_std=("adjusted_temp", "std"), alert_rate=("alert", "mean"), reading_count=("sensor_id", "size"))
        .reset_index()
        .sort_values("site")
        .reset_index(drop=True)
    )
    got_site = site_summary.sort_values("site").reset_index(drop=True)
    best_site = expected_site.sort_values(["temp_std", "site"]).iloc[0]["site"]
    checks = {
        "rows_clean": report.get("rows_clean") == len(df),
        "total_alerts": int(report.get("total_alerts", -1)) == int(df["alert"].sum()),
        "worst_site": str(report.get("worst_site", "")).lower() == str(worst_site).lower(),
        "peak_hour": int(report.get("peak_hour", -1)) == peak_hour,
        "best_site_by_temp_stability": str(report.get("best_site_by_temp_stability", "")).lower() == str(best_site).lower(),
        "mean_alert_rate": _close(report.get("mean_alert_rate"), df["alert"].mean()),
        "site_summary_columns": {"site", "temp_std", "alert_rate", "reading_count"}.issubset(got_site.columns),
        "site_summary_rows": len(got_site) == len(expected_site),
        "site_summary_alert_rate": "alert_rate" in got_site and _allclose_values(got_site["alert_rate"], expected_site["alert_rate"]),
    }
    return _score(checks, "sensor xlong checks completed", {"report": report, "site_summary": site_summary.to_dict("records")})


def _sensor_xxlong_score(work_dir: Path) -> ScoreResult:
    report, message = _load_json(work_dir / "sensor_xxlong_report.json")
    if report is None:
        return ScoreResult(False, 0.0, message, {})
    site_path = work_dir / "sensor_xxlong_site_summary.csv"
    if not site_path.exists():
        return ScoreResult(False, 0.0, "sensor_xxlong_site_summary.csv was not created", {})
    site_summary = pd.read_csv(site_path)
    df = _sensor_cleaned(work_dir)
    worst_site = df.groupby("site")["alert"].sum().idxmax()
    peak_hour = int(df.groupby("hour")["alert"].sum().idxmax())
    expected_site = (
        df.groupby("site")
        .agg(temp_std=("adjusted_temp", "std"), alert_rate=("alert", "mean"), reading_count=("sensor_id", "size"))
        .reset_index()
        .sort_values("site")
        .reset_index(drop=True)
    )
    got_site = site_summary.sort_values("site").reset_index(drop=True)
    best_site = expected_site.sort_values(["temp_std", "site"]).iloc[0]["site"]
    worst_sensor = df.groupby("sensor_id")["alert"].sum().idxmax()
    site_hour = df.groupby(["site", "hour"])["alert"].sum().reset_index().sort_values(["alert", "site", "hour"], ascending=[False, True, True]).iloc[0]
    peak_site_hour = f"{site_hour['site']}-{int(site_hour['hour'])}"
    checks = {
        "rows_clean": report.get("rows_clean") == len(df),
        "total_alerts": int(report.get("total_alerts", -1)) == int(df["alert"].sum()),
        "worst_site": str(report.get("worst_site", "")).lower() == str(worst_site).lower(),
        "peak_hour": int(report.get("peak_hour", -1)) == peak_hour,
        "best_site_by_temp_stability": str(report.get("best_site_by_temp_stability", "")).lower() == str(best_site).lower(),
        "mean_alert_rate": _close(report.get("mean_alert_rate"), df["alert"].mean()),
        "site_summary_columns": {"site", "temp_std", "alert_rate", "reading_count"}.issubset(got_site.columns),
        "site_summary_rows": len(got_site) == len(expected_site),
        "worst_sensor_by_alerts": str(report.get("worst_sensor_by_alerts", "")).lower() == str(worst_sensor).lower(),
        "peak_site_hour": str(report.get("peak_site_hour", "")).lower() == peak_site_hour.lower(),
    }
    return _score(checks, "sensor xxlong checks completed", {"report": report, "site_summary": site_summary.to_dict("records")})


SENSOR_SHORT_REFERENCE = """import json
import pandas as pd
r=pd.read_csv('readings.csv')
m=pd.read_csv('sensor_meta.csv')
r['raw_temp']=pd.to_numeric(r['raw_temp'],errors='coerce')
r['raw_pressure']=pd.to_numeric(r['raw_pressure'],errors='coerce')
r=r[(r['status']=='ok') & r['raw_temp'].notna()].copy()
df=r.merge(m,on='sensor_id',how='left')
df['adjusted_temp']=df['raw_temp']+df['temp_offset']
df['adjusted_pressure']=df['raw_pressure']*df['pressure_scale']
df['alert']=(df['adjusted_temp']>76) | (df['adjusted_pressure']>105)
out={'rows_clean':int(len(df)),'mean_adjusted_temp':float(df['adjusted_temp'].mean()),'alert_count':int(df['alert'].sum())}
with open('sensor_short_report.json','w',encoding='utf-8') as f:
    json.dump(out,f)
"""


SENSOR_MEDIUM_REFERENCE = """import pandas as pd
r=pd.read_csv('readings.csv')
m=pd.read_csv('sensor_meta.csv')
r['raw_temp']=pd.to_numeric(r['raw_temp'],errors='coerce')
r['raw_pressure']=pd.to_numeric(r['raw_pressure'],errors='coerce')
r=r[(r['status']=='ok') & r['raw_temp'].notna()].copy()
df=r.merge(m,on='sensor_id',how='left')
df['adjusted_temp']=df['raw_temp']+df['temp_offset']
df['adjusted_pressure']=df['raw_pressure']*df['pressure_scale']
df['alert']=(df['adjusted_temp']>76) | (df['adjusted_pressure']>105)
summary=df.groupby('site').agg(mean_adjusted_temp=('adjusted_temp','mean'),alert_count=('alert','sum'),reading_count=('sensor_id','size')).reset_index().sort_values('site')
summary.to_csv('sensor_medium_site_summary.csv',index=False)
"""


SENSOR_LONG_REFERENCE = """import json
import pandas as pd
r=pd.read_csv('readings.csv')
m=pd.read_csv('sensor_meta.csv')
r['raw_temp']=pd.to_numeric(r['raw_temp'],errors='coerce')
r['raw_pressure']=pd.to_numeric(r['raw_pressure'],errors='coerce')
r=r[(r['status']=='ok') & r['raw_temp'].notna()].copy()
df=r.merge(m,on='sensor_id',how='left')
df['timestamp']=pd.to_datetime(df['timestamp'])
df['hour']=df['timestamp'].dt.hour
df['adjusted_temp']=df['raw_temp']+df['temp_offset']
df['adjusted_pressure']=df['raw_pressure']*df['pressure_scale']
df['alert']=(df['adjusted_temp']>76) | (df['adjusted_pressure']>105)
hourly=df.groupby(['site','hour']).agg(alert_count=('alert','sum'),mean_adjusted_temp=('adjusted_temp','mean')).reset_index()
hourly.to_csv('sensor_long_hourly_summary.csv',index=False)
out={'rows_clean':int(len(df)),'total_alerts':int(df['alert'].sum()),'worst_site':str(df.groupby('site')['alert'].sum().idxmax()),'peak_hour':int(df.groupby('hour')['alert'].sum().idxmax())}
with open('sensor_long_report.json','w',encoding='utf-8') as f:
    json.dump(out,f)
"""


SENSOR_XLONG_REFERENCE = """import json
import pandas as pd
r=pd.read_csv('readings.csv')
m=pd.read_csv('sensor_meta.csv')
r['raw_temp']=pd.to_numeric(r['raw_temp'],errors='coerce')
r['raw_pressure']=pd.to_numeric(r['raw_pressure'],errors='coerce')
r=r[(r['status']=='ok') & r['raw_temp'].notna()].copy()
df=r.merge(m,on='sensor_id',how='left')
df['timestamp']=pd.to_datetime(df['timestamp'])
df['hour']=df['timestamp'].dt.hour
df['adjusted_temp']=df['raw_temp']+df['temp_offset']
df['adjusted_pressure']=df['raw_pressure']*df['pressure_scale']
df['alert']=(df['adjusted_temp']>76) | (df['adjusted_pressure']>105)
hourly=df.groupby(['site','hour']).agg(alert_count=('alert','sum'),mean_adjusted_temp=('adjusted_temp','mean')).reset_index()
site_summary=df.groupby('site').agg(temp_std=('adjusted_temp','std'),alert_rate=('alert','mean'),reading_count=('sensor_id','size')).reset_index().sort_values('site')
site_summary.to_csv('sensor_xlong_site_summary.csv',index=False)
best_site=site_summary.sort_values(['temp_std','site']).iloc[0]['site']
out={'rows_clean':int(len(df)),'total_alerts':int(df['alert'].sum()),'worst_site':str(df.groupby('site')['alert'].sum().idxmax()),'peak_hour':int(df.groupby('hour')['alert'].sum().idxmax()),'best_site_by_temp_stability':str(best_site),'mean_alert_rate':float(df['alert'].mean())}
with open('sensor_xlong_report.json','w',encoding='utf-8') as f:
    json.dump(out,f)
"""


SENSOR_XXLONG_REFERENCE = """import json
import pandas as pd
r=pd.read_csv('readings.csv')
m=pd.read_csv('sensor_meta.csv')
r['raw_temp']=pd.to_numeric(r['raw_temp'],errors='coerce')
r['raw_pressure']=pd.to_numeric(r['raw_pressure'],errors='coerce')
r=r[(r['status']=='ok') & r['raw_temp'].notna()].copy()
df=r.merge(m,on='sensor_id',how='left')
df['timestamp']=pd.to_datetime(df['timestamp'])
df['hour']=df['timestamp'].dt.hour
df['adjusted_temp']=df['raw_temp']+df['temp_offset']
df['adjusted_pressure']=df['raw_pressure']*df['pressure_scale']
df['alert']=(df['adjusted_temp']>76) | (df['adjusted_pressure']>105)
hourly=df.groupby(['site','hour']).agg(alert_count=('alert','sum'),mean_adjusted_temp=('adjusted_temp','mean')).reset_index()
site_summary=df.groupby('site').agg(temp_std=('adjusted_temp','std'),alert_rate=('alert','mean'),reading_count=('sensor_id','size')).reset_index().sort_values('site')
site_summary.to_csv('sensor_xxlong_site_summary.csv',index=False)
best_site=site_summary.sort_values(['temp_std','site']).iloc[0]['site']
site_hour=df.groupby(['site','hour'])['alert'].sum().reset_index().sort_values(['alert','site','hour'],ascending=[False,True,True]).iloc[0]
out={'rows_clean':int(len(df)),'total_alerts':int(df['alert'].sum()),'worst_site':str(df.groupby('site')['alert'].sum().idxmax()),'peak_hour':int(df.groupby('hour')['alert'].sum().idxmax()),'best_site_by_temp_stability':str(best_site),'mean_alert_rate':float(df['alert'].mean()),'worst_sensor_by_alerts':str(df.groupby('sensor_id')['alert'].sum().idxmax()),'peak_site_hour':f"{site_hour['site']}-{int(site_hour['hour'])}"}
with open('sensor_xxlong_report.json','w',encoding='utf-8') as f:
    json.dump(out,f)
"""


def _campaign_setup(work_dir: Path) -> None:
    _write_csv(
        work_dir / "campaigns.csv",
        [
            {"campaign_id": 1, "channel_id": 1, "impressions": 1000, "clicks": 100, "conversions": 10, "spend": 200, "revenue": 500},
            {"campaign_id": 2, "channel_id": 2, "impressions": 1500, "clicks": 90, "conversions": 9, "spend": 180, "revenue": 300},
            {"campaign_id": 3, "channel_id": 1, "impressions": 800, "clicks": 120, "conversions": 12, "spend": 160, "revenue": 480},
            {"campaign_id": 4, "channel_id": 3, "impressions": 1200, "clicks": 60, "conversions": 6, "spend": 100, "revenue": 210},
            {"campaign_id": 5, "channel_id": 2, "impressions": 900, "clicks": 45, "conversions": 3, "spend": 90, "revenue": 135},
        ],
    )
    _write_csv(
        work_dir / "channels.csv",
        [
            {"channel_id": 1, "channel": "search", "channel_name": 1, "owner": "growth"},
            {"channel_id": 2, "channel": "social", "channel_name": 2, "owner": "brand"},
            {"channel_id": 3, "channel": "email", "channel_name": 3, "owner": "crm"},
        ],
    )


def _campaign_cleaned(work_dir: Path) -> pd.DataFrame:
    campaigns = pd.read_csv(work_dir / "campaigns.csv")
    channels = pd.read_csv(work_dir / "channels.csv")
    for col in ["impressions", "clicks", "conversions", "spend", "revenue"]:
        campaigns[col] = pd.to_numeric(campaigns[col], errors="coerce")
    campaigns = campaigns.dropna(subset=["clicks", "spend", "revenue"]).copy()
    df = campaigns.merge(channels, on="channel_id", how="left")
    df["ctr"] = df["clicks"] / df["impressions"]
    df["cvr"] = df["conversions"] / df["clicks"]
    df["roi"] = (df["revenue"] - df["spend"]) / df["spend"]
    return df


def _campaign_short_score(work_dir: Path) -> ScoreResult:
    data, message = _load_json(work_dir / "campaign_short_report.json")
    if data is None:
        return ScoreResult(False, 0.0, message, {})
    df = _campaign_cleaned(work_dir)
    top = df.sort_values("roi", ascending=False).iloc[0]["campaign_id"]
    checks = {
        "rows_clean": data.get("rows_clean") == len(df),
        "mean_roi": _close(data.get("mean_roi"), df["roi"].mean()),
        "top_campaign": _same_id(data.get("top_campaign"), top),
    }
    return _score(checks, "campaign short checks completed", data)


def _campaign_medium_score(work_dir: Path) -> ScoreResult:
    path = work_dir / "campaign_medium_channel_summary.csv"
    if not path.exists():
        return ScoreResult(False, 0.0, "campaign_medium_channel_summary.csv was not created", {})
    got = pd.read_csv(path)
    if "channel" not in got.columns and "channel_name" in got.columns:
        got = got.rename(columns={"channel_name": "channel"})
    if "mean_roi" not in got.columns and "roi" in got.columns:
        got = got.rename(columns={"roi": "mean_roi"})
    if "channel" not in got.columns:
        return ScoreResult(False, 0.0, "campaign_medium_channel_summary.csv missing channel/channel_name column", {"columns": list(got.columns)})
    got = got.sort_values("channel").reset_index(drop=True)
    clean = _campaign_cleaned(work_dir)
    got_numeric_channel = "channel" in got.columns and pd.to_numeric(got["channel"], errors="coerce").notna().all()
    group_key = "channel_id" if got_numeric_channel else "channel"
    expected = (
        clean.groupby(group_key)
        .agg(spend=("spend", "sum"), revenue=("revenue", "sum"), conversions=("conversions", "sum"), mean_roi=("roi", "mean"))
        .reset_index()
        .rename(columns={group_key: "channel"})
        .sort_values("channel")
        .reset_index(drop=True)
    )
    checks = {
        "columns": {"channel", "spend", "revenue", "conversions", "mean_roi"}.issubset(got.columns),
        "channels": _same_channel_values(got.get("channel", pd.Series(dtype=str)).tolist(), expected["channel"].tolist()),
        "spend": "spend" in got and _allclose_values(got["spend"], expected["spend"]),
        "revenue": "revenue" in got and _allclose_values(got["revenue"], expected["revenue"]),
        "conversions": "conversions" in got and _allclose_values(got["conversions"], expected["conversions"]),
    }
    return _score(checks, "campaign medium checks completed", got.to_dict("records"))


def _campaign_long_score(work_dir: Path) -> ScoreResult:
    data, message = _load_json(work_dir / "campaign_long_report.json")
    if data is None:
        return ScoreResult(False, 0.0, message, {})
    df = _campaign_cleaned(work_dir)
    top_channel = df.groupby("channel")["revenue"].sum().idxmax()
    top_channel_id = df.groupby("channel_id")["revenue"].sum().idxmax()
    expected_pred = float(np.poly1d(np.polyfit(df["spend"], df["revenue"], 1))(250))
    rows_clean_observed = data.get("rows_clean")
    if isinstance(rows_clean_observed, list):
        rows_clean_ok = len(rows_clean_observed) == len(df)
    else:
        rows_clean_ok = data.get("rows_clean") == len(df)
    observed_top_channel = data.get("top_channel_by_revenue")
    observed_best_campaign = data.get("best_roi_campaign")
    checks = {
        "rows_clean": rows_clean_ok,
        "total_profit": _close(data.get("total_profit"), (df["revenue"] - df["spend"]).sum()),
        "top_channel_by_revenue": (
            str(observed_top_channel).lower() == str(top_channel).lower()
            or _same_id(_field_from_object(observed_top_channel, "channel_id"), top_channel_id)
            or str(_field_from_object(observed_top_channel, "channel")).lower() == str(top_channel).lower()
        ),
        "best_roi_campaign": _same_id(_field_from_object(observed_best_campaign, "campaign_id"), df.sort_values("roi", ascending=False).iloc[0]["campaign_id"]),
        "predicted_revenue_for_spend_250": _close(data.get("predicted_revenue_for_spend_250"), expected_pred, atol=1e-2),
    }
    return _score(checks, "campaign long checks completed", data)


def _campaign_xlong_score(work_dir: Path) -> ScoreResult:
    data, message = _load_json(work_dir / "campaign_xlong_report.json")
    if data is None:
        return ScoreResult(False, 0.0, message, {})
    df = _campaign_cleaned(work_dir)
    top_channel = df.groupby("channel")["revenue"].sum().idxmax()
    top_channel_id = df.groupby("channel_id")["revenue"].sum().idxmax()
    expected_pred = float(np.poly1d(np.polyfit(df["spend"], df["revenue"], 1))(250))
    best_owner = (df["revenue"] - df["spend"]).groupby(df["owner"]).sum().idxmax()
    observed_top_channel = data.get("top_channel_by_revenue")
    checks = {
        "rows_clean": data.get("rows_clean") == len(df),
        "total_profit": _close(data.get("total_profit"), (df["revenue"] - df["spend"]).sum()),
        "top_channel_by_revenue": (
            str(observed_top_channel).lower() == str(top_channel).lower()
            or _same_id(_field_from_object(observed_top_channel, "channel_id"), top_channel_id)
            or str(_field_from_object(observed_top_channel, "channel")).lower() == str(top_channel).lower()
        ),
        "best_roi_campaign": _same_id(data.get("best_roi_campaign"), df.sort_values("roi", ascending=False).iloc[0]["campaign_id"]),
        "predicted_revenue_for_spend_250": _close(data.get("predicted_revenue_for_spend_250"), expected_pred, atol=1e-2),
        "best_owner_by_profit": str(data.get("best_owner_by_profit", "")).lower() == str(best_owner).lower(),
        "overall_ctr": _close(data.get("overall_ctr"), df["clicks"].sum() / df["impressions"].sum()),
    }
    return _score(checks, "campaign xlong checks completed", data)


def _campaign_xxlong_score(work_dir: Path) -> ScoreResult:
    data, message = _load_json(work_dir / "campaign_xxlong_report.json")
    if data is None:
        return ScoreResult(False, 0.0, message, {})
    df = _campaign_cleaned(work_dir)
    top_channel = df.groupby("channel")["revenue"].sum().idxmax()
    top_channel_id = df.groupby("channel_id")["revenue"].sum().idxmax()
    model = np.poly1d(np.polyfit(df["spend"], df["revenue"], 1))
    expected_pred = float(model(250))
    best_owner = (df["revenue"] - df["spend"]).groupby(df["owner"]).sum().idxmax()
    top_channel_roi = df.groupby("channel")["roi"].mean().idxmax()
    observed_top_channel = data.get("top_channel_by_revenue")
    checks = {
        "rows_clean": data.get("rows_clean") == len(df),
        "total_profit": _close(data.get("total_profit"), (df["revenue"] - df["spend"]).sum()),
        "top_channel_by_revenue": (
            str(observed_top_channel).lower() == str(top_channel).lower()
            or _same_id(_field_from_object(observed_top_channel, "channel_id"), top_channel_id)
            or str(_field_from_object(observed_top_channel, "channel")).lower() == str(top_channel).lower()
        ),
        "best_roi_campaign": _same_id(data.get("best_roi_campaign"), df.sort_values("roi", ascending=False).iloc[0]["campaign_id"]),
        "predicted_revenue_for_spend_250": _close(data.get("predicted_revenue_for_spend_250"), expected_pred, atol=1e-2),
        "best_owner_by_profit": str(data.get("best_owner_by_profit", "")).lower() == str(best_owner).lower(),
        "overall_ctr": _close(data.get("overall_ctr"), df["clicks"].sum() / df["impressions"].sum()),
        "top_channel_by_mean_roi": str(data.get("top_channel_by_mean_roi", "")).lower() == str(top_channel_roi).lower(),
        "predicted_profit_for_spend_250": _close(data.get("predicted_profit_for_spend_250"), expected_pred - 250, atol=1e-2),
    }
    return _score(checks, "campaign xxlong checks completed", data)


CAMPAIGN_SHORT_REFERENCE = """import json
import pandas as pd
c=pd.read_csv('campaigns.csv')
ch=pd.read_csv('channels.csv')
for col in ['impressions','clicks','conversions','spend','revenue']:
    c[col]=pd.to_numeric(c[col],errors='coerce')
c=c.dropna(subset=['clicks','spend','revenue']).copy()
df=c.merge(ch,on='channel_id',how='left')
df['roi']=(df['revenue']-df['spend'])/df['spend']
out={'rows_clean':int(len(df)),'mean_roi':float(df['roi'].mean()),'top_campaign':str(df.sort_values('roi',ascending=False).iloc[0]['campaign_id'])}
with open('campaign_short_report.json','w',encoding='utf-8') as f:
    json.dump(out,f)
"""


CAMPAIGN_MEDIUM_REFERENCE = """import pandas as pd
c=pd.read_csv('campaigns.csv')
ch=pd.read_csv('channels.csv')
for col in ['impressions','clicks','conversions','spend','revenue']:
    c[col]=pd.to_numeric(c[col],errors='coerce')
c=c.dropna(subset=['clicks','spend','revenue']).copy()
df=c.merge(ch,on='channel_id',how='left')
df['roi']=(df['revenue']-df['spend'])/df['spend']
summary=df.groupby('channel').agg(spend=('spend','sum'),revenue=('revenue','sum'),conversions=('conversions','sum'),mean_roi=('roi','mean')).reset_index().sort_values('channel')
summary.to_csv('campaign_medium_channel_summary.csv',index=False)
"""


CAMPAIGN_LONG_REFERENCE = """import json
import numpy as np
import pandas as pd
c=pd.read_csv('campaigns.csv')
ch=pd.read_csv('channels.csv')
for col in ['impressions','clicks','conversions','spend','revenue']:
    c[col]=pd.to_numeric(c[col],errors='coerce')
c=c.dropna(subset=['clicks','spend','revenue']).copy()
df=c.merge(ch,on='channel_id',how='left')
df['ctr']=df['clicks']/df['impressions']
df['cvr']=df['conversions']/df['clicks']
df['roi']=(df['revenue']-df['spend'])/df['spend']
top_channel=df.groupby('channel')['revenue'].sum().idxmax()
best_roi=df.sort_values('roi',ascending=False).iloc[0]['campaign_id']
coef=np.polyfit(df['spend'],df['revenue'],1)
pred=float(np.poly1d(coef)(250))
out={'rows_clean':int(len(df)),'total_profit':float((df['revenue']-df['spend']).sum()),'top_channel_by_revenue':str(top_channel),'best_roi_campaign':str(best_roi),'predicted_revenue_for_spend_250':pred}
with open('campaign_long_report.json','w',encoding='utf-8') as f:
    json.dump(out,f)
"""


CAMPAIGN_XLONG_REFERENCE = """import json
import numpy as np
import pandas as pd
c=pd.read_csv('campaigns.csv')
ch=pd.read_csv('channels.csv')
for col in ['impressions','clicks','conversions','spend','revenue']:
    c[col]=pd.to_numeric(c[col],errors='coerce')
c=c.dropna(subset=['clicks','spend','revenue']).copy()
df=c.merge(ch,on='channel_id',how='left')
df['ctr']=df['clicks']/df['impressions']
df['cvr']=df['conversions']/df['clicks']
df['roi']=(df['revenue']-df['spend'])/df['spend']
df['profit']=df['revenue']-df['spend']
top_channel=df.groupby('channel')['revenue'].sum().idxmax()
best_roi=df.sort_values('roi',ascending=False).iloc[0]['campaign_id']
pred=float(np.poly1d(np.polyfit(df['spend'],df['revenue'],1))(250))
best_owner=df.groupby('owner')['profit'].sum().idxmax()
out={'rows_clean':int(len(df)),'total_profit':float(df['profit'].sum()),'top_channel_by_revenue':str(top_channel),'best_roi_campaign':str(best_roi),'predicted_revenue_for_spend_250':pred,'best_owner_by_profit':str(best_owner),'overall_ctr':float(df['clicks'].sum()/df['impressions'].sum())}
with open('campaign_xlong_report.json','w',encoding='utf-8') as f:
    json.dump(out,f)
"""


CAMPAIGN_XXLONG_REFERENCE = """import json
import numpy as np
import pandas as pd
c=pd.read_csv('campaigns.csv')
ch=pd.read_csv('channels.csv')
for col in ['impressions','clicks','conversions','spend','revenue']:
    c[col]=pd.to_numeric(c[col],errors='coerce')
c=c.dropna(subset=['clicks','spend','revenue']).copy()
df=c.merge(ch,on='channel_id',how='left')
df['ctr']=df['clicks']/df['impressions']
df['cvr']=df['conversions']/df['clicks']
df['roi']=(df['revenue']-df['spend'])/df['spend']
df['profit']=df['revenue']-df['spend']
top_channel=df.groupby('channel')['revenue'].sum().idxmax()
best_roi=df.sort_values('roi',ascending=False).iloc[0]['campaign_id']
model=np.poly1d(np.polyfit(df['spend'],df['revenue'],1))
pred=float(model(250))
best_owner=df.groupby('owner')['profit'].sum().idxmax()
top_roi_channel=df.groupby('channel')['roi'].mean().idxmax()
out={'rows_clean':int(len(df)),'total_profit':float(df['profit'].sum()),'top_channel_by_revenue':str(top_channel),'best_roi_campaign':str(best_roi),'predicted_revenue_for_spend_250':pred,'best_owner_by_profit':str(best_owner),'overall_ctr':float(df['clicks'].sum()/df['impressions'].sum()),'top_channel_by_mean_roi':str(top_roi_channel),'predicted_profit_for_spend_250':float(pred-250)}
with open('campaign_xxlong_report.json','w',encoding='utf-8') as f:
    json.dump(out,f)
"""


def _score(checks: dict[str, bool], message: str, observed: object) -> ScoreResult:
    score = sum(bool(v) for v in checks.values()) / len(checks)
    return ScoreResult(bool(score == 1.0), float(score), message, {"checks": checks, "observed": observed})


SCHEMA_NOTES = {
    "orders_kpi": (
        "Input schemas: orders.csv has order_id, customer_id, units, unit_price, price, unit_cost, "
        "discount_rate, discount_amount, order_count, status. customers.csv has customer_id, region, "
        "customer_region, tier. customer_id is a numeric ID in both files. Coerce only numeric measure columns such as units, unit_price, price, "
        "unit_cost, discount_rate, discount_amount, order_count. Preserve join/id/status columns including "
        "customer_id, order_id, and status; do not replace the whole orders table with select_dtypes()."
    ),
    "sensor_quality": (
        "Input schemas: readings.csv has timestamp, sensor_id, raw_temp, raw_pressure, status. "
        "sensor_meta.csv has sensor_id, site, site_id, temp_offset, calibration_temp, pressure_scale, "
        "calibration_pressure. Coerce only raw_temp/raw_pressure as numeric. Preserve timestamp, sensor_id, "
        "status, and site metadata for joins and grouping."
    ),
    "campaign_roi": (
        "Input schemas: campaigns.csv has campaign_id, channel_id, impressions, clicks, conversions, spend, "
        "revenue. channels.csv has channel_id, channel, channel_name, owner. Coerce only impressions, clicks, "
        "conversions, spend, and revenue as numeric. Preserve campaign_id and channel_id for joins and outputs."
    ),
}


def _prompt(family: str, horizon: str, output: str, stages: tuple[str, ...]) -> str:
    stage_text = "\n".join(f"{i + 1}. {stage}" for i, stage in enumerate(stages))
    schema_note = SCHEMA_NOTES.get(family, "")
    return (
        f"This is a {len(stages)}-stage multi-stage planning-coordination horizon task for `{family}`.\n"
        "Write one Python script that performs every stage in order and executes only once at the end.\n"
        f"{schema_note}\n"
        f"Required stages:\n{stage_text}\n"
        f"Required final output: `{output}`. Create exactly this file name and required columns/keys.\n"
        "For JSON reports, write one JSON object with the named keys, not a DataFrame/table dump; cast pandas/numpy scalars with int(), float(), or str() before json.dump.\n"
        "Use pandas/numpy/sklearn only when helpful. Keep the script deterministic and read files from the current directory."
    )


def _task(
    task_id: str,
    family: str,
    horizon: str,
    name: str,
    output: str,
    stages: tuple[str, ...],
    setup,
    score,
    reference_script: str,
) -> ToyTask:
    return ToyTask(
        task_id=task_id,
        name=name,
        prompt=_prompt(family, horizon, output, stages),
        setup=setup,
        score=score,
        task_family=family,
        horizon_level=horizon,
        horizon_stages=HORIZON_ORDER[horizon],
        stage_specs=stages,
        reference_script=reference_script,
    )


ORDERS_STAGES = {
    "short": (
        "Read orders.csv and customers.csv.",
        "Coerce numeric order fields, keep complete orders, and drop rows with missing units.",
        "Join customer regions, compute net_revenue = units * unit_price * (1 - discount_rate) using unit_price or the identical price alias, and write one JSON object orders_short_report.json with rows_clean = cleaned order row count, total_net_revenue = sum of net_revenue, and top_region = region with the largest summed net_revenue.",
    ),
    "medium": (
        "Read orders.csv and customers.csv.",
        "Coerce numeric order fields, keep complete orders, and drop rows with missing units.",
        "Compute gross_revenue, net_revenue, and margin using unit_price or the identical price alias.",
        "Join customer regions and tiers.",
        "Aggregate by region and write orders_medium_region_summary.csv with region, net_revenue, margin, order_count sorted by region.",
    ),
    "long": (
        "Read orders.csv and customers.csv.",
        "Coerce numeric order fields, keep complete orders, and drop rows with missing units.",
        "Compute gross_revenue, net_revenue, and margin using unit_price or the identical price alias.",
        "Join customer regions and tiers.",
        "Aggregate net_revenue by distinct customer_id and count high_value_customers as customers with total net_revenue >= 30, including customers exactly equal to 30.",
        "Fit LinearRegression predicting net_revenue from units, unit_price, discount_rate and compute training MAE.",
        "Write one-row orders_long_report.csv with columns rows_clean, total_net_revenue, total_margin, top_region, high_value_customers, model_mae.",
    ),
    "xlong": (
        "Read orders.csv and customers.csv.",
        "Coerce numeric order fields, keep complete orders, and drop rows with missing units.",
        "Compute gross_revenue, net_revenue, and margin using unit_price or the identical price alias.",
        "Join customer regions and tiers.",
        "Aggregate net_revenue by distinct customer_id and count high_value_customers as customers with total net_revenue >= 30, including customers exactly equal to 30.",
        "Fit LinearRegression predicting net_revenue from units, unit_price, discount_rate and compute training MAE.",
        "Aggregate total margin by tier and identify best_tier_by_margin.",
        "Compute margin_per_unit = total_margin / total units over cleaned rows.",
        "Write one JSON object orders_xlong_report.json with rows_clean, total_net_revenue, total_margin, top_region, high_value_customers, model_mae, best_tier_by_margin, margin_per_unit.",
    ),
    "xxlong": (
        "Read orders.csv and customers.csv.",
        "Coerce numeric order fields, keep complete orders, and drop rows with missing units.",
        "Compute gross_revenue, net_revenue, and margin using unit_price or the identical price alias.",
        "Join customer regions and tiers.",
        "Aggregate net_revenue by distinct customer_id and count high_value_customers as customers with total net_revenue >= 30, including customers exactly equal to 30.",
        "Fit LinearRegression predicting net_revenue from units, unit_price, discount_rate and compute training MAE.",
        "Aggregate total margin by tier and identify best_tier_by_margin.",
        "Compute margin_per_unit = total_margin / total units over cleaned rows.",
        "Identify top_customer_id_by_net_revenue from customer net revenue totals.",
        "Use the fitted model to predict net revenue for units=6, unit_price=9, discount_rate=0.",
        "Write one JSON object orders_xxlong_report.json with rows_clean, total_net_revenue, total_margin, top_region, high_value_customers, model_mae, best_tier_by_margin, margin_per_unit, top_customer_id_by_net_revenue, predicted_net_revenue_for_units_6_price_9_discount_0.",
    ),
}


SENSOR_STAGES = {
    "short": (
        "Read readings.csv and sensor_meta.csv.",
        "Coerce raw_temp/raw_pressure, keep ok readings, and drop rows missing raw_temp.",
        "Join calibration metadata, compute adjusted_temp = raw_temp + temp_offset, adjusted_pressure = raw_pressure * pressure_scale, and alert = (adjusted_temp > 76) | (adjusted_pressure > 105). Write exactly one JSON object sensor_short_report.json, not a list and not grouped by sensor_id, with rows_clean = count of cleaned reading rows where status == ok and raw_temp is present, mean_adjusted_temp = mean adjusted_temp over those cleaned rows, and alert_count = number of cleaned rows where alert is true.",
    ),
    "medium": (
        "Read readings.csv and sensor_meta.csv.",
        "Coerce raw_temp/raw_pressure, keep ok readings, and drop rows missing raw_temp.",
        "Join sensor metadata.",
        "Compute adjusted_temp, adjusted_pressure, and alert where adjusted_temp > 76 or adjusted_pressure > 105.",
        "Aggregate by site and write sensor_medium_site_summary.csv with site, mean_adjusted_temp, alert_count, reading_count sorted by site. Use a safe pattern like groupby('site').agg(mean_adjusted_temp=('adjusted_temp','mean'), alert_count=('alert','sum'), reading_count=('sensor_id','size')).reset_index().",
    ),
    "long": (
        "Read readings.csv and sensor_meta.csv.",
        "Coerce raw_temp/raw_pressure, keep ok readings, and drop rows missing raw_temp.",
        "Join sensor metadata.",
        "Parse timestamp and derive hour.",
        "Compute adjusted_temp, adjusted_pressure, and alert where adjusted_temp > 76 or adjusted_pressure > 105.",
        "Aggregate site-hour alerts and write sensor_long_hourly_summary.csv.",
        "Write sensor_long_report.json with rows_clean, total_alerts, worst_site, peak_hour.",
    ),
    "xlong": (
        "Read readings.csv and sensor_meta.csv.",
        "Coerce raw_temp/raw_pressure, keep ok readings, and drop rows missing raw_temp.",
        "Join sensor metadata.",
        "Parse timestamp and derive hour.",
        "Compute adjusted_temp = raw_temp + temp_offset and adjusted_pressure = raw_pressure * pressure_scale.",
        "Compute alert = (adjusted_temp > 76) | (adjusted_pressure > 105).",
        "Aggregate site-hour alerts as an intermediate check.",
        "Aggregate by site: temp_std is std(adjusted_temp), reading_count is cleaned rows, alert_count is alert rows, and alert_rate = alert_count / reading_count. Write sensor_xlong_site_summary.csv with at least site, temp_std, alert_rate, reading_count sorted by site.",
        "Write one JSON object sensor_xlong_report.json with rows_clean, total_alerts, worst_site, peak_hour, best_site_by_temp_stability, mean_alert_rate = total_alerts / rows_clean over all cleaned rows, not the unweighted mean of site alert rates.",
    ),
    "xxlong": (
        "Read readings.csv and sensor_meta.csv.",
        "Coerce raw_temp/raw_pressure, keep ok readings, and drop rows missing raw_temp.",
        "Join sensor metadata.",
        "Parse timestamp and derive hour.",
        "Compute adjusted_temp = raw_temp + temp_offset and adjusted_pressure = raw_pressure * pressure_scale.",
        "Compute alert = (adjusted_temp > 76) | (adjusted_pressure > 105).",
        "Aggregate site-hour alerts as an intermediate check.",
        "Aggregate by site: temp_std is std(adjusted_temp), reading_count is cleaned rows, alert_count is alert rows, and alert_rate = alert_count / reading_count. Write sensor_xxlong_site_summary.csv with at least site, temp_std, alert_rate, reading_count sorted by site.",
        "Identify best_site_by_temp_stability as the site with the smallest adjusted_temp standard deviation.",
        "Identify worst_sensor_by_alerts and peak_site_hour formatted as site-hour, for example beta-10.",
        "Write one JSON object sensor_xxlong_report.json with rows_clean, total_alerts, worst_site, peak_hour, best_site_by_temp_stability, mean_alert_rate = total_alerts / rows_clean over all cleaned rows, worst_sensor_by_alerts, peak_site_hour.",
    ),
}


CAMPAIGN_STAGES = {
    "short": (
        "Read campaigns.csv and channels.csv.",
        "Coerce only impressions, clicks, conversions, spend, and revenue with pd.to_numeric; campaign_id and channel_id are numeric IDs and must be kept.",
        "Join channels on channel_id, compute ROI = (revenue - spend) / spend, and write one JSON object campaign_short_report.json where rows_clean = cleaned campaign row count, mean_roi = mean ROI over cleaned rows, and top_campaign = campaign_id with the highest ROI.",
    ),
    "medium": (
        "Read campaigns.csv and channels.csv.",
        "Coerce only impressions, clicks, conversions, spend, and revenue with pd.to_numeric; keep campaign_id/channel_id.",
        "Join channel names on channel_id.",
        "Compute ROI for each campaign.",
        "Aggregate by channel and write campaign_medium_channel_summary.csv with channel, spend, revenue, conversions, mean_roi sorted by channel.",
    ),
    "long": (
        "Read campaigns.csv and channels.csv.",
        "Coerce only impressions, clicks, conversions, spend, and revenue with pd.to_numeric; keep campaign_id/channel_id.",
        "Join channel names on channel_id.",
        "Compute CTR, CVR, and ROI.",
        "Aggregate channel revenue and identify the top channel by revenue.",
        "Fit a simple numpy polynomial line predicting revenue from spend and predict revenue for spend 250.",
        "Write campaign_long_report.json with rows_clean, total_profit, top_channel_by_revenue, best_roi_campaign, predicted_revenue_for_spend_250.",
    ),
    "xlong": (
        "Read campaigns.csv and channels.csv.",
        "Coerce only impressions, clicks, conversions, spend, and revenue with pd.to_numeric; keep campaign_id/channel_id.",
        "Join channel names and owners on channel_id.",
        "Compute CTR = clicks / impressions, CVR = conversions / clicks, ROI = (revenue - spend) / spend, and profit = revenue - spend.",
        "Aggregate channel revenue and identify top_channel_by_revenue.",
        "Identify best_roi_campaign as the campaign_id with the highest ROI.",
        "Fit a simple numpy polynomial line predicting revenue from spend and predict revenue for spend 250.",
        "Aggregate profit by owner and identify best_owner_by_profit.",
        "Write one JSON object campaign_xlong_report.json with rows_clean, total_profit, top_channel_by_revenue, best_roi_campaign, predicted_revenue_for_spend_250, best_owner_by_profit, overall_ctr = total clicks / total impressions over all cleaned rows, not mean row CTR.",
    ),
    "xxlong": (
        "Read campaigns.csv and channels.csv.",
        "Coerce only impressions, clicks, conversions, spend, and revenue with pd.to_numeric; keep campaign_id/channel_id.",
        "Join channel names and owners on channel_id.",
        "Compute CTR = clicks / impressions, CVR = conversions / clicks, ROI = (revenue - spend) / spend, and profit = revenue - spend.",
        "Aggregate channel revenue and identify top_channel_by_revenue.",
        "Identify best_roi_campaign as the campaign_id with the highest ROI.",
        "Fit a simple numpy polynomial line predicting revenue from spend and predict revenue for spend 250.",
        "Aggregate profit by owner and identify best_owner_by_profit.",
        "Compute overall_ctr = total clicks / total impressions over all cleaned rows, not mean row CTR.",
        "Aggregate mean ROI by channel and identify top_channel_by_mean_roi.",
        "Write one JSON object campaign_xxlong_report.json with rows_clean, total_profit, top_channel_by_revenue, best_roi_campaign, predicted_revenue_for_spend_250, best_owner_by_profit, overall_ctr, top_channel_by_mean_roi, predicted_profit_for_spend_250.",
    ),
}


TASKS: dict[str, ToyTask] = {
    "orders_kpi_short": _task("orders_kpi_short", "orders_kpi", "short", "Orders KPI short horizon", "orders_short_report.json", ORDERS_STAGES["short"], _orders_setup, _orders_short_score, ORDERS_SHORT_REFERENCE),
    "orders_kpi_medium": _task("orders_kpi_medium", "orders_kpi", "medium", "Orders KPI medium horizon", "orders_medium_region_summary.csv", ORDERS_STAGES["medium"], _orders_setup, _orders_medium_score, ORDERS_MEDIUM_REFERENCE),
    "orders_kpi_long": _task("orders_kpi_long", "orders_kpi", "long", "Orders KPI long horizon", "orders_long_report.csv", ORDERS_STAGES["long"], _orders_setup, _orders_long_score, ORDERS_LONG_REFERENCE),
    "orders_kpi_xlong": _task("orders_kpi_xlong", "orders_kpi", "xlong", "Orders KPI xlong horizon", "orders_xlong_report.json", ORDERS_STAGES["xlong"], _orders_setup, _orders_xlong_score, ORDERS_XLONG_REFERENCE),
    "orders_kpi_xxlong": _task("orders_kpi_xxlong", "orders_kpi", "xxlong", "Orders KPI xxlong horizon", "orders_xxlong_report.json", ORDERS_STAGES["xxlong"], _orders_setup, _orders_xxlong_score, ORDERS_XXLONG_REFERENCE),
    "sensor_quality_short": _task("sensor_quality_short", "sensor_quality", "short", "Sensor quality short horizon", "sensor_short_report.json", SENSOR_STAGES["short"], _sensor_setup, _sensor_short_score, SENSOR_SHORT_REFERENCE),
    "sensor_quality_medium": _task("sensor_quality_medium", "sensor_quality", "medium", "Sensor quality medium horizon", "sensor_medium_site_summary.csv", SENSOR_STAGES["medium"], _sensor_setup, _sensor_medium_score, SENSOR_MEDIUM_REFERENCE),
    "sensor_quality_long": _task("sensor_quality_long", "sensor_quality", "long", "Sensor quality long horizon", "sensor_long_report.json", SENSOR_STAGES["long"], _sensor_setup, _sensor_long_score, SENSOR_LONG_REFERENCE),
    "sensor_quality_xlong": _task("sensor_quality_xlong", "sensor_quality", "xlong", "Sensor quality xlong horizon", "sensor_xlong_report.json", SENSOR_STAGES["xlong"], _sensor_setup, _sensor_xlong_score, SENSOR_XLONG_REFERENCE),
    "sensor_quality_xxlong": _task("sensor_quality_xxlong", "sensor_quality", "xxlong", "Sensor quality xxlong horizon", "sensor_xxlong_report.json", SENSOR_STAGES["xxlong"], _sensor_setup, _sensor_xxlong_score, SENSOR_XXLONG_REFERENCE),
    "campaign_roi_short": _task("campaign_roi_short", "campaign_roi", "short", "Campaign ROI short horizon", "campaign_short_report.json", CAMPAIGN_STAGES["short"], _campaign_setup, _campaign_short_score, CAMPAIGN_SHORT_REFERENCE),
    "campaign_roi_medium": _task("campaign_roi_medium", "campaign_roi", "medium", "Campaign ROI medium horizon", "campaign_medium_channel_summary.csv", CAMPAIGN_STAGES["medium"], _campaign_setup, _campaign_medium_score, CAMPAIGN_MEDIUM_REFERENCE),
    "campaign_roi_long": _task("campaign_roi_long", "campaign_roi", "long", "Campaign ROI long horizon", "campaign_long_report.json", CAMPAIGN_STAGES["long"], _campaign_setup, _campaign_long_score, CAMPAIGN_LONG_REFERENCE),
    "campaign_roi_xlong": _task("campaign_roi_xlong", "campaign_roi", "xlong", "Campaign ROI xlong horizon", "campaign_xlong_report.json", CAMPAIGN_STAGES["xlong"], _campaign_setup, _campaign_xlong_score, CAMPAIGN_XLONG_REFERENCE),
    "campaign_roi_xxlong": _task("campaign_roi_xxlong", "campaign_roi", "xxlong", "Campaign ROI xxlong horizon", "campaign_xxlong_report.json", CAMPAIGN_STAGES["xxlong"], _campaign_setup, _campaign_xxlong_score, CAMPAIGN_XXLONG_REFERENCE),
}


def selected_horizon_tasks(task_ids: list[str] | None = None, families: list[str] | None = None, horizons: list[str] | None = None) -> list[ToyTask]:
    tasks = list(TASKS.values())
    if families:
        wanted_families = set(families)
        tasks = [task for task in tasks if task.task_family in wanted_families]
    if horizons:
        wanted_horizons = set(horizons)
        tasks = [task for task in tasks if task.horizon_level in wanted_horizons]
    if task_ids:
        missing = [task_id for task_id in task_ids if task_id not in TASKS]
        if missing:
            raise KeyError(f"Unknown Phase 3 task id(s): {', '.join(missing)}")
        task_map = {task.task_id: task for task in tasks}
        tasks = [task_map[task_id] for task_id in task_ids if task_id in task_map]
    return sorted(tasks, key=lambda task: (task.task_family, HORIZON_ORDER[task.horizon_level]))
