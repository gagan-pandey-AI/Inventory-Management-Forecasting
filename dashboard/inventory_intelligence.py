"""Inventory intelligence calculations using explicit, auditable inputs."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd


TURNOVER_COLUMNS = [
    "product_id",
    "product_name",
    "average_inventory_units",
    "average_inventory_value",
    "cogs",
    "turnover_ratio",
    "days_inventory_outstanding",
    "annualized_turnover",
    "movement_classification",
    "data_status",
    "snapshot_count",
]

TURNOVER_SNAPSHOT_REQUIRED_COLUMNS = {
    "product_id",
    "snapshot_date",
    "stock_quantity",
    "unit_cost",
}
TURNOVER_SALES_REQUIRED_COLUMNS = {
    "product_id",
    "sale_date",
    "quantity_sold",
    "unit_cost",
}


def _as_date(value: date | str | pd.Timestamp | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now().normalize()
    return pd.Timestamp(value).normalize()


def _number(value: object) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else math.nan


def read_inventory_turnover_csv(
    source: IO[bytes] | IO[str] | str | Path,
) -> pd.DataFrame:
    """Read an uploaded turnover CSV while preserving product identifiers."""
    return pd.read_csv(source, dtype={"product_id": "string"})


def validate_inventory_turnover_data(
    snapshots: pd.DataFrame,
    sales: pd.DataFrame,
) -> list[str]:
    """Return user-readable issues found in the uploaded turnover data."""
    errors: list[str] = []
    sources = (
        (
            "Inventory snapshots",
            snapshots,
            TURNOVER_SNAPSHOT_REQUIRED_COLUMNS,
            "snapshot_date",
            ("stock_quantity", "unit_cost"),
        ),
        (
            "Costed sales",
            sales,
            TURNOVER_SALES_REQUIRED_COLUMNS,
            "sale_date",
            ("quantity_sold", "unit_cost"),
        ),
    )
    for label, frame, required, date_column, numeric_columns in sources:
        if frame.empty:
            errors.append(f"{label} CSV is empty.")
            continue

        missing_columns = required.difference(frame.columns)
        if missing_columns:
            errors.append(
                f"{label} CSV is missing required columns: "
                + ", ".join(sorted(missing_columns))
                + "."
            )

        for column in required.intersection(frame.columns):
            values = frame[column]
            missing_values = values.isna()
            if column == "product_id":
                missing_values |= values.astype("string").str.strip().eq("")
            if missing_values.any():
                errors.append(
                    f"{label} CSV has missing values in '{column}' "
                    f"({int(missing_values.sum())} row(s))."
                )

        if date_column in frame.columns:
            parsed_dates = pd.to_datetime(
                frame[date_column],
                errors="coerce",
            )
            invalid_dates = parsed_dates.isna()
            if invalid_dates.any() and not invalid_dates.equals(
                frame[date_column].isna()
            ):
                errors.append(
                    f"{label} CSV has invalid dates in '{date_column}' "
                    f"({int((invalid_dates & frame[date_column].notna()).sum())} "
                    "row(s))."
                )

        for column in numeric_columns:
            if column not in frame.columns:
                continue
            values = frame[column]
            parsed_values = pd.to_numeric(values, errors="coerce")
            invalid_values = parsed_values.isna() & values.notna()
            if invalid_values.any():
                errors.append(
                    f"{label} CSV has non-numeric values in '{column}' "
                    f"({int(invalid_values.sum())} row(s))."
                )
            non_finite_values = parsed_values.notna() & ~np.isfinite(
                parsed_values
            )
            if non_finite_values.any():
                errors.append(
                    f"{label} CSV has non-finite values in '{column}'."
                )
            negative_values = parsed_values.lt(0)
            if negative_values.any():
                errors.append(
                    f"{label} CSV has negative values in '{column}' "
                    f"({int(negative_values.sum())} row(s))."
                )
    return errors


def build_inventory_analysis(
    inventory: pd.DataFrame,
    forecast: pd.DataFrame,
    historical_sales: pd.DataFrame | None = None,
    *,
    as_of_date: date | str | pd.Timestamp | None = None,
    forecast_horizon_days: int = 30,
    historical_lookback_days: int = 90,
    safety_buffer_days: int = 7,
    overstock_target_days: int = 60,
    reorder_target_days: int = 30,
) -> pd.DataFrame:
    """Combine live stock with dated demand, retaining missing values as unknown."""
    required = {
        "product_id",
        "product_name",
        "stock_quantity",
        "reorder_level",
        "lead_time_days",
    }
    missing = required.difference(inventory.columns)
    if missing:
        raise ValueError(
            "Inventory data is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if forecast_horizon_days < 1:
        raise ValueError("Forecast horizon must be at least one day.")
    if historical_lookback_days < 1:
        raise ValueError("Historical lookback must be at least one day.")
    if safety_buffer_days < 0 or overstock_target_days < 1:
        raise ValueError("Inventory coverage thresholds are invalid.")
    if reorder_target_days < 0:
        raise ValueError("Reorder target days cannot be negative.")

    as_of = _as_date(as_of_date)
    inventory_data = inventory.copy()
    duplicate_ids = set(
        inventory_data.loc[
            inventory_data["product_id"].duplicated(keep=False),
            "product_id",
        ].dropna()
    )
    inventory_data = inventory_data.drop_duplicates(
        subset="product_id",
        keep="first",
    )

    for column in ("stock_quantity", "reorder_level", "lead_time_days"):
        inventory_data[column] = pd.to_numeric(
            inventory_data[column],
            errors="coerce",
        )

    forecast_summary = pd.DataFrame(
        columns=["product_id", "forecast_daily_demand", "forecast_days"]
    )
    forecast_columns = {
        "product_id",
        "forecast_date",
        "predicted_demand",
    }
    if forecast_columns.issubset(forecast.columns):
        forecast_data = forecast[list(forecast_columns)].copy()
        forecast_data["forecast_date"] = pd.to_datetime(
            forecast_data["forecast_date"],
            errors="coerce",
        ).dt.normalize()
        forecast_data["predicted_demand"] = pd.to_numeric(
            forecast_data["predicted_demand"],
            errors="coerce",
        )
        forecast_data = forecast_data.loc[
            forecast_data["forecast_date"].ge(as_of)
            & forecast_data["predicted_demand"].ge(0)
        ].dropna(subset=["product_id", "forecast_date", "predicted_demand"])
        forecast_data = forecast_data.sort_values("forecast_date")
        forecast_data["forecast_day"] = forecast_data.groupby(
            ["product_id", "forecast_date"]
        )["predicted_demand"].transform("sum")
        forecast_data = forecast_data.drop_duplicates(
            subset=["product_id", "forecast_date"]
        )
        forecast_data["forecast_day_number"] = forecast_data.groupby(
            "product_id"
        )["forecast_date"].rank(method="first")
        forecast_data = forecast_data.loc[
            forecast_data["forecast_day_number"] <= forecast_horizon_days
        ]
        forecast_summary = (
            forecast_data.groupby("product_id", as_index=False)
            .agg(
                forecast_daily_demand=("forecast_day", "mean"),
                forecast_days=("forecast_date", "nunique"),
            )
        )

    historical_summary = pd.DataFrame(
        columns=["product_id", "historical_daily_demand"]
    )
    if historical_sales is not None and {
        "product_id",
        "sale_date",
        "quantity_sold",
    }.issubset(historical_sales.columns):
        sales_data = historical_sales[
            ["product_id", "sale_date", "quantity_sold"]
        ].copy()
        sales_data["sale_date"] = pd.to_datetime(
            sales_data["sale_date"],
            errors="coerce",
        ).dt.normalize()
        sales_data["quantity_sold"] = pd.to_numeric(
            sales_data["quantity_sold"],
            errors="coerce",
        )
        sales_data = sales_data.loc[
            sales_data["sale_date"].between(
                as_of - pd.Timedelta(days=historical_lookback_days - 1),
                as_of,
            )
            & sales_data["quantity_sold"].ge(0)
        ].dropna(subset=["product_id", "sale_date", "quantity_sold"])
        if not sales_data.empty:
            historical_summary = (
                sales_data.groupby("product_id", as_index=False)
                .agg(historical_units_sold=("quantity_sold", "sum"))
            )
            historical_summary["historical_daily_demand"] = (
                historical_summary["historical_units_sold"]
                / historical_lookback_days
            )
            historical_summary = historical_summary[
                ["product_id", "historical_daily_demand"]
            ]

    result = inventory_data.merge(
        forecast_summary,
        on="product_id",
        how="left",
        validate="one_to_one",
    ).merge(
        historical_summary,
        on="product_id",
        how="left",
        validate="one_to_one",
    )
    result["demand_source"] = np.where(
        result["forecast_daily_demand"].notna(),
        "Forecast",
        np.where(
            result["historical_daily_demand"].notna(),
            "Historical sales fallback",
            "Unavailable",
        ),
    )
    result["average_daily_demand"] = result[
        "forecast_daily_demand"
    ].combine_first(result["historical_daily_demand"])
    stock_values = result["stock_quantity"].to_numpy(dtype=float)
    demand_values = result["average_daily_demand"].to_numpy(dtype=float)
    coverage_values = np.full(len(result), np.nan)
    np.divide(
        stock_values,
        demand_values,
        out=coverage_values,
        where=np.isfinite(demand_values) & (demand_values > 0),
    )
    result["days_inventory_remaining"] = coverage_values
    result["expected_demand_overstock_horizon"] = (
        result["average_daily_demand"] * overstock_target_days
    )
    result["excess_quantity"] = np.nan
    result["classification"] = "Data unavailable"
    result["classification_explanation"] = (
        "Current stock, demand, or lead-time data is incomplete."
    )
    result["expected_lead_time_demand"] = (
        result["average_daily_demand"] * result["lead_time_days"]
    )
    result["reorder_point"] = np.nan
    result["recommended_order_quantity"] = 0
    result["reorder_urgency"] = "No recommendation"
    result["estimated_arrival_date"] = pd.NaT
    result["supplier_display"] = "Supplier unavailable"
    if "supplier_name" in result.columns:
        supplier_names = result["supplier_name"].astype("string").str.strip()
        result["supplier_display"] = supplier_names.where(
            supplier_names.ne("").fillna(False),
            "Supplier unavailable",
        )
    result["pending_order_check"] = "Not available; verify open orders"
    if "pending_order_quantity" in result.columns:
        result["pending_order_quantity"] = pd.to_numeric(
            result["pending_order_quantity"],
            errors="coerce",
        )
        result.loc[
            result["pending_order_quantity"].notna(),
            "pending_order_check",
        ] = "Included in inventory position"
    else:
        result["pending_order_quantity"] = np.nan

    for index, row in result.iterrows():
        stock = _number(row["stock_quantity"])
        reorder_level = _number(row["reorder_level"])
        lead_time = _number(row["lead_time_days"])
        demand = _number(row["average_daily_demand"])
        pending = _number(row["pending_order_quantity"])
        duplicate = row["product_id"] in duplicate_ids

        if duplicate:
            result.at[index, "classification"] = "Data unavailable"
            result.at[index, "classification_explanation"] = (
                "Multiple current inventory rows share this product ID; "
                "reconcile them before using product-level recommendations."
            )
            continue
        if not math.isfinite(stock) or stock < 0:
            result.at[index, "classification_explanation"] = (
                "A valid, non-negative current stock quantity is required."
            )
            continue
        if not math.isfinite(demand) or demand < 0:
            if stock == 0 or (
                math.isfinite(reorder_level) and stock <= reorder_level
            ):
                result.at[index, "classification"] = "Critical Low Stock"
                result.at[index, "classification_explanation"] = (
                    "Current stock is zero or at/below the recorded reorder "
                    "level; demand coverage cannot be calculated."
                )
            else:
                result.at[index, "classification_explanation"] = (
                    "No dated forecast or historical demand is available."
                )
            continue

        coverage = (
            stock / demand
            if demand > 0
            else math.inf
        )
        if demand == 0:
            result.at[index, "excess_quantity"] = stock
            if stock == 0 or (
                math.isfinite(reorder_level) and stock <= reorder_level
            ):
                classification = "Critical Low Stock"
                explanation = (
                    "Current stock is at or below the recorded reorder level; "
                    "the available forecast has zero demand."
                )
            else:
                classification = "Overstock"
                explanation = (
                    "No positive demand is forecast for this product; all "
                    "current units exceed forecast demand over the selected "
                    f"{overstock_target_days}-day target."
                )
        elif (
            stock == 0
            or (math.isfinite(reorder_level) and stock <= reorder_level)
            or (math.isfinite(lead_time) and lead_time >= 0 and coverage <= lead_time)
        ):
            classification = "Critical Low Stock"
            explanation = (
                "Stock is at/below the recorded reorder level or is expected "
                "to run out before the supplier lead time."
            )
        elif math.isfinite(lead_time) and lead_time >= 0 and coverage <= (
            lead_time + safety_buffer_days
        ):
            classification = "Low Stock"
            explanation = (
                "Projected coverage reaches the supplier lead time plus the "
                f"configured {safety_buffer_days}-day safety buffer."
            )
        elif not math.isfinite(lead_time) and coverage <= safety_buffer_days:
            classification = "Low Stock"
            explanation = (
                "Coverage is within the configured safety buffer; supplier "
                "lead time is unavailable."
            )
        elif coverage > overstock_target_days:
            classification = "Overstock"
            excess = max(
                stock - demand * overstock_target_days,
                0,
            )
            result.at[index, "excess_quantity"] = excess
            explanation = (
                f"Stock exceeds the configured {overstock_target_days}-day "
                "forecast-demand target."
            )
        else:
            classification = "Healthy Stock"
            explanation = (
                "Stock is above the low-stock thresholds and within the "
                f"configured {overstock_target_days}-day coverage target."
            )

        result.at[index, "classification"] = classification
        result.at[index, "classification_explanation"] = explanation

        if (
            not duplicate
            and math.isfinite(reorder_level)
            and reorder_level >= 0
            and math.isfinite(lead_time)
            and lead_time >= 0
            and stock >= 0
            and demand >= 0
        ):
            pending_value = pending if math.isfinite(pending) and pending >= 0 else 0
            inventory_position = stock + pending_value
            lead_demand = demand * lead_time
            buffer_demand = demand * safety_buffer_days
            reorder_point = max(
                reorder_level,
                lead_demand + buffer_demand,
            )
            result.at[index, "reorder_point"] = reorder_point
            if inventory_position <= reorder_point:
                target = max(
                    reorder_level,
                    demand
                    * (
                        lead_time
                        + safety_buffer_days
                        + reorder_target_days
                    ),
                )
                quantity = max(
                    math.ceil(target - inventory_position),
                    0,
                )
                if quantity > 0:
                    result.at[index, "recommended_order_quantity"] = quantity
                    if inventory_position <= lead_demand:
                        urgency = "Critical"
                    elif inventory_position <= reorder_level:
                        urgency = "High"
                    else:
                        urgency = "Moderate"
                    result.at[index, "reorder_urgency"] = urgency
                    result.at[index, "estimated_arrival_date"] = (
                        as_of + pd.Timedelta(days=lead_time)
                    )

    return result


def calculate_inventory_turnover(
    snapshots: pd.DataFrame | None,
    sales: pd.DataFrame | None,
    *,
    period_days: int = 90,
    as_of_date: date | str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, str]:
    """Calculate cost-based turnover only when costed sales and snapshots exist."""
    if period_days < 1:
        raise ValueError("Turnover period must be at least one day.")
    if snapshots is None or snapshots.empty:
        return (
            pd.DataFrame(columns=TURNOVER_COLUMNS),
            "Upload at least two dated inventory snapshots per product, "
            "including unit cost.",
        )
    if sales is None:
        return (
            pd.DataFrame(columns=TURNOVER_COLUMNS),
            "Costed sales transactions are required to calculate COGS.",
        )

    missing_snapshot = TURNOVER_SNAPSHOT_REQUIRED_COLUMNS.difference(
        snapshots.columns
    )
    missing_sales = TURNOVER_SALES_REQUIRED_COLUMNS.difference(sales.columns)
    if missing_snapshot:
        return (
            pd.DataFrame(columns=TURNOVER_COLUMNS),
            "Snapshot CSV is missing required columns: "
            + ", ".join(sorted(missing_snapshot))
            + ".",
        )
    if missing_sales:
        return (
            pd.DataFrame(columns=TURNOVER_COLUMNS),
            "Sales CSV is missing required columns: "
            + ", ".join(sorted(missing_sales))
            + ". unit_price is selling price and is not a substitute for unit_cost.",
        )

    duplicate_snapshot_count = int(snapshots.duplicated().sum())
    duplicate_sales_count = int(sales.duplicated().sum())
    snapshot_data = snapshots.drop_duplicates().copy()
    sales_data = sales.drop_duplicates().copy()
    snapshot_data["snapshot_date"] = pd.to_datetime(
        snapshot_data["snapshot_date"],
        errors="coerce",
    ).dt.normalize()
    sales_data["sale_date"] = pd.to_datetime(
        sales_data["sale_date"],
        errors="coerce",
    ).dt.normalize()
    for frame, columns in (
        (snapshot_data, ("stock_quantity", "unit_cost")),
        (sales_data, ("quantity_sold", "unit_cost")),
    ):
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    valid_snapshot_dates = snapshot_data["snapshot_date"].dropna()
    if valid_snapshot_dates.empty:
        return (
            pd.DataFrame(columns=TURNOVER_COLUMNS),
            "Snapshot CSV contains no valid snapshot dates.",
        )
    as_of = _as_date(as_of_date) if as_of_date is not None else (
        valid_snapshot_dates.max()
    )
    period_start = as_of - pd.Timedelta(days=period_days)
    snapshot_data = snapshot_data.loc[
        snapshot_data["snapshot_date"].between(period_start, as_of)
        | snapshot_data["snapshot_date"].isna()
    ]
    sales_data = sales_data.loc[
        sales_data["sale_date"].between(period_start, as_of)
        | sales_data["sale_date"].isna()
    ]
    product_ids = pd.Index(
        snapshot_data["product_id"].dropna().unique()
    ).union(sales_data["product_id"].dropna().unique())

    records: list[dict[str, object]] = []
    for product_id in product_ids:
        product_snapshots = snapshot_data.loc[
            snapshot_data["product_id"].eq(product_id)
        ].copy()
        product_sales = sales_data.loc[
            sales_data["product_id"].eq(product_id)
        ].copy()
        product_name = (
            product_snapshots["product_name"].dropna().iloc[0]
            if "product_name" in product_snapshots.columns
            and product_snapshots["product_name"].notna().any()
            else None
        )
        base: dict[str, object] = {
            "product_id": product_id,
            "product_name": product_name,
            "average_inventory_units": np.nan,
            "average_inventory_value": np.nan,
            "cogs": np.nan,
            "turnover_ratio": np.nan,
            "days_inventory_outstanding": np.nan,
            "annualized_turnover": np.nan,
            "movement_classification": "Unavailable",
            "data_status": "Insufficient cost or snapshot data",
            "snapshot_count": 0,
        }

        sales_cost_complete = True
        if product_sales.empty:
            cogs = 0.0
        elif (
            product_sales[
                ["sale_date", "quantity_sold", "unit_cost"]
            ].isna().any().any()
        ):
            cogs = math.nan
            sales_cost_complete = False
            base["data_status"] = (
                "Missing sale date, quantity, or per-sale unit cost; COGS and "
                "turnover are not reported."
            )
        elif (
            product_sales["quantity_sold"].lt(0).any()
            or product_sales["unit_cost"].lt(0).any()
        ):
            cogs = math.nan
            sales_cost_complete = False
            base["data_status"] = (
                "Negative sales quantity or unit cost; data requires review."
            )
        else:
            cogs = float(
                (
                    product_sales["quantity_sold"]
                    * product_sales["unit_cost"]
                ).sum()
            )
        base["cogs"] = cogs

        snapshot_data_complete = not (
            product_snapshots[
                ["snapshot_date", "stock_quantity", "unit_cost"]
            ].isna().any().any()
            or product_snapshots["stock_quantity"].lt(0).any()
            or product_snapshots["unit_cost"].lt(0).any()
        )
        if not snapshot_data_complete:
            if sales_cost_complete:
                base["data_status"] = (
                    "Missing or invalid snapshot date, quantity, or unit cost; "
                    "average inventory and turnover are not reported."
                )
            records.append(base)
            continue

        product_snapshots = product_snapshots.dropna(
            subset=["snapshot_date", "stock_quantity", "unit_cost"]
        )
        product_snapshots = product_snapshots.loc[
            product_snapshots["stock_quantity"].ge(0)
            & product_snapshots["unit_cost"].ge(0)
        ]
        product_snapshots["inventory_value"] = (
            product_snapshots["stock_quantity"]
            * product_snapshots["unit_cost"]
        )
        dated_snapshots = (
            product_snapshots.groupby("snapshot_date", as_index=False)
            .agg(
                stock_quantity=("stock_quantity", "sum"),
                inventory_value=("inventory_value", "sum"),
            )
            .sort_values("snapshot_date")
        )
        base["snapshot_count"] = len(dated_snapshots)
        if len(dated_snapshots) < 2:
            if sales_cost_complete:
                base["data_status"] = (
                    "At least two dated inventory snapshots are required."
                )
            records.append(base)
            continue

        elapsed_days = (
            dated_snapshots["snapshot_date"].diff().dt.total_seconds()
            / 86400
        ).iloc[1:].to_numpy()
        if not np.isfinite(elapsed_days).all() or elapsed_days.sum() <= 0:
            base["data_status"] = "Snapshot dates must span a positive period."
            records.append(base)
            continue

        inventory_units = dated_snapshots["stock_quantity"].to_numpy()
        inventory_values = dated_snapshots["inventory_value"].to_numpy()
        average_units = float(
            (
                (inventory_units[:-1] + inventory_units[1:])
                / 2
                * elapsed_days
            ).sum()
            / elapsed_days.sum()
        )
        average_value = float(
            (
                (inventory_values[:-1] + inventory_values[1:])
                / 2
                * elapsed_days
            ).sum()
            / elapsed_days.sum()
        )
        base["average_inventory_units"] = average_units
        base["average_inventory_value"] = average_value

        if not sales_cost_complete:
            records.append(base)
            continue

        if average_value <= 0:
            base["data_status"] = (
                "Average inventory value must be positive for cost-based "
                "turnover."
            )
            records.append(base)
            continue

        turnover = cogs / average_value
        base["turnover_ratio"] = turnover
        base["days_inventory_outstanding"] = (
            period_days / turnover if turnover > 0 else np.nan
        )
        base["annualized_turnover"] = turnover * 365 / period_days
        base["data_status"] = "Calculated from uploaded costed data."
        records.append(base)

    result = pd.DataFrame.from_records(records, columns=TURNOVER_COLUMNS)
    eligible = result["annualized_turnover"].dropna()
    if not eligible.empty:
        median_turnover = eligible.median()
        result.loc[
            result["annualized_turnover"].ge(median_turnover)
            & result["annualized_turnover"].gt(0),
            "movement_classification",
        ] = "Fast-moving (at/above portfolio median)"
        result.loc[
            result["annualized_turnover"].lt(median_turnover)
            | result["annualized_turnover"].eq(0),
            "movement_classification",
        ] = "Slow-moving (below portfolio median)"
    return (
        result.sort_values(
            "annualized_turnover",
            ascending=False,
            na_position="last",
        ),
        (
            "COGS uses transaction quantity × transaction unit_cost. "
            "Average inventory value is the time-weighted trapezoidal average "
            "of dated stock_quantity × snapshot unit_cost. Turnover is COGS "
            "÷ average inventory value; DIO is period days ÷ turnover. "
            "Fast/slow labels are relative to the eligible portfolio median."
            + (
                f" Ignored {duplicate_snapshot_count} exact duplicate "
                "snapshot row(s) and "
                f"{duplicate_sales_count} exact duplicate sales row(s)."
                if duplicate_snapshot_count or duplicate_sales_count
                else ""
            )
        ),
    )


def get_reorder_recommendations(
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    """Return one actionable suggestion per product; never creates an order."""
    if analysis.empty:
        return analysis.copy()
    recommendations = analysis.loc[
        analysis["recommended_order_quantity"].gt(0)
    ].copy()
    recommendations = recommendations.drop_duplicates(
        subset="product_id",
        keep="first",
    )
    urgency_order = pd.Categorical(
        recommendations["reorder_urgency"],
        categories=["Critical", "High", "Moderate"],
        ordered=True,
    )
    recommendations = recommendations.assign(
        _urgency_sort=urgency_order
    ).sort_values(
        ["_urgency_sort", "recommended_order_quantity"],
        ascending=[True, False],
        kind="stable",
    )
    return recommendations.drop(columns="_urgency_sort")
