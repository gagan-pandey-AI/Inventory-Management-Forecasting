import unittest

import pandas as pd

from dashboard.inventory_intelligence import (
    build_inventory_analysis,
    calculate_inventory_turnover,
    get_reorder_recommendations,
)


class InventoryAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.inventory = pd.DataFrame(
            [
                {
                    "product_id": 1,
                    "product_name": "Critical item",
                    "stock_quantity": 0,
                    "reorder_level": 5,
                    "lead_time_days": 4,
                    "supplier_name": None,
                    "pending_order_quantity": 0,
                },
                {
                    "product_id": 2,
                    "product_name": "Healthy item",
                    "stock_quantity": 120,
                    "reorder_level": 10,
                    "lead_time_days": 5,
                    "supplier_name": "Supplier A",
                    "pending_order_quantity": 0,
                },
                {
                    "product_id": 3,
                    "product_name": "Overstock item",
                    "stock_quantity": 400,
                    "reorder_level": 10,
                    "lead_time_days": 5,
                    "supplier_name": "Supplier B",
                    "pending_order_quantity": 0,
                },
                {
                    "product_id": 4,
                    "product_name": "No demand",
                    "stock_quantity": 25,
                    "reorder_level": 0,
                    "lead_time_days": 5,
                    "supplier_name": None,
                    "pending_order_quantity": 0,
                },
            ]
        )
        self.forecast = pd.DataFrame(
            [
                {
                    "product_id": product_id,
                    "product_name": name,
                    "forecast_date": "2026-11-02",
                    "predicted_demand": demand,
                }
                for product_id, name, demand in (
                    (1, "Critical item", 2),
                    (2, "Healthy item", 2),
                    (3, "Overstock item", 2),
                    (4, "No demand", 0),
                )
            ]
        )

    def test_classifications_and_excess_use_explicit_horizons(self):
        analysis = build_inventory_analysis(
            self.inventory,
            self.forecast,
            as_of_date="2026-11-01",
            safety_buffer_days=7,
            overstock_target_days=60,
        ).set_index("product_id")

        self.assertEqual(analysis.loc[1, "classification"], "Critical Low Stock")
        self.assertEqual(analysis.loc[2, "classification"], "Healthy Stock")
        self.assertEqual(analysis.loc[3, "classification"], "Overstock")
        self.assertEqual(analysis.loc[3, "excess_quantity"], 280)
        self.assertEqual(analysis.loc[4, "classification"], "Overstock")
        self.assertEqual(analysis.loc[4, "excess_quantity"], 25)
        self.assertTrue(pd.isna(analysis.loc[4, "days_inventory_remaining"]))

    def test_reorder_quantity_accounts_for_pending_orders_and_missing_supplier(self):
        analysis = build_inventory_analysis(
            self.inventory,
            self.forecast,
            as_of_date="2026-11-01",
            safety_buffer_days=7,
            reorder_target_days=30,
        )
        recommendations = get_reorder_recommendations(analysis).set_index(
            "product_id"
        )

        self.assertIn(1, recommendations.index)
        self.assertEqual(
            recommendations.loc[1, "recommended_order_quantity"],
            82,
        )
        self.assertEqual(
            recommendations.loc[1, "supplier_display"],
            "Supplier unavailable",
        )

        covered_inventory = self.inventory.copy()
        covered_inventory.loc[
            covered_inventory["product_id"] == 1,
            "pending_order_quantity",
        ] = 100
        covered = build_inventory_analysis(
            covered_inventory,
            self.forecast,
            as_of_date="2026-11-01",
        )
        self.assertNotIn(
            1,
            get_reorder_recommendations(covered)["product_id"].tolist(),
        )

    def test_missing_demand_and_duplicate_inventory_do_not_recommend(self):
        inventory = pd.DataFrame(
            [
                {
                    "product_id": 5,
                    "product_name": "Missing demand",
                    "stock_quantity": 0,
                    "reorder_level": 10,
                    "lead_time_days": 4,
                },
                {
                    "product_id": 6,
                    "product_name": "Duplicate",
                    "stock_quantity": 0,
                    "reorder_level": 10,
                    "lead_time_days": 4,
                },
                {
                    "product_id": 6,
                    "product_name": "Duplicate",
                    "stock_quantity": 1,
                    "reorder_level": 10,
                    "lead_time_days": 4,
                },
            ]
        )
        analysis = build_inventory_analysis(
            inventory,
            pd.DataFrame(),
            as_of_date="2026-11-01",
        )

        self.assertEqual(
            analysis.loc[analysis["product_id"] == 5, "classification"].iloc[0],
            "Critical Low Stock",
        )
        self.assertEqual(
            analysis.loc[analysis["product_id"] == 6, "classification"].iloc[0],
            "Data unavailable",
        )
        self.assertTrue(get_reorder_recommendations(analysis).empty)

    def test_zero_stock_is_critical_even_without_demand_or_reorder_level(self):
        inventory = pd.DataFrame(
            [
                {
                    "product_id": 7,
                    "product_name": "No stock",
                    "stock_quantity": 0,
                    "reorder_level": None,
                    "lead_time_days": None,
                }
            ]
        )
        analysis = build_inventory_analysis(
            inventory,
            pd.DataFrame(),
            as_of_date="2026-11-01",
        )

        self.assertEqual(
            analysis.loc[0, "classification"],
            "Critical Low Stock",
        )
        self.assertTrue(pd.isna(analysis.loc[0, "days_inventory_remaining"]))

    def test_missing_current_stock_remains_unavailable(self):
        inventory = pd.DataFrame(
            [
                {
                    "product_id": 8,
                    "product_name": "Missing stock",
                    "stock_quantity": None,
                    "reorder_level": 5,
                    "lead_time_days": 4,
                }
            ]
        )
        forecast = pd.DataFrame(
            {
                "product_id": [8],
                "forecast_date": ["2026-11-02"],
                "predicted_demand": [2],
            }
        )
        analysis = build_inventory_analysis(
            inventory,
            forecast,
            as_of_date="2026-11-01",
        )

        self.assertEqual(analysis.loc[0, "classification"], "Data unavailable")
        self.assertTrue(get_reorder_recommendations(analysis).empty)

    def test_historical_sales_fallback_is_explicit(self):
        sales = pd.DataFrame(
            {
                "product_id": [2, 2],
                "sale_date": ["2026-10-31", "2026-11-01"],
                "quantity_sold": [45, 45],
            }
        )
        analysis = build_inventory_analysis(
            self.inventory.iloc[[1]],
            pd.DataFrame(),
            sales,
            as_of_date="2026-11-01",
            historical_lookback_days=90,
        )

        self.assertEqual(
            analysis.loc[0, "demand_source"],
            "Historical sales fallback",
        )
        self.assertAlmostEqual(
            analysis.loc[0, "average_daily_demand"],
            1,
        )


class InventoryTurnoverTests(unittest.TestCase):
    def setUp(self):
        self.snapshots = pd.DataFrame(
            {
                "product_id": [1, 1],
                "product_name": ["Test product", "Test product"],
                "snapshot_date": ["2026-02-01", "2026-03-02"],
                "stock_quantity": [100, 100],
                "unit_cost": [10, 10],
            }
        )
        self.sales = pd.DataFrame(
            {
                "product_id": [1],
                "sale_date": ["2026-02-20"],
                "quantity_sold": [10],
                "unit_cost": [10],
            }
        )

    def test_cost_based_turnover_and_dio(self):
        result, explanation = calculate_inventory_turnover(
            self.snapshots,
            self.sales,
            period_days=30,
            as_of_date="2026-03-02",
        )
        row = result.iloc[0]

        self.assertAlmostEqual(row["average_inventory_units"], 100)
        self.assertAlmostEqual(row["average_inventory_value"], 1000)
        self.assertAlmostEqual(row["cogs"], 100)
        self.assertAlmostEqual(row["turnover_ratio"], 0.1)
        self.assertAlmostEqual(row["days_inventory_outstanding"], 300)
        self.assertIn("time-weighted", explanation)

    def test_missing_snapshots_or_costs_are_reported_not_fabricated(self):
        empty_result, empty_note = calculate_inventory_turnover(
            None,
            self.sales,
        )
        self.assertTrue(empty_result.empty)
        self.assertIn("snapshots", empty_note)

        missing_cost, missing_cost_note = calculate_inventory_turnover(
            self.snapshots.drop(columns="unit_cost"),
            self.sales,
        )
        self.assertTrue(missing_cost.empty)
        self.assertIn("unit_cost", missing_cost_note)

        missing_row_cost = self.sales.copy()
        missing_row_cost.loc[0, "unit_cost"] = None
        result, _ = calculate_inventory_turnover(
            self.snapshots,
            missing_row_cost,
            period_days=30,
            as_of_date="2026-03-02",
        )
        self.assertTrue(pd.isna(result.loc[0, "cogs"]))
        self.assertIn("per-sale unit cost", result.loc[0, "data_status"])

        missing_snapshot_cost = self.snapshots.copy()
        missing_snapshot_cost.loc[0, "unit_cost"] = None
        result, _ = calculate_inventory_turnover(
            missing_snapshot_cost,
            self.sales,
            period_days=30,
            as_of_date="2026-03-02",
        )
        self.assertEqual(result.loc[0, "cogs"], 100)
        self.assertTrue(pd.isna(result.loc[0, "average_inventory_value"]))

    def test_one_snapshot_and_zero_inventory_value_are_not_ratios(self):
        result, _ = calculate_inventory_turnover(
            self.snapshots.iloc[[0]],
            self.sales,
            period_days=30,
            as_of_date="2026-03-02",
        )
        self.assertAlmostEqual(result.loc[0, "cogs"], 100)
        self.assertEqual(result.loc[0, "movement_classification"], "Unavailable")

        zero_value = self.snapshots.assign(
            stock_quantity=0,
            unit_cost=0,
        )
        result, _ = calculate_inventory_turnover(
            zero_value,
            self.sales,
            period_days=30,
            as_of_date="2026-03-02",
        )
        self.assertTrue(pd.isna(result.loc[0, "turnover_ratio"]))

    def test_zero_sales_has_zero_cogs_and_undefined_dio(self):
        sales_without_transactions = pd.DataFrame(
            columns=["product_id", "sale_date", "quantity_sold", "unit_cost"]
        )
        result, _ = calculate_inventory_turnover(
            self.snapshots,
            sales_without_transactions,
            period_days=30,
            as_of_date="2026-03-02",
        )

        self.assertEqual(result.loc[0, "cogs"], 0)
        self.assertEqual(result.loc[0, "turnover_ratio"], 0)
        self.assertTrue(pd.isna(result.loc[0, "days_inventory_outstanding"]))
        self.assertIn("Slow-moving", result.loc[0, "movement_classification"])


if __name__ == "__main__":
    unittest.main()
