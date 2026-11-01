import streamlit as st
import pandas as pd
import plotly.express as px
import mysql.connector
from pathlib import Path

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="INVENTRA | Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"


def connect_to_database(**config):
    return mysql.connector.connect(**config, connect_timeout=5)


# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    monthly = pd.read_csv(REPORTS_DIR / "monthly_sales.csv")
    products = pd.read_csv(REPORTS_DIR / "product_sales_analysis.csv")
    forecast = pd.read_csv(REPORTS_DIR / "forecast_summary.csv")
    return monthly, products, forecast

try:
    monthly, products, forecast = load_data()
except Exception as e:
    st.error(f"Unable to load project data: {e}")
    st.stop()

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #080F1E;
    color: #E5E7EB;
}

[data-testid="stSidebar"] {
    background: #0D1729;
    border-right: 1px solid #1E293B;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

.hero {
    background: linear-gradient(120deg, #132C4F, #123D55, #075B69);
    padding: 32px;
    border-radius: 18px;
    border: 1px solid #24536D;
    margin-bottom: 28px;
}

.hero h1 {
    color: #FFFFFF;
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #B9D5E5;
    font-size: 14px;
    margin-bottom: 0;
}

.section-title {
    color: #F1F5F9;
    font-size: 21px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 16px;
}

.kpi {
    background: linear-gradient(145deg, #14243B, #101B2D);
    border: 1px solid #263A55;
    border-radius: 15px;
    padding: 22px;
    min-height: 135px;
}

.kpi-label {
    color: #94A3B8;
    font-size: 13px;
    font-weight: 600;
}

.kpi-value {
    color: #F8FAFC;
    font-size: 27px;
    font-weight: 800;
    margin-top: 13px;
}

.kpi-icon {
    font-size: 22px;
    float: right;
}

.info-box {
    background: #101E32;
    border: 1px solid #263A55;
    padding: 18px;
    border-radius: 13px;
    color: #CBD5E1;
}

.footer {
    text-align: center;
    color: #64748B;
    font-size: 12px;
    padding-top: 30px;
}

div[data-testid="stPlotlyChart"] {
    background: #101B2D;
    border: 1px solid #263A55;
    border-radius: 14px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## 📊 INVENTRA")
    st.caption("INTELLIGENT INVENTORY ANALYTICS")

    st.divider()

    page = st.radio(
    "NAVIGATION",
    [
        "Overview",
        "Sales Intelligence",
        "AI Forecast",
        "Inventory Monitoring",
        "Supplier Analytics",
        "Forecast Performance",
        "Project Guide"
    ],
    label_visibility="visible"
)

    st.divider()

    st.markdown("### PROJECT INFORMATION")
    st.caption("Inventory Management")
    st.caption("Demand Forecasting")
    st.caption("Machine Learning Analytics")

    st.divider()
    st.caption("Developed Team Power Rangers")

# ---------------- KPI CALCULATIONS ----------------

total_revenue = monthly["revenue"].sum()
total_units = monthly["total_quantity"].sum()
total_products = products["product_name"].nunique()
total_forecast = forecast["predicted_demand"].sum()

# ---------------- HERO HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>INVENTORY INTELLIGENCE</h1>
    <p>
        Smart Inventory Management & AI-Based Demand Forecasting Platform
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- KPI CARDS ----------------

def show_kpi(label, value, icon):
    st.markdown(f"""
    <div class="kpi">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

if page == "Overview":

    st.markdown('<div class="section-title">BUSINESS PERFORMANCE</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        show_kpi("TOTAL REVENUE", f"₹{total_revenue:,.0f}", "💰")

    with c2:
        show_kpi("UNITS SOLD", f"{total_units:,}", "📦")

    with c3:
        show_kpi("PRODUCTS ANALYSED", f"{total_products}", "🛒")

    with c4:
        show_kpi("NOVEMBER FORECAST", f"{total_forecast:,}", "🤖")

    st.markdown('<div class="section-title">REVENUE PERFORMANCE</div>',
                unsafe_allow_html=True)

    fig = px.area(
        monthly,
        x="sales_month",
        y="revenue",
        markers=True
    )

    fig.update_traces(
        line=dict(color="#22D3EE", width=3),
        fillcolor="rgba(34,211,238,0.12)"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Month",
        yaxis_title="Revenue (₹)",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">TOP PRODUCTS</div>',
                    unsafe_allow_html=True)

        top = products.nlargest(5, "revenue").sort_values("revenue")

        fig2 = px.bar(
            top,
            x="revenue",
            y="product_name",
            orientation="h",
            text_auto=".2s"
        )

        fig2.update_traces(marker_color="#38BDF8")

        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1",
            margin=dict(l=10, r=10, t=15, b=15),
            xaxis_title="Revenue (₹)",
            yaxis_title=""
        )

        st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">DEMAND FORECAST</div>',
                    unsafe_allow_html=True)

        fig3 = px.pie(
            forecast,
            names="product_name",
            values="predicted_demand",
            hole=0.55
        )

        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1",
            margin=dict(l=10, r=10, t=15, b=15),
            legend_title="Products"
        )

        st.plotly_chart(fig3, use_container_width=True)

elif page == "Sales Intelligence":

    st.markdown('<div class="section-title">SALES ANALYTICS</div>',
                unsafe_allow_html=True)

    st.dataframe(
        products.sort_values("revenue", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        products.sort_values("revenue", ascending=True),
        x="revenue",
        y="product_name",
        orientation="h",
        color="revenue",
        color_continuous_scale="Teal"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        xaxis_title="Revenue (₹)",
        yaxis_title="Product"
    )

    st.plotly_chart(fig, use_container_width=True)

elif page == "AI Forecast":

    st.markdown('<div class="section-title">AI DEMAND FORECASTING</div>',
                unsafe_allow_html=True)

    st.info(
        "Machine Learning-based demand predictions for November 2026."
    )

    st.metric("Total Predicted Demand", f"{total_forecast:,} Units")

    st.markdown("### 🤖 MACHINE LEARNING MODEL PERFORMANCE")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Random Forest MAE",
        "0.955",
        help="Mean Absolute Error of the Random Forest model."
    )

    m2.metric(
        "7-Day Baseline MAE",
        "1.193",
        help="Mean Absolute Error of the baseline model."
    )

    improvement = ((1.193 - 0.955) / 1.193) * 100

    m3.metric(
        "Error Reduction",
        f"{improvement:.2f}%",
        delta="Better than baseline"
    )

    comparison = pd.DataFrame({
        "Model": ["Random Forest", "7-Day Baseline"],
        "MAE": [0.955, 1.193]
    })

    fig_model = px.bar(
        comparison,
        x="Model",
        y="MAE",
        color="Model",
        text_auto=".3f",
        title="Model Performance Comparison"
    )

    fig_model.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        showlegend=False
    )

    st.plotly_chart(fig_model, use_container_width=True)

    st.download_button(
        label="📥 Download Forecast CSV",
        data=forecast.to_csv(index=False).encode("utf-8"),
        file_name="november_2026_demand_forecast.csv",
        mime="text/csv"
    )

    fig = px.bar(
        forecast.sort_values("predicted_demand", ascending=True),
        x="predicted_demand",
        y="product_name",
        orientation="h",
        text_auto=True,
        color="predicted_demand",
        color_continuous_scale="Teal"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1",
        xaxis_title="Predicted Units",
        yaxis_title="Product"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### FORECAST DETAILS")

    st.dataframe(
        forecast.sort_values("predicted_demand", ascending=False),
        use_container_width=True,
        hide_index=True
    )

elif page == "Inventory Monitoring":

    st.markdown(
        '<div class="section-title">LIVE INVENTORY MONITORING</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Sales and forecasting reports are available without MySQL. "
        "Inventory monitoring requires a reachable MySQL server; "
        "127.0.0.1 refers to this app's host, not your computer."
    )

    with st.form("mysql_connection"):
        db_host = st.text_input("Host")
        db_user = st.text_input("Username")
        db_password = st.text_input("MySQL Password", type="password")

        connect = st.form_submit_button("Connect to MySQL")

    if connect:
        connection = None
        st.session_state.pop("inventory_data", None)
        try:
            connection = connect_to_database(
                host=db_host,
                user=db_user,
                password=db_password,
                database="inventoryforecastdb",
                port=3306
            )

            query = """
            SELECT
                p.product_name,
                p.unit_price,
                i.stock_quantity,
                i.reorder_level,
                i.lead_time_days,
                i.last_updated
            FROM inventory i
            JOIN products p
                ON i.product_id = p.product_id
            ORDER BY i.stock_quantity ASC
            """

            inventory_data = pd.read_sql(query, connection)
            connection.close()

            st.session_state["inventory_data"] = inventory_data
            st.success("MySQL connected successfully!")

        except mysql.connector.Error:
            st.error(
                "Unable to load inventory data from MySQL. Check that "
                "the server is reachable and the credentials, database, "
                "and schema are correct. Connection details are hidden."
            )
        finally:
            if connection is not None and connection.is_connected():
                connection.close()

    if "inventory_data" in st.session_state:

        df = st.session_state["inventory_data"]

        low_stock = df[
            df["stock_quantity"] <= df["reorder_level"]
        ]

        out_of_stock = df[
            df["stock_quantity"] == 0
        ]

        reorder_units = (
            low_stock["reorder_level"] -
            low_stock["stock_quantity"]
        ).clip(lower=0).sum()

                # SMART REORDER RECOMMENDATION

        reorder_data = low_stock.copy()

        reorder_data["suggested_order"] = (
            reorder_data["reorder_level"] -
            reorder_data["stock_quantity"]
        ).clip(lower=0)

        def get_priority(row):
            if row["stock_quantity"] == 0:
                return "CRITICAL"
            elif row["lead_time_days"] >= 10:
                return "HIGH"
            else:
                return "MEDIUM"

        reorder_data["priority"] = reorder_data.apply(
            get_priority,
            axis=1
        )

        st.markdown("### 🧠 SMART REORDER RECOMMENDATIONS")

        st.dataframe(
            reorder_data[
                [
                    "product_name",
                    "stock_quantity",
                    "reorder_level",
                    "suggested_order",
                    "lead_time_days",
                    "priority"
                ]
            ].sort_values(
                "suggested_order",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download Reorder Recommendations",
            data=reorder_data.to_csv(index=False).encode("utf-8"),
            file_name="smart_reorder_recommendations.csv",
            mime="text/csv"
        )

        a, b, c, d = st.columns(4)

        a.metric("Total Stock", int(df["stock_quantity"].sum()))
        b.metric("Low Stock Products", len(low_stock))
        c.metric("Out of Stock", len(out_of_stock))
        d.metric("Reorder Quantity", int(reorder_units))

        st.markdown("### Stock Distribution")

        fig_stock = px.bar(
            df,
            x="product_name",
            y="stock_quantity",
            color="stock_quantity",
            color_continuous_scale="Teal",
            title="Product-wise Available Stock"
        )

        fig_stock.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1"
        )

        st.plotly_chart(
            fig_stock,
            use_container_width=True
        )

        st.markdown("### ⚠️ Low Stock Alerts")

        if not low_stock.empty:
            st.dataframe(
                low_stock,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No low-stock products.")

        st.markdown("### Complete Inventory")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


elif page == "Supplier Analytics":

    st.markdown(
        '<div class="section-title">SUPPLIER PERFORMANCE ANALYTICS</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Supplier analytics requires a reachable MySQL server. "
        "On Streamlit Community Cloud, use an external database host; "
        "localhost refers to the app container."
    )

    with st.form("supplier_connection"):

        db_host = st.text_input(
            "MySQL Host"
        )

        db_user = st.text_input(
            "MySQL Username"
        )

        db_password = st.text_input(
            "MySQL Password",
            type="password"
        )

        connect = st.form_submit_button(
            "Connect to Supplier Database"
        )

    if connect:
        connection = None
        cursor = None
        for key in ("supplier_data", "status_data", "supplier_kpis"):
            st.session_state.pop(key, None)
        try:

            connection = connect_to_database(
                host=db_host,
                user=db_user,
                password=db_password,
                database="inventoryforecastdb",
                port=3306
            )

            cursor = connection.cursor(dictionary=True)

            # SUPPLIER PERFORMANCE

            supplier_query = """
            SELECT
                s.supplier_name,
                s.city,
                COUNT(DISTINCT po.order_id) AS total_orders,
                COALESCE(SUM(poi.quantity_ordered), 0)
                    AS total_quantity,
                COALESCE(
                    ROUND(
                        SUM(
                            poi.quantity_ordered * poi.unit_cost
                        ), 2
                    ), 0
                ) AS total_purchase_cost

            FROM suppliers s

            LEFT JOIN purchase_orders po
                ON s.supplier_id = po.supplier_id

            LEFT JOIN purchase_order_items poi
                ON po.order_id = poi.order_id

            GROUP BY
                s.supplier_id,
                s.supplier_name,
                s.city

            ORDER BY total_purchase_cost DESC
            """

            cursor.execute(supplier_query)

            supplier_data = pd.DataFrame(
                cursor.fetchall()
            )

            # ORDER STATUS

            status_query = """
            SELECT
                order_status,
                COUNT(*) AS total_orders
            FROM purchase_orders
            GROUP BY order_status
            """

            cursor.execute(status_query)

            status_data = pd.DataFrame(
                cursor.fetchall()
            )

            # PURCHASE KPIs

            kpi_query = """
            SELECT
                COUNT(DISTINCT po.order_id) AS total_orders,

                COALESCE(
                    SUM(poi.quantity_ordered), 0
                ) AS total_units,

                COALESCE(
                    ROUND(
                        SUM(
                            poi.quantity_ordered * poi.unit_cost
                        ), 2
                    ), 0
                ) AS purchase_value,

                COUNT(DISTINCT
                    CASE
                        WHEN po.order_status = 'Pending'
                        THEN po.order_id
                    END
                ) AS pending_orders

            FROM purchase_orders po

            JOIN purchase_order_items poi
                ON po.order_id = poi.order_id
            """

            cursor.execute(kpi_query)

            kpi_data = cursor.fetchone()

            st.session_state["supplier_data"] = supplier_data
            st.session_state["status_data"] = status_data
            st.session_state["supplier_kpis"] = kpi_data

            st.success("Supplier database connected successfully!")

        except mysql.connector.Error:
            st.error(
                "Unable to load supplier analytics from MySQL. Check "
                "that the server is reachable and the credentials, "
                "database, and schema are correct. Connection details "
                "are hidden."
            )
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None and connection.is_connected():
                connection.close()

    # DISPLAY ANALYTICS

    if "supplier_data" in st.session_state:

        df = st.session_state["supplier_data"]

        status_df = st.session_state["status_data"]

        kpis = st.session_state["supplier_kpis"]

        st.markdown("### PURCHASE OVERVIEW")

        a, b, c, d = st.columns(4)

        a.metric(
            "Total Orders",
            int(kpis["total_orders"])
        )

        b.metric(
            "Units Ordered",
            int(kpis["total_units"])
        )

        c.metric(
            "Purchase Value",
            f"₹{kpis['purchase_value']:,.0f}"
        )

        d.metric(
            "Pending Orders",
            int(kpis["pending_orders"])
        )

        # SUPPLIER COST CHART

        st.markdown("### Supplier-wise Purchase Cost")

        fig_cost = px.bar(
            df,
            x="supplier_name",
            y="total_purchase_cost",
            color="supplier_name",
            title="Purchase Cost by Supplier"
        )

        fig_cost.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1",
            showlegend=False
        )

        st.plotly_chart(
            fig_cost,
            use_container_width=True
        )

        # QUANTITY CHART

        st.markdown("### Supplier-wise Order Quantity")

        fig_quantity = px.bar(
            df,
            x="supplier_name",
            y="total_quantity",
            color="total_quantity",
            title="Total Quantity Ordered"
        )

        fig_quantity.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1"
        )

        st.plotly_chart(
            fig_quantity,
            use_container_width=True
        )

        # ORDER STATUS

        st.markdown("### Purchase Order Status")

        if not status_df.empty:

            fig_status = px.pie(
                status_df,
                names="order_status",
                values="total_orders",
                title="Delivered vs Pending Orders",
                hole=0.45
            )

            st.plotly_chart(
                fig_status,
                use_container_width=True
            )

        # SUPPLIER TABLE

        st.markdown("### Complete Supplier Performance")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        # DOWNLOAD REPORT

        st.download_button(
            "Download Supplier Analytics CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="supplier_performance_report.csv",
            mime="text/csv"
        )



# ==========================================
# FORECAST PERFORMANCE
# ==========================================

elif page == "Forecast Performance":

    st.markdown(
        '<div class="section-title">FORECAST PERFORMANCE ANALYTICS</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Historical model evaluation using actual sales "
        "and predictions from the held-out test period."
    )

    try:

        evaluation = pd.read_csv(
            REPORTS_DIR / "forecast_evaluation.csv"
        )

        product_accuracy = pd.read_csv(
            REPORTS_DIR / "product_forecast_accuracy.csv"
        )

        model_performance = pd.read_csv(
            REPORTS_DIR / "model_performance.csv"
        )

        # MODEL METRICS

        rf_mae = float(
            model_performance.loc[
                model_performance["model"] == "Random Forest",
                "MAE"
            ].iloc[0]
        )

        baseline_mae = float(
            model_performance.loc[
                model_performance["model"] == "7-Day Baseline",
                "MAE"
            ].iloc[0]
        )

        improvement = (
            (baseline_mae - rf_mae) / baseline_mae * 100
            if baseline_mae != 0 else 0
        )

        st.markdown("### MODEL PERFORMANCE")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Random Forest MAE",
            f"{rf_mae:.3f}"
        )

        c2.metric(
            "7-Day Baseline MAE",
            f"{baseline_mae:.3f}"
        )

        c3.metric(
            "Error Reduction",
            f"{improvement:.2f}%",
            delta="Compared with baseline"
        )

        st.caption(
            "MAE (Mean Absolute Error): Lower values indicate "
            "smaller average prediction errors."
        )

        # MODEL COMPARISON

        st.markdown("### MODEL COMPARISON")

        fig_model = px.bar(
            model_performance,
            x="model",
            y="MAE",
            color="model",
            text_auto=".3f",
            title="Random Forest vs 7-Day Baseline"
        )

        fig_model.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1",
            showlegend=False,
            xaxis_title="Model",
            yaxis_title="Mean Absolute Error"
        )

        st.plotly_chart(
            fig_model,
            use_container_width=True
        )

        # ACTUAL VS PREDICTED

        st.markdown("### ACTUAL VS PREDICTED SALES")

        evaluation["evaluation_date"] = pd.to_datetime(
            evaluation["evaluation_date"]
        )

        daily_comparison = (
            evaluation.groupby("evaluation_date")[
                [
                    "actual_sales",
                    "random_forest_prediction",
                    "baseline_prediction"
                ]
            ]
            .sum()
            .reset_index()
        )

        comparison_long = daily_comparison.melt(
            id_vars="evaluation_date",
            value_vars=[
                "actual_sales",
                "random_forest_prediction",
                "baseline_prediction"
            ],
            var_name="Sales Type",
            value_name="Units"
        )

        fig_actual = px.line(
            comparison_long,
            x="evaluation_date",
            y="Units",
            color="Sales Type",
            markers=True,
            title="Daily Actual vs Predicted Sales"
        )

        fig_actual.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1",
            xaxis_title="Date",
            yaxis_title="Units Sold",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_actual,
            use_container_width=True
        )

        # PRODUCT-WISE ACCURACY

        st.markdown("### PRODUCT-WISE FORECAST ACCURACY")

        st.dataframe(
            product_accuracy.sort_values(
                "random_forest_mae"
            ),
            use_container_width=True,
            hide_index=True
        )

        fig_product = px.bar(
            product_accuracy.sort_values(
                "random_forest_mae",
                ascending=True
            ),
            x="random_forest_mae",
            y="product_name",
            orientation="h",
            color="random_forest_mae",
            color_continuous_scale="Teal",
            title="Random Forest MAE by Product"
        )

        fig_product.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1",
            xaxis_title="MAE",
            yaxis_title="Product"
        )

        st.plotly_chart(
            fig_product,
            use_container_width=True
        )

        # DOWNLOAD REPORTS

        st.markdown("### DOWNLOAD EVALUATION REPORTS")

        d1, d2, d3 = st.columns(3)

        d1.download_button(
            "Download Evaluation CSV",
            data=evaluation.to_csv(index=False).encode("utf-8"),
            file_name="forecast_evaluation.csv",
            mime="text/csv"
        )

        d2.download_button(
            "Download Product Accuracy",
            data=product_accuracy.to_csv(index=False).encode("utf-8"),
            file_name="product_forecast_accuracy.csv",
            mime="text/csv"
        )

        d3.download_button(
            "Download Model Performance",
            data=model_performance.to_csv(index=False).encode("utf-8"),
            file_name="model_performance.csv",
            mime="text/csv"
        )

    except FileNotFoundError:

        st.error(
            "Evaluation reports not found. "
            "Please run src/demand_forecasting.py first."
        )

    except Exception as e:

        st.error(f"Unable to load forecast performance: {e}")


elif page == "Project Guide":

    st.title("📘 INVENTRA PROJECT GUIDE")

    st.markdown("""
    ### Inventory Management & Demand Forecasting System

    Welcome to INVENTRA!

    INVENTRA is an intelligent inventory management and
    demand forecasting platform developed using Python,
    MySQL, Machine Learning and Streamlit.

    This guide explains how the complete system works.
    """)

    st.divider()

    # 1. PROJECT OVERVIEW
    st.header("1. Project Overview")

    st.markdown("""
    INVENTRA helps businesses monitor inventory, analyse
    sales, manage suppliers and forecast future product demand.

    **Main objectives:**
    - Monitor available inventory.
    - Identify low-stock products.
    - Analyse historical sales.
    - Understand product performance.
    - Forecast future product demand.
    - Support data-driven purchasing decisions.
    """)

    # 2. TECHNOLOGY STACK
    st.header("2. Technologies Used")

    tech = {
        "Technology": [
            "Python",
            "MySQL",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Streamlit",
            "Plotly",
            "Matplotlib"
        ],
        "Purpose": [
            "Main programming language",
            "Database management",
            "Data analysis",
            "Numerical operations",
            "Machine Learning",
            "Interactive dashboard",
            "Interactive charts",
            "Data visualisation"
        ]
    }

    st.dataframe(tech, use_container_width=True)

    # 3. SYSTEM WORKFLOW
    st.header("3. How INVENTRA Works")

    st.markdown("""
    **Step 1 — Database**

    MySQL stores products, suppliers, inventory, sales
    and purchase-order information.

    **Step 2 — Data Generation**

    Python generates demonstration sales records.

    **Step 3 — Data Analysis**

    Pandas analyses sales, revenue and product performance.

    **Step 4 — Feature Engineering**

    Historical sales are converted into useful features
    such as lag values and rolling averages.

    **Step 5 — Machine Learning**

    Random Forest predicts future product demand.

    **Step 6 — Model Evaluation**

    Predictions are compared with actual sales using MAE.

    **Step 7 — Dashboard**

    Streamlit displays business insights, inventory
    information and forecasting results.
    """)

    st.info(
        "MySQL → Python → Data Analysis → Machine Learning "
        "→ Evaluation → Streamlit Dashboard"
    )

    # 4. DATABASE STRUCTURE
    st.header("4. Database Structure")

    tables = {
        "Database Table": [
            "categories",
            "suppliers",
            "products",
            "inventory",
            "sales",
            "purchase_orders",
            "purchase_order_items"
        ],
        "Purpose": [
            "Stores product categories",
            "Stores supplier information",
            "Stores product details",
            "Maintains stock information",
            "Stores sales transactions",
            "Stores purchase order details",
            "Stores products within purchase orders"
        ]
    }

    st.dataframe(tables, use_container_width=True)

    # 5. PYTHON FILES
    st.header("5. Python Modules Explained")

    with st.expander("generate_sales.py"):
        st.write("""
        Generates simulated historical sales data
        and stores the records in MySQL.
        """)

    with st.expander("analyze_sales.py"):
        st.write("""
        Analyses sales data, calculates monthly revenue,
        identifies top-selling products and generates
        charts and CSV reports.
        """)

    with st.expander("demand_forecasting.py"):
        st.write("""
        Loads sales history, creates lag and rolling
        features, trains the Random Forest model,
        evaluates predictions and generates future
        demand forecasts.
        """)

    with st.expander("dashboard/app.py"):
        st.write("""
        Runs the Streamlit dashboard and displays
        inventory, supplier, sales and forecasting analytics.
        """)

    # 6. MACHINE LEARNING
    st.header("6. Demand Forecasting Model")

    st.markdown("""
    INVENTRA uses the Random Forest Regressor algorithm.

    The model learns patterns from historical product sales.

    **Important features:**
    - Previous day sales (lag_1)
    - Previous 7-day sales (lag_7)
    - Previous 14-day sales (lag_14)
    - Previous 28-day sales (lag_28)
    - 7-day rolling average
    - 28-day rolling average
    - Day of week and month
    """)

    # 7. MODEL PERFORMANCE
    st.header("7. Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Random Forest MAE", "0.955")
    col2.metric("Baseline MAE", "1.193")
    col3.metric("Error Reduction", "19.97%")

    st.markdown("""
    **What is MAE?**

    Mean Absolute Error measures the average absolute
    difference between actual and predicted sales.

    A lower MAE indicates better predictions.

    In this evaluation, Random Forest performed better
    than the 7-day baseline on the held-out test data.
    """)

    st.caption(
        "These evaluation results are based on simulated "
        "project data and do not guarantee real-world accuracy."
    )

    # 8. DASHBOARD MODULES
    st.header("8. Dashboard Navigation Guide")

    modules = {
        "Dashboard Section": [
            "Overview",
            "Sales Intelligence",
            "AI Forecast",
            "Inventory Monitoring",
            "Supplier Analytics",
            "Forecast Performance",
            "Project Guide"
        ],
        "What It Does": [
            "Displays overall business KPIs",
            "Shows sales and revenue analytics",
            "Displays future demand predictions",
            "Monitors stock and reorder levels",
            "Analyses suppliers and purchase orders",
            "Evaluates machine learning performance",
            "Explains the complete project"
        ]
    }

    st.dataframe(modules, use_container_width=True)

    # 9. PROJECT EXECUTION
    st.header("9. How to Run the Project")

    st.markdown(
        "**Step 1 — Open PowerShell in the project folder**"
    )

    st.markdown("**Step 2 — Generate sales data**")

    st.code(
        "python src/generate_sales.py",
        language="powershell"
    )

    st.markdown("**Step 3 — Run sales analysis**")

    st.code(
        "python src/analyze_sales.py",
        language="powershell"
    )

    st.markdown("**Step 4 — Generate demand forecast**")

    st.code(
        "python src/demand_forecasting.py",
        language="powershell"
    )

    st.markdown("**Step 5 — Launch dashboard**")

    st.code(
        "python -m streamlit run dashboard/app.py",
        language="powershell"
    )

    st.warning(
        "The report-backed dashboard sections run from the included "
        "CSV files without MySQL. MySQL is needed only for the data "
        "generation and analysis scripts and the live inventory and "
        "supplier sections; Streamlit Cloud requires a reachable "
        "external database."
    )

    # 10. REPORTS
    st.header("10. Generated Reports")

    reports = [
        "monthly_sales.csv",
        "product_sales_analysis.csv",
        "demand_forecast.csv",
        "forecast_summary.csv",
        "forecast_evaluation.csv",
        "product_forecast_accuracy.csv",
        "model_performance.csv",
        "monthly_revenue.png",
        "top_products.png",
        "forecast_comparison.png",
        "forecast_actual_comparison.png"
    ]

    for report in reports:
        st.markdown(f"- `{report}`")

    st.caption("These files are generated and stored in the reports folder.")

    # 11. LIMITATIONS
    st.header("11. Limitations and Future Scope")

    st.markdown("""
    **Current limitations:**
    - Sales data is simulated for demonstration.
    - Forecasting history covers six months.
    - Forecasts are estimates, not guaranteed outcomes.
    - Reorder recommendations use basic stock thresholds.

    **Future enhancements:**
    - Integration with live retail data.
    - Advanced forecasting algorithms.
    - Automated purchase recommendations.
    - Real-time inventory updates.
    - User authentication and role-based access.
    - Cloud deployment.
    """)

    # 12. CONCLUSION
    st.header("12. Project Conclusion")

    st.success("""
    INVENTRA combines database management, data analytics,
    machine learning and interactive visualisation into
    one inventory intelligence platform.

    Its purpose is to help businesses understand their
    inventory, analyse historical sales and make better
    demand-planning decisions.
    """)

    st.divider()

    st.markdown(
        "**INVENTRA — Smart Inventory Management & "
        "AI-Based Demand Forecasting Platform**"
    )

    st.caption("Project developed by the INVENTRA Project Team.")




# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    INVENTRA | Inventory Management & Demand Forecasting<br>
    Powered by Python, Machine Learning, Streamlit & Plotly
</div>
""", unsafe_allow_html=True)

st.caption(
    "Note: Historical sales in this demonstration use simulated data. "
    "Forecasts are model estimates, not guaranteed future demand."
)
