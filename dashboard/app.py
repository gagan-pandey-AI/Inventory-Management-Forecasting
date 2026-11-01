from datetime import date
from pathlib import Path

import mysql.connector
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from inventory_intelligence import (
    build_inventory_analysis,
    calculate_inventory_turnover,
    get_reorder_recommendations,
)

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


def get_mysql_table_columns(cursor, table_name):
    cursor.execute(
        """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
        """,
        (table_name,)
    )
    return {row[0] for row in cursor.fetchall()}


def load_inventory_intelligence_from_mysql(config):
    connection = connect_to_database(**config)
    cursor = None
    try:
        cursor = connection.cursor()
        inventory_columns = get_mysql_table_columns(cursor, "inventory")
        product_columns = get_mysql_table_columns(cursor, "products")
        sales_columns = get_mysql_table_columns(cursor, "sales")
        supplier_columns = get_mysql_table_columns(cursor, "suppliers")
        purchase_columns = get_mysql_table_columns(cursor, "purchase_orders")
        purchase_item_columns = get_mysql_table_columns(
            cursor,
            "purchase_order_items"
        )

        required_inventory_columns = {
            "product_id",
            "stock_quantity",
            "reorder_level",
            "lead_time_days",
        }
        if not required_inventory_columns.issubset(inventory_columns):
            raise ValueError("The inventory table is missing required columns.")
        if not {"product_id", "product_name"}.issubset(product_columns):
            raise ValueError("The products table is missing required columns.")

        supplier_join = ""
        supplier_select = ""
        if (
            "supplier_id" in product_columns
            and {"supplier_id", "supplier_name"}.issubset(supplier_columns)
        ):
            supplier_join = (
                " LEFT JOIN suppliers s ON p.supplier_id = s.supplier_id"
            )
            supplier_select = ", s.supplier_name"

        inventory_query = f"""
            SELECT
                p.product_id,
                p.product_name,
                i.stock_quantity,
                i.reorder_level,
                i.lead_time_days,
                i.last_updated
                {supplier_select}
            FROM inventory i
            JOIN products p ON i.product_id = p.product_id
            {supplier_join}
            ORDER BY p.product_id
        """
        cursor.execute(inventory_query)
        inventory = pd.DataFrame(
            cursor.fetchall(),
            columns=[column[0] for column in cursor.description],
        )
        if "supplier_name" in inventory.columns:
            inventory["supplier_mapping_source"] = "Product master"

        sales_select = "sale_date, product_id, quantity_sold"
        if "unit_cost" in sales_columns:
            sales_select += ", unit_cost"
        cursor.execute(
            f"""
            SELECT {sales_select}
            FROM sales
            WHERE sale_date >= DATE_SUB(CURRENT_DATE, INTERVAL 90 DAY)
              AND sale_date <= CURRENT_DATE
            ORDER BY sale_date
            """
        )
        sales = pd.DataFrame(
            cursor.fetchall(),
            columns=[column[0] for column in cursor.description],
        )

        if (
            "product_id" in purchase_item_columns
            and "quantity_ordered" in purchase_item_columns
            and "order_id" in purchase_item_columns
            and {"order_id", "order_status"}.issubset(purchase_columns)
        ):
            cursor.execute(
                """
                SELECT
                    poi.product_id,
                    SUM(poi.quantity_ordered) AS pending_order_quantity
                FROM purchase_order_items poi
                JOIN purchase_orders po ON po.order_id = poi.order_id
                WHERE po.order_status = 'Pending'
                GROUP BY poi.product_id
                """
            )
            pending = pd.DataFrame(
                cursor.fetchall(),
                columns=[column[0] for column in cursor.description],
            )
            inventory = inventory.merge(
                pending,
                on="product_id",
                how="left",
                validate="one_to_one",
            )
            inventory["pending_order_quantity"] = (
                inventory["pending_order_quantity"].fillna(0)
            )

        if (
            "supplier_name" not in inventory.columns
            and "product_id" in purchase_item_columns
            and "quantity_ordered" in purchase_item_columns
            and "order_id" in purchase_item_columns
            and {"order_id", "supplier_id"}.issubset(purchase_columns)
            and {"supplier_id", "supplier_name"}.issubset(supplier_columns)
        ):
            cursor.execute(
                """
                SELECT
                    poi.product_id,
                    s.supplier_name,
                    SUM(poi.quantity_ordered) AS ordered_units
                FROM purchase_order_items poi
                JOIN purchase_orders po ON po.order_id = poi.order_id
                JOIN suppliers s ON s.supplier_id = po.supplier_id
                GROUP BY poi.product_id, s.supplier_id, s.supplier_name
                ORDER BY poi.product_id, ordered_units DESC, s.supplier_name
                """
            )
            supplier_history = pd.DataFrame(
                cursor.fetchall(),
                columns=[column[0] for column in cursor.description],
            )
            supplier_history = supplier_history.drop_duplicates(
                "product_id",
                keep="first",
            )[["product_id", "supplier_name"]]
            inventory = inventory.merge(
                supplier_history,
                on="product_id",
                how="left",
                validate="one_to_one",
            )
            inventory["supplier_mapping_source"] = (
                "Most ordered historical supplier"
            )
        if "supplier_name" not in inventory.columns:
            inventory["supplier_name"] = pd.NA
            inventory["supplier_mapping_source"] = "Supplier not mapped"

        return inventory, sales
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


def get_mysql_secret_config():
    try:
        has_mysql_secrets = "mysql" in st.secrets
    except StreamlitSecretNotFoundError:
        return None
    if not has_mysql_secrets:
        return None
    configured = st.secrets["mysql"]
    required = ("host", "user", "password")
    missing = [key for key in required if not configured.get(key)]
    if missing:
        raise ValueError(
            "The mysql Streamlit secret is missing required settings: "
            + ", ".join(missing)
        )
    return {
        "host": configured["host"],
        "user": configured["user"],
        "password": configured["password"],
        "database": configured.get("database", "inventoryforecastdb"),
        "port": int(configured.get("port", 3306)),
    }


def render_intelligence_data_source(section_key):
    st.markdown("### Inventory data source")
    st.caption(
        "Connect to a reachable MySQL server or upload a current inventory "
        "export. On Streamlit Community Cloud, localhost is the app container."
    )
    source = st.radio(
        "Choose inventory source",
        ["Upload inventory CSV", "Connect to MySQL"],
        horizontal=True,
        key=f"{section_key}_data_source",
    )

    if source == "Upload inventory CSV":
        uploaded_inventory = st.file_uploader(
            "Current inventory CSV",
            type=["csv"],
            help=(
                "Required columns: product_id, product_name, stock_quantity, "
                "reorder_level, lead_time_days. Optional: supplier_name and "
                "pending_order_quantity."
            ),
            key=f"{section_key}_inventory_upload",
        )
        uploaded_sales = st.file_uploader(
            "Optional recent sales CSV",
            type=["csv"],
            help=(
                "For products without an available forecast, provide "
                "product_id, sale_date, and quantity_sold."
            ),
            key=f"{section_key}_sales_upload",
        )
        if uploaded_inventory is None:
            return None, None
        try:
            inventory = pd.read_csv(uploaded_inventory)
            required = {
                "product_id",
                "product_name",
                "stock_quantity",
                "reorder_level",
                "lead_time_days",
            }
            missing = required.difference(inventory.columns)
            if missing:
                st.error(
                    "Inventory CSV is missing required columns: "
                    + ", ".join(sorted(missing))
                )
                return None, None
            sales = (
                pd.read_csv(uploaded_sales)
                if uploaded_sales is not None
                else None
            )
            if sales is not None:
                sales_required = {
                    "product_id",
                    "sale_date",
                    "quantity_sold",
                }
                missing_sales = sales_required.difference(sales.columns)
                if missing_sales:
                    st.error(
                        "Sales CSV is missing required columns: "
                        + ", ".join(sorted(missing_sales))
                    )
                    return None, None
            return inventory, sales
        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
            OSError,
        ):
            st.error("Unable to read the uploaded CSV file.")
            return None, None

    try:
        secret_config = get_mysql_secret_config()
    except ValueError as error:
        st.error(str(error))
        return None, None

    if secret_config is not None:
        if st.button(
            "Load live inventory from configured MySQL",
            key=f"{section_key}_secret_mysql_submit",
        ):
            try:
                inventory, sales = load_inventory_intelligence_from_mysql(
                    secret_config
                )
                st.session_state["intelligence_inventory"] = inventory
                st.session_state["intelligence_sales"] = sales
                st.success("Live inventory data loaded.")
            except (mysql.connector.Error, ValueError):
                st.session_state.pop("intelligence_inventory", None)
                st.session_state.pop("intelligence_sales", None)
                st.error(
                    "Unable to load inventory intelligence data. Check the "
                    "reachable database, credentials, and required schema. "
                    "Connection details are not displayed."
                )
        return (
            st.session_state.get("intelligence_inventory"),
            st.session_state.get("intelligence_sales"),
        )

    st.caption(
        "For Streamlit Community Cloud, you may instead configure a "
        "[mysql] section in the app's Secrets settings. Never commit "
        "database credentials."
    )
    st.code(
        '[mysql]\nhost = "your-external-database-host"\n'
        'user = "your-database-user"\npassword = "..."\n'
        'database = "inventoryforecastdb"\nport = 3306',
        language="toml",
    )
    with st.form(f"{section_key}_mysql_form"):
        host = st.text_input("MySQL Host", key=f"{section_key}_mysql_host")
        user = st.text_input("MySQL Username", key=f"{section_key}_mysql_user")
        password = st.text_input(
            "MySQL Password",
            type="password",
            key=f"{section_key}_mysql_password",
        )
        database = st.text_input(
            "Database",
            value="inventoryforecastdb",
            key=f"{section_key}_mysql_database",
        )
        connect = st.form_submit_button("Load live inventory")

    if connect:
        if not all((host, user, password, database)):
            st.warning("Enter the MySQL host, username, password, and database.")
        else:
            try:
                inventory, sales = load_inventory_intelligence_from_mysql(
                    {
                        "host": host,
                        "user": user,
                        "password": password,
                        "database": database,
                        "port": 3306,
                    }
                )
                st.session_state["intelligence_inventory"] = inventory
                st.session_state["intelligence_sales"] = sales
                st.success("Live inventory data loaded.")
            except (mysql.connector.Error, ValueError):
                st.session_state.pop("intelligence_inventory", None)
                st.session_state.pop("intelligence_sales", None)
                st.error(
                    "Unable to load inventory intelligence data. Check the "
                    "reachable database, credentials, and required schema. "
                    "Connection details are not displayed."
                )

    return (
        st.session_state.get("intelligence_inventory"),
        st.session_state.get("intelligence_sales"),
    )


# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    monthly = pd.read_csv(REPORTS_DIR / "monthly_sales.csv")
    products = pd.read_csv(REPORTS_DIR / "product_sales_analysis.csv")
    forecast = pd.read_csv(REPORTS_DIR / "forecast_summary.csv")
    return monthly, products, forecast


@st.cache_data
def load_daily_forecast():
    return pd.read_csv(REPORTS_DIR / "demand_forecast.csv")


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
        "Inventory Turnover",
        "Overstock Intelligence",
        "AI Reorder Recommendations",
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

elif page == "Inventory Turnover":

    st.markdown(
        '<div class="section-title">INVENTORY TURNOVER ANALYSIS</div>',
        unsafe_allow_html=True
    )
    st.info(
        "A single current-stock row cannot establish average inventory. "
        "Upload dated stock snapshots and costed sales transactions to "
        "calculate actual cost-based turnover; sales unit_price is not COGS."
    )

    snapshot_upload = st.file_uploader(
        "Inventory snapshots CSV",
        type=["csv"],
        help=(
            "Required columns: product_id, snapshot_date, stock_quantity, "
            "unit_cost. Include at least two dated snapshots per product."
        ),
        key="turnover_snapshots_upload",
    )
    sales_upload = st.file_uploader(
        "Costed sales transactions CSV",
        type=["csv"],
        help=(
            "Required columns: product_id, sale_date, quantity_sold, unit_cost. "
            "unit_cost must be acquisition cost, not sales unit_price."
        ),
        key="turnover_sales_upload",
    )
    turnover_period_days = st.slider(
        "Analysis period (days)",
        min_value=30,
        max_value=365,
        value=90,
        step=30,
        key="turnover_period_days",
    )
    turnover_as_of = st.date_input(
        "Period end date",
        value=date.today(),
        key="turnover_as_of_date",
    )

    if snapshot_upload is None or sales_upload is None:
        st.warning(
            "Turnover results are unavailable until both source CSVs are "
            "provided. No stock history or acquisition cost is inferred."
        )
    else:
        try:
            snapshot_data = pd.read_csv(snapshot_upload)
            costed_sales = pd.read_csv(sales_upload)
            turnover, turnover_explanation = calculate_inventory_turnover(
                snapshot_data,
                costed_sales,
                period_days=turnover_period_days,
                as_of_date=turnover_as_of,
            )
        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
            OSError,
        ):
            st.error("Unable to read one of the uploaded turnover CSV files.")
        else:
            st.caption(turnover_explanation)
            if turnover.empty:
                st.warning(
                    "No product-level turnover can be calculated from the "
                    "provided files. Check the required columns and dates."
                )
            else:
                eligible_turnover = turnover.dropna(
                    subset=["turnover_ratio", "average_inventory_value"]
                )
                if eligible_turnover.empty:
                    st.warning(
                        "The files do not contain enough valid costed sales "
                        "and dated inventory snapshots for a turnover ratio."
                    )
                else:
                    total_cogs = eligible_turnover["cogs"].sum()
                    total_average_inventory = eligible_turnover[
                        "average_inventory_value"
                    ].sum()
                    portfolio_turnover = (
                        total_cogs / total_average_inventory
                        if total_average_inventory > 0
                        else 0
                    )
                    portfolio_dio = (
                        turnover_period_days / portfolio_turnover
                        if portfolio_turnover > 0
                        else float("nan")
                    )

                    st.markdown("### PORTFOLIO PERFORMANCE")
                    turnover_kpis = st.columns(4)
                    turnover_kpis[0].metric(
                        "COGS",
                        f"₹{total_cogs:,.2f}",
                    )
                    turnover_kpis[1].metric(
                        "Average Inventory Value",
                        f"₹{total_average_inventory:,.2f}",
                    )
                    turnover_kpis[2].metric(
                        "Inventory Turnover",
                        f"{portfolio_turnover:.2f}×",
                    )
                    turnover_kpis[3].metric(
                        "Days Inventory Outstanding",
                        f"{portfolio_dio:.1f}"
                        if pd.notna(portfolio_dio)
                        else "N/A",
                    )

                    st.markdown("### PRODUCT-WISE TURNOVER")
                    fig_turnover = px.bar(
                        eligible_turnover.sort_values(
                            "annualized_turnover",
                            ascending=True,
                        ),
                        x="annualized_turnover",
                        y="product_name",
                        color="movement_classification",
                        orientation="h",
                        title="Annualized Turnover by Product",
                        labels={
                            "annualized_turnover": "Turns per year",
                            "product_name": "Product",
                        },
                    )
                    fig_turnover.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#CBD5E1",
                    )
                    st.plotly_chart(
                        fig_turnover,
                        use_container_width=True,
                    )
                st.dataframe(
                    turnover,
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    "Download Inventory Turnover CSV",
                    data=turnover.to_csv(index=False).encode("utf-8"),
                    file_name="inventory_turnover_analysis.csv",
                    mime="text/csv",
                )


elif page == "Overstock Intelligence":

    st.markdown(
        '<div class="section-title">OVERSTOCK INTELLIGENCE</div>',
        unsafe_allow_html=True
    )
    inventory_data, historical_sales = render_intelligence_data_source(
        "overstock"
    )
    safety_buffer_days = st.slider(
        "Low-stock safety buffer (days)",
        min_value=0,
        max_value=30,
        value=7,
        help="Used after supplier lead time to identify low-stock coverage.",
        key="overstock_safety_buffer",
    )
    overstock_target_days = st.slider(
        "Overstock target coverage (days)",
        min_value=15,
        max_value=180,
        value=60,
        help=(
            "Stock above forecast demand for this many days is classified "
            "as overstock. This is a configurable planning policy."
        ),
        key="overstock_target",
    )
    if inventory_data is None:
        st.info("Provide current inventory data to calculate stock coverage.")
    else:
        try:
            daily_forecast = load_daily_forecast()
        except FileNotFoundError:
            daily_forecast = pd.DataFrame()
            st.warning(
                "The daily demand forecast report is missing. Historical "
                "sales will be used only where supplied."
            )
        analysis = build_inventory_analysis(
            inventory_data,
            daily_forecast,
            historical_sales,
            safety_buffer_days=safety_buffer_days,
            overstock_target_days=overstock_target_days,
        )

        st.markdown("### STOCK HEALTH")
        category_counts = analysis["classification"].value_counts()
        overstocked = analysis.loc[
            analysis["classification"].eq("Overstock")
        ]
        known_excess = overstocked["excess_quantity"].dropna()
        kpis = st.columns(4)
        kpis[0].metric("Products Analyzed", len(analysis))
        kpis[1].metric(
            "Critical / Low",
            int(
                category_counts.get("Critical Low Stock", 0)
                + category_counts.get("Low Stock", 0)
            ),
        )
        kpis[2].metric(
            "Overstocked Products",
            int(category_counts.get("Overstock", 0)),
        )
        kpis[3].metric(
            "Known Excess Units",
            f"{known_excess.sum():,.0f}" if not known_excess.empty else "N/A",
        )

        st.caption(
            "Critical: at/below reorder level or forecast to run out by "
            "supplier lead time. Low: coverage within lead time plus the "
            f"{safety_buffer_days}-day buffer. Overstock: coverage exceeds "
            f"the configurable {overstock_target_days}-day target. Zero "
            "forecast demand makes days of cover undefined; positive stock "
            "is shown as excess against zero forecast demand. Demand uses "
            "the mean of available future daily forecasts; where no forecast "
            "exists, it falls back to the supplied 90-day sales history."
        )
        coverage_chart = analysis.loc[
            analysis["days_inventory_remaining"].notna()
        ]
        if not coverage_chart.empty:
            fig_coverage = px.bar(
                coverage_chart.sort_values(
                    "days_inventory_remaining",
                    ascending=True,
                ),
                x="days_inventory_remaining",
                y="product_name",
                color="classification",
                orientation="h",
                title="Estimated Days of Inventory Remaining",
                labels={
                    "days_inventory_remaining": "Days of inventory",
                    "product_name": "Product",
                },
            )
            fig_coverage.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#CBD5E1",
            )
            st.plotly_chart(
                fig_coverage,
                use_container_width=True,
            )

        display_columns = [
            "product_id",
            "product_name",
            "stock_quantity",
            "average_daily_demand",
            "demand_source",
            "days_inventory_remaining",
            "expected_demand_overstock_horizon",
            "excess_quantity",
            "classification",
            "classification_explanation",
        ]
        st.dataframe(
            analysis[display_columns],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download Overstock Analysis CSV",
            data=analysis.to_csv(index=False).encode("utf-8"),
            file_name="overstock_intelligence.csv",
            mime="text/csv",
        )


elif page == "AI Reorder Recommendations":

    st.markdown(
        '<div class="section-title">AI REORDER RECOMMENDATIONS</div>',
        unsafe_allow_html=True
    )
    st.info(
        "Recommendations are suggestions only. Review inventory accuracy, "
        "open purchase orders, supplier, and lead time before manually "
        "approving an order."
    )
    inventory_data, historical_sales = render_intelligence_data_source(
        "reorder"
    )
    safety_buffer_days = st.slider(
        "Safety stock buffer (days)",
        min_value=0,
        max_value=30,
        value=7,
        key="reorder_safety_buffer",
    )
    reorder_target_days = st.slider(
        "Order-up-to target beyond lead time (days)",
        min_value=0,
        max_value=90,
        value=30,
        help=(
            "Suggested stock target includes forecast lead-time demand, "
            "the safety buffer, and this configurable coverage target."
        ),
        key="reorder_target",
    )
    if inventory_data is None:
        st.info("Provide current inventory data to generate suggestions.")
    else:
        try:
            daily_forecast = load_daily_forecast()
        except FileNotFoundError:
            daily_forecast = pd.DataFrame()
            st.warning(
                "The daily demand forecast report is missing. Historical "
                "sales will be used only where supplied."
            )
        analysis = build_inventory_analysis(
            inventory_data,
            daily_forecast,
            historical_sales,
            safety_buffer_days=safety_buffer_days,
            reorder_target_days=reorder_target_days,
        )
        recommendations = get_reorder_recommendations(analysis)

        open_orders_verified = (
            "pending_order_quantity" in inventory_data.columns
            and inventory_data["pending_order_quantity"].notna().all()
        )
        if not open_orders_verified:
            st.warning(
                "Open purchase orders could not be verified for every "
                "product. Check for outstanding orders before approving "
                "these suggestions."
            )

        st.markdown("### REPLENISHMENT SUMMARY")
        reorder_kpis = st.columns(3)
        reorder_kpis[0].metric(
            "Products Requiring Review",
            len(recommendations),
        )
        reorder_kpis[1].metric(
            "Suggested Units",
            f"{recommendations['recommended_order_quantity'].sum():,.0f}"
            if not recommendations.empty
            else "0",
        )
        reorder_kpis[2].metric(
            "Critical Urgency",
            int(
                recommendations["reorder_urgency"].eq("Critical").sum()
            )
            if not recommendations.empty
            else 0,
        )

        st.caption(
            "Reorder point = max(recorded reorder level, forecast lead-time "
            "demand + safety-buffer demand). Order-up-to quantity includes "
            "lead-time demand, safety buffer, and the selected target days, "
            "minus current stock and verified pending quantities. Demand is "
            "the mean of available future forecasts, with a 90-day historical "
            "sales fallback where forecasts are unavailable."
        )
        if recommendations.empty:
            st.success(
                "No product currently meets the configured replenishment "
                "trigger, or demand/lead-time data is unavailable."
            )
        else:
            chart = px.bar(
                recommendations.sort_values(
                    "recommended_order_quantity",
                    ascending=True,
                ),
                x="recommended_order_quantity",
                y="product_name",
                color="reorder_urgency",
                orientation="h",
                title="Suggested Replenishment Quantity",
                labels={
                    "recommended_order_quantity": "Suggested units",
                    "product_name": "Product",
                },
            )
            chart.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#CBD5E1",
            )
            st.plotly_chart(
                chart,
                use_container_width=True,
            )

        recommendation_columns = [
            "product_id",
            "product_name",
            "stock_quantity",
            "average_daily_demand",
            "demand_source",
            "expected_lead_time_demand",
            "lead_time_days",
            "reorder_level",
            "pending_order_quantity",
            "recommended_order_quantity",
            "reorder_urgency",
            "supplier_display",
            "estimated_arrival_date",
            "pending_order_check",
        ]
        for optional_column in ("supplier_mapping_source",):
            if optional_column in recommendations.columns:
                recommendation_columns.append(optional_column)
        if not recommendations.empty:
            st.dataframe(
                recommendations[recommendation_columns],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download Reorder Suggestions CSV",
                data=recommendations.to_csv(index=False).encode("utf-8"),
                file_name="ai_reorder_recommendations.csv",
                mime="text/csv",
            )


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
            "Inventory Turnover",
            "Overstock Intelligence",
            "AI Reorder Recommendations",
            "Project Guide"
        ],
        "What It Does": [
            "Displays overall business KPIs",
            "Shows sales and revenue analytics",
            "Displays future demand predictions",
            "Monitors stock and reorder levels",
            "Analyses suppliers and purchase orders",
            "Evaluates machine learning performance",
            "Calculates cost-based turnover from supplied snapshots",
            "Classifies stock coverage against forecast demand",
            "Suggests replenishment quantities for manual review",
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


