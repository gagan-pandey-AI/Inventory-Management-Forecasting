import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
from getpass import getpass
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# MySQL connection
password = getpass("Enter MySQL root password: ")

db = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password=password,
    database="inventoryforecastdb"
)

cursor = db.cursor(dictionary=True)

print("\nConnected to MySQL successfully!")

# 1. Monthly sales analysis
cursor.execute("""
    SELECT
        DATE_FORMAT(sale_date, '%Y-%m') AS sales_month,
        SUM(quantity_sold) AS total_quantity,
        ROUND(SUM(quantity_sold * unit_price), 2) AS revenue
    FROM sales
    GROUP BY DATE_FORMAT(sale_date, '%Y-%m')
    ORDER BY sales_month
""")

monthly_df = pd.DataFrame(cursor.fetchall())

# 2. Product-wise sales analysis
cursor.execute("""
    SELECT
        p.product_name,
        SUM(s.quantity_sold) AS total_quantity,
        ROUND(SUM(s.quantity_sold * s.unit_price), 2) AS revenue
    FROM sales s
    JOIN products p
        ON s.product_id = p.product_id
    GROUP BY p.product_id, p.product_name
    ORDER BY revenue DESC
""")

product_df = pd.DataFrame(cursor.fetchall())

# 3. Save CSV reports
monthly_df.to_csv(
    REPORT_DIR / "monthly_sales.csv",
    index=False
)

product_df.to_csv(
    REPORT_DIR / "product_sales_analysis.csv",
    index=False
)

# 4. Monthly revenue chart
plt.figure(figsize=(10, 5))
plt.plot(
    monthly_df["sales_month"],
    monthly_df["revenue"],
    marker="o"
)
plt.title("Monthly Revenue Analysis")
plt.xlabel("Month")
plt.ylabel("Revenue (INR)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(REPORT_DIR / "monthly_revenue.png")
plt.close()

# 5. Top products chart
top_products = product_df.head(10)

plt.figure(figsize=(10, 6))
plt.barh(
    top_products["product_name"],
    top_products["revenue"]
)
plt.title("Product-wise Revenue Analysis")
plt.xlabel("Revenue (INR)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(REPORT_DIR / "top_products.png")
plt.close()

# 6. Print summary
total_revenue = product_df["revenue"].sum()
total_quantity = product_df["total_quantity"].sum()

print("\n========== SALES ANALYSIS ==========")
print("Total Quantity Sold:", total_quantity)
print("Total Revenue: ₹", round(total_revenue, 2))

print("\nMONTHLY SALES:")
print(monthly_df.to_string(index=False))

print("\nTOP PRODUCTS:")
print(product_df.head(5).to_string(index=False))

print("\nReports saved in:", REPORT_DIR)

cursor.close()
db.close()