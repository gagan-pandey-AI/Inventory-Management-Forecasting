import mysql.connector
import random
from datetime import date, timedelta
from getpass import getpass

random.seed(42)

password = getpass("Enter MySQL root password: ")

db = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password=password,
    database="inventoryforecastdb"
)

cursor = db.cursor()

# Prevent duplicate sales generation
cursor.execute("SELECT COUNT(*) FROM sales")
existing_records = cursor.fetchone()[0]

if existing_records > 0:
    print(f"Sales table already has {existing_records} records.")
    print("Stopping to prevent duplicates.")
    cursor.close()
    db.close()
    raise SystemExit

cursor.execute("""
    SELECT product_id, unit_price
    FROM products
""")

products = cursor.fetchall()

demand = {
    1: 0.2,
    2: 2.0,
    3: 1.5,
    4: 5.0,
    5: 3.0,
    6: 2.5,
    7: 1.2,
    8: 0.5,
    9: 0.4,
    10: 7.0
}

start_date = date(2026, 5, 1)
end_date = date(2026, 11, 1)

sales_records = []
current_date = start_date

while current_date < end_date:

    for product_id, price in products:

        average_demand = demand[product_id]

        if current_date.weekday() >= 5:
            average_demand *= 1.25

        quantity = max(
            0,
            round(random.gauss(
                average_demand,
                max(0.5, average_demand * 0.35)
            ))
        )

        if quantity > 0:
            sales_records.append((
                product_id,
                current_date,
                quantity,
                price
            ))

    current_date += timedelta(days=1)

query = """
INSERT INTO sales
(product_id, sale_date, quantity_sold, unit_price)
VALUES (%s, %s, %s, %s)
"""

try:
    cursor.executemany(query, sales_records)
    db.commit()

    print("\nSales data generated successfully!")
    print("Total records:", len(sales_records))

except Exception as error:
    db.rollback()
    print("Error:", error)

finally:
    cursor.close()
    db.close()