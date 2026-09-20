"""Load generated CSVs into a SQLite database using the defined schema."""
import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB_PATH = ROOT / "retail_analytics.db"
SCHEMA_SQL = ROOT / "sql" / "01_define_schema.sql"

def main():
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_SQL.read_text())

    table_files = {
        "stores": "store_locations.csv",
        "products": "product_catalog.csv",
        "customers": "customer_profiles.csv",
        "holidays": "holiday_calendar.csv",
        "weather": "weather_observations.csv",
        "promotions": "promotion_campaigns.csv",
        "transactions": "sales_transactions.csv",
        "inventory": "inventory_snapshots.csv",
    }

    for table, fname in table_files.items():
        df = pd.read_csv(DATA / fname)
        df.to_sql(table, conn, if_exists="append", index=False)
        print(f"Loaded {len(df):>8,} rows -> {table}")

    conn.commit()

    # Sanity check + build the views / stored-procedure-equivalents
    views_sql = (ROOT / "sql" / "02_reporting_views.sql").read_text()
    conn.executescript(views_sql)
    conn.commit()
    print("Views created.")
    conn.close()
    print("\nDatabase ready at", DB_PATH)

if __name__ == "__main__":
    main()
