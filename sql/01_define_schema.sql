-- ============================================================
-- Retail Intelligence Platform - Schema
-- ============================================================

DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS inventory;
DROP TABLE IF EXISTS promotions;
DROP TABLE IF EXISTS weather;
DROP TABLE IF EXISTS holidays;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS stores;

CREATE TABLE stores (
    store_id        INTEGER PRIMARY KEY,
    store_name      TEXT NOT NULL,
    city            TEXT NOT NULL,
    city_tier       INTEGER NOT NULL,
    latitude        REAL,
    longitude       REAL,
    footfall_index  REAL,
    store_size_sqft INTEGER,
    opened_date     DATE
);

CREATE TABLE products (
    product_id       INTEGER PRIMARY KEY,
    product_name     TEXT NOT NULL,
    category         TEXT NOT NULL,
    unit_cost        REAL NOT NULL,
    unit_price       REAL NOT NULL,
    weather_sensitive INTEGER,
    launch_date      DATE
);

CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY,
    segment       TEXT,
    city          TEXT,
    signup_date   DATE,
    age           REAL,
    gender        TEXT
);

CREATE TABLE holidays (
    date          DATE PRIMARY KEY,
    holiday_name  TEXT
);

CREATE TABLE weather (
    date          DATE,
    city          TEXT,
    temp_celsius  REAL,
    rainfall_mm   REAL,
    is_rainy      INTEGER,
    PRIMARY KEY (date, city)
);

CREATE TABLE promotions (
    promo_id      INTEGER PRIMARY KEY,
    promo_name    TEXT,
    category      TEXT,
    start_date    DATE,
    end_date      DATE,
    discount_pct  INTEGER
);

CREATE TABLE transactions (
    transaction_id        INTEGER PRIMARY KEY,
    date                  DATE NOT NULL,
    store_id              INTEGER NOT NULL REFERENCES stores(store_id),
    product_id            INTEGER NOT NULL REFERENCES products(product_id),
    customer_id           INTEGER NOT NULL REFERENCES customers(customer_id),
    units_sold            INTEGER NOT NULL,
    unit_price_effective  REAL NOT NULL,
    discount_pct          INTEGER,
    revenue               REAL NOT NULL
);

CREATE TABLE inventory (
    date            DATE,
    store_id        INTEGER,
    product_id      INTEGER,
    units_on_hand   INTEGER,
    reorder_point   INTEGER
);

-- ============================================================
-- INDEXING STRATEGY
-- Rationale documented per index  
-- ============================================================

-- Covering index: almost every analytical query filters/groups by date
-- range first, then store. This index lets the planner prune massive
-- chunks of the table before touching product/customer columns
-- (the "poor man's partition pruning" mentioned above)
CREATE INDEX idx_txn_date_store ON transactions(date, store_id);

-- Supports product-level rollups (top-N products, category trends)
CREATE INDEX idx_txn_product_date ON transactions(product_id, date);

-- Supports customer-level RFM / CLV queries (recency/frequency scans
-- per customer are the single most expensive query pattern here)
CREATE INDEX idx_txn_customer_date ON transactions(customer_id, date);

-- Inventory lookups are always store+product for a given week
CREATE INDEX idx_inv_store_product_date ON inventory(store_id, product_id, date);

-- Promotions are scanned by category + active date range
CREATE INDEX idx_promo_category_dates ON promotions(category, start_date, end_date);

CREATE INDEX idx_weather_city_date ON weather(city, date);
