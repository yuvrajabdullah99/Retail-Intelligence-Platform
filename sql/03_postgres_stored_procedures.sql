-- ============================================================
-- POSTGRES-ONLY: Stored Procedures & Partitioning
-- ============================================================

-- ---- 1. Native range partitioning on the transactions table ----
CREATE TABLE transactions_p (
    transaction_id        BIGINT,
    date                  DATE NOT NULL,
    store_id              INTEGER NOT NULL,
    product_id            INTEGER NOT NULL,
    customer_id           INTEGER NOT NULL,
    units_sold            INTEGER NOT NULL,
    unit_price_effective  NUMERIC(10,2) NOT NULL,
    discount_pct          INTEGER,
    revenue               NUMERIC(12,2) NOT NULL
) PARTITION BY RANGE (date);

-- One partition per quarter keeps partition count manageable while 
-- still letting the planner skip ~75% of the table for any single-quarter

CREATE TABLE transactions_2024_q1 PARTITION OF transactions_p
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');
CREATE TABLE transactions_2024_q2 PARTITION OF transactions_p
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');
CREATE TABLE transactions_2024_q3 PARTITION OF transactions_p
    FOR VALUES FROM ('2024-07-01') TO ('2024-10-01');
CREATE TABLE transactions_2024_q4 PARTITION OF transactions_p
    FOR VALUES FROM ('2024-10-01') TO ('2025-01-01');
CREATE TABLE transactions_2025_q1 PARTITION OF transactions_p
    FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');
CREATE TABLE transactions_2025_q2 PARTITION OF transactions_p
    FOR VALUES FROM ('2025-04-01') TO ('2025-07-01');
CREATE TABLE transactions_2025_q3 PARTITION OF transactions_p
    FOR VALUES FROM ('2025-07-01') TO ('2025-10-01');
CREATE TABLE transactions_2025_q4 PARTITION OF transactions_p
    FOR VALUES FROM ('2025-10-01') TO ('2026-01-01');

CREATE INDEX ON transactions_p (store_id, date);
CREATE INDEX ON transactions_p (product_id, date);

-- ---- 2. Stored procedure:- refresh a materialized rollup table ----
-- Mirrors what a nightly Airflow job would call to refresh the
-- dashboard's pre-aggregated KPI table instead of hitting raw
-- transactions at query time.
CREATE OR REPLACE PROCEDURE refresh_daily_store_kpis()
LANGUAGE plpgsql
AS $$
BEGIN
    TRUNCATE TABLE daily_store_kpis;

    INSERT INTO daily_store_kpis (date, store_id, revenue, units, unique_customers)
    SELECT
        date,
        store_id,
        SUM(revenue),
        SUM(units_sold),
        COUNT(DISTINCT customer_id)
    FROM transactions_p
    GROUP BY date, store_id;

    RAISE NOTICE 'daily_store_kpis refreshed at %', now();
END;
$$;



-- ---- 3. Function: Customer Lifetime Value (parameterized, callable) ----
CREATE OR REPLACE FUNCTION get_customer_ltv(p_customer_id INT)
RETURNS NUMERIC AS $$
DECLARE
    v_ltv NUMERIC;
BEGIN
    SELECT COALESCE(SUM(revenue), 0) INTO v_ltv
    FROM transactions_p
    WHERE customer_id = p_customer_id;

    RETURN v_ltv;
END;
$$ LANGUAGE plpgsql;

-- Usage: SELECT get_customer_ltv(1042);

-- ---- 4. Function: revenue impact of a hypothetical inventory increase ----
-- Used to generate the "increase inventory of X by 15% -> +€Y profit"
-- style business recommendation programmatically.
CREATE OR REPLACE FUNCTION estimate_inventory_uplift_profit(
    p_product_id INT,
    p_region TEXT,
    p_pct_increase NUMERIC
) RETURNS NUMERIC AS $$
DECLARE
    v_avg_daily_units NUMERIC;
    v_margin NUMERIC;
    v_result NUMERIC;
BEGIN
    SELECT AVG(daily_units) INTO v_avg_daily_units
    FROM (
        SELECT date, SUM(t.units_sold) AS daily_units
        FROM transactions_p t
        JOIN stores s ON s.store_id = t.store_id
        WHERE t.product_id = p_product_id AND s.city = p_region
        GROUP BY date
    ) sub;

    SELECT (unit_price - unit_cost) INTO v_margin
    FROM products WHERE product_id = p_product_id;

    v_result := v_avg_daily_units * (p_pct_increase / 100.0) * v_margin * 30; -- projected over 30 days
    RETURN ROUND(v_result, 2);
END;
$$ LANGUAGE plpgsql;
