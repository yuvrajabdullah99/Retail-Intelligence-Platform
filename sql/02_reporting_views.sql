-- ============================================================
-- VIEWS
-- ============================================================

DROP VIEW IF EXISTS vw_daily_store_revenue;
DROP VIEW IF EXISTS vw_daily_category_revenue;
DROP VIEW IF EXISTS vw_customer_rfm;
DROP VIEW IF EXISTS vw_product_revenue_rank_by_city;
DROP VIEW IF EXISTS vw_rolling_30d_revenue;
DROP VIEW IF EXISTS vw_monthly_growth;
DROP VIEW IF EXISTS vw_repeat_purchase_rate;

-- Daily revenue at store grain (base view many others build on)
CREATE VIEW vw_daily_store_revenue AS
SELECT
    t.date,
    t.store_id,
    s.city,
    s.city_tier,
    SUM(t.revenue)               AS daily_revenue,
    SUM(t.units_sold)            AS daily_units,
    COUNT(DISTINCT t.customer_id) AS daily_unique_customers,
    COUNT(DISTINCT t.transaction_id) AS daily_transactions
FROM transactions t
JOIN stores s ON s.store_id = t.store_id
GROUP BY t.date, t.store_id, s.city, s.city_tier;

-- Daily revenue at category grain
CREATE VIEW vw_daily_category_revenue AS
SELECT
    t.date,
    p.category,
    SUM(t.revenue)    AS daily_revenue,
    SUM(t.units_sold) AS daily_units
FROM transactions t
JOIN products p ON p.product_id = t.product_id
GROUP BY t.date, p.category;

-- Rolling 30-day revenue per store (window function)
CREATE VIEW vw_rolling_30d_revenue AS
SELECT
    date,
    store_id,
    daily_revenue,
    SUM(daily_revenue) OVER (
        PARTITION BY store_id
        ORDER BY date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS rolling_30d_revenue,
    AVG(daily_revenue) OVER (
        PARTITION BY store_id
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_avg_revenue
FROM vw_daily_store_revenue;

-- Month-over-month growth per store 
CREATE VIEW vw_monthly_growth AS
WITH monthly AS (
    SELECT
        store_id,
        strftime('%Y-%m', date) AS year_month,
        SUM(daily_revenue) AS monthly_revenue
    FROM vw_daily_store_revenue
    GROUP BY store_id, strftime('%Y-%m', date)
)
SELECT
    store_id,
    year_month,
    monthly_revenue,
    LAG(monthly_revenue) OVER (PARTITION BY store_id ORDER BY year_month) AS prev_month_revenue,
    ROUND(
        100.0 * (monthly_revenue - LAG(monthly_revenue) OVER (PARTITION BY store_id ORDER BY year_month))
        / NULLIF(LAG(monthly_revenue) OVER (PARTITION BY store_id ORDER BY year_month), 0),
        2
    ) AS mom_growth_pct
FROM monthly;

-- Customer RFM (Recency, Frequency, Monetary) — feeds both BI and feature engineering
CREATE VIEW vw_customer_rfm AS
WITH agg AS (
    SELECT
        customer_id,
        MAX(date)              AS last_purchase_date,
        COUNT(DISTINCT date)   AS frequency,
        SUM(revenue)           AS monetary,
        COUNT(DISTINCT transaction_id) AS n_transactions
    FROM transactions
    GROUP BY customer_id
)
SELECT
    customer_id,
    last_purchase_date,
    CAST(julianday((SELECT MAX(date) FROM transactions)) - julianday(last_purchase_date) AS INTEGER) AS recency_days,
    frequency,
    ROUND(monetary, 2) AS monetary,
    ROUND(monetary / NULLIF(n_transactions, 0), 2) AS avg_basket_value,
    n_transactions
FROM agg;

-- Repeat purchase rate helper: flags customers with >1 distinct purchase day
CREATE VIEW vw_repeat_purchase_rate AS
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN frequency > 1 THEN 1 ELSE 0 END) AS repeat_customers,
    ROUND(100.0 * SUM(CASE WHEN frequency > 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS repeat_purchase_rate_pct
FROM vw_customer_rfm;

-- Top products by revenue, ranked WITHIN each city (window function: RANK)
CREATE VIEW vw_product_revenue_rank_by_city AS
WITH city_product_rev AS (
    SELECT
        s.city,
        p.product_id,
        p.product_name,
        p.category,
        SUM(t.revenue) AS total_revenue
    FROM transactions t
    JOIN stores s ON s.store_id = t.store_id
    JOIN products p ON p.product_id = t.product_id
    GROUP BY s.city, p.product_id, p.product_name, p.category
)
SELECT
    city,
    product_id,
    product_name,
    category,
    total_revenue,
    RANK() OVER (PARTITION BY city ORDER BY total_revenue DESC) AS revenue_rank_in_city
FROM city_product_rev;
