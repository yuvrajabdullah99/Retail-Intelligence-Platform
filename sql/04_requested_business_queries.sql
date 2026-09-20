-- ============================================================
-- BUSINESS QUERIES 
-- ============================================================

-- 1. Top 10 products by revenue, in EVERY city
SELECT city, product_name, category, total_revenue, revenue_rank_in_city
FROM vw_product_revenue_rank_by_city
WHERE revenue_rank_in_city <= 10
ORDER BY city, revenue_rank_in_city;

-- 2. Rolling 30-day revenue (per store, latest 10 days shown)
SELECT date, store_id, daily_revenue, rolling_30d_revenue, rolling_7d_avg_revenue
FROM vw_rolling_30d_revenue
WHERE store_id = 1
ORDER BY date DESC
LIMIT 10;

-- 3. Customer Lifetime Value (top 20 highest-value customers)
SELECT customer_id, monetary AS lifetime_value, frequency, recency_days, avg_basket_value
FROM vw_customer_rfm
ORDER BY monetary DESC
LIMIT 20;

-- 4. Repeat purchase rate (overall)
SELECT * FROM vw_repeat_purchase_rate;

-- 4b. Repeat purchase rate by customer segment
SELECT
    c.segment,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN r.frequency > 1 THEN 1 ELSE 0 END) AS repeat_customers,
    ROUND(100.0 * SUM(CASE WHEN r.frequency > 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS repeat_rate_pct
FROM vw_customer_rfm r
JOIN customers c ON c.customer_id = r.customer_id
GROUP BY c.segment
ORDER BY repeat_rate_pct DESC;

-- 5. Average basket value (overall + by store)
SELECT
    s.store_name,
    s.city,
    ROUND(AVG(t.revenue), 2) AS avg_line_value,
    ROUND(SUM(t.revenue) / COUNT(DISTINCT t.transaction_id), 2) AS avg_basket_value
FROM transactions t
JOIN stores s ON s.store_id = t.store_id
GROUP BY s.store_name, s.city
ORDER BY avg_basket_value DESC;

-- 6. Monthly growth (month-over-month %, per store)
SELECT store_id, year_month, monthly_revenue, mom_growth_pct
FROM vw_monthly_growth
WHERE store_id = 1
ORDER BY year_month;

-- ============================================================
-- CTE-heavy example — cohort-style "new vs returning revenue"
-- split per month, demonstrating multi-CTE composition
-- ============================================================
WITH first_purchase AS (
    SELECT customer_id, MIN(date) AS first_purchase_date
    FROM transactions
    GROUP BY customer_id
),
txn_with_cohort AS (
    SELECT
        t.*,
        strftime('%Y-%m', t.date) AS txn_month,
        strftime('%Y-%m', fp.first_purchase_date) AS cohort_month
    FROM transactions t
    JOIN first_purchase fp ON fp.customer_id = t.customer_id
)
SELECT
    txn_month,
    SUM(CASE WHEN txn_month = cohort_month THEN revenue ELSE 0 END) AS new_customer_revenue,
    SUM(CASE WHEN txn_month != cohort_month THEN revenue ELSE 0 END) AS returning_customer_revenue
FROM txn_with_cohort
GROUP BY txn_month
ORDER BY txn_month;

-- ============================================================
-- Stockout detection via CTE (sales history says nonzero,
-- but inventory record shows 0 on hand) — feeds inventory prediction
-- ============================================================
WITH product_avg_demand AS (
    SELECT store_id, product_id, AVG(units_sold) AS avg_daily_units
    FROM transactions
    GROUP BY store_id, product_id
)
SELECT
    i.date, i.store_id, i.product_id, i.units_on_hand, pad.avg_daily_units
FROM inventory i
JOIN product_avg_demand pad
    ON pad.store_id = i.store_id AND pad.product_id = i.product_id
WHERE i.units_on_hand <= i.reorder_point
ORDER BY i.date DESC
LIMIT 20;
