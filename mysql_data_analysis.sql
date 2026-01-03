/* ============================================================================
   MYSQL FOR DATA ANALYSIS - COMPLETE GUIDE
   From Basic to Advanced with Real-World Examples
   ============================================================================ */

-- ============================================================================
-- PART 1: DATABASE SETUP & SAMPLE DATA CREATION
-- ============================================================================

-- Create a database for e-commerce analysis
CREATE DATABASE IF NOT EXISTS ecommerce_analysis;
USE ecommerce_analysis;

-- Create tables with proper data types and constraints
CREATE TABLE customers (
    customer_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    country VARCHAR(50),
    city VARCHAR(50),
    registration_date DATE,
    age INT,
    gender ENUM('M', 'F', 'Other')
);

CREATE TABLE products (
    product_id INT PRIMARY KEY AUTO_INCREMENT,
    product_name VARCHAR(100) NOT NULL,
    category VARCHAR(50),
    price DECIMAL(10, 2),
    stock_quantity INT,
    supplier VARCHAR(50)
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT,
    order_date DATE,
    total_amount DECIMAL(10, 2),
    status ENUM('Pending', 'Shipped', 'Delivered', 'Cancelled'),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE order_items (
    order_item_id INT PRIMARY KEY AUTO_INCREMENT,
    order_id INT,
    product_id INT,
    quantity INT,
    unit_price DECIMAL(10, 2),
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Insert sample data
INSERT INTO customers (first_name, last_name, email, country, city, registration_date, age, gender) VALUES
('John', 'Doe', 'john.doe@email.com', 'USA', 'New York', '2023-01-15', 35, 'M'),
('Jane', 'Smith', 'jane.smith@email.com', 'UK', 'London', '2023-02-20', 28, 'F'),
('Mike', 'Johnson', 'mike.j@email.com', 'USA', 'Chicago', '2023-03-10', 42, 'M'),
('Sarah', 'Williams', 'sarah.w@email.com', 'Canada', 'Toronto', '2023-04-05', 31, 'F'),
('David', 'Brown', 'david.b@email.com', 'USA', 'Los Angeles', '2023-05-12', 45, 'M');

INSERT INTO products (product_name, category, price, stock_quantity, supplier) VALUES
('Laptop Pro', 'Electronics', 1200.00, 50, 'TechCorp'),
('Wireless Mouse', 'Electronics', 25.00, 200, 'TechCorp'),
('Office Chair', 'Furniture', 350.00, 30, 'FurniMax'),
('Desk Lamp', 'Furniture', 45.00, 100, 'FurniMax'),
('Notebook Set', 'Stationery', 15.00, 500, 'PaperPro');

INSERT INTO orders (customer_id, order_date, total_amount, status) VALUES
(1, '2024-01-10', 1245.00, 'Delivered'),
(2, '2024-01-15', 395.00, 'Delivered'),
(1, '2024-02-20', 60.00, 'Shipped'),
(3, '2024-03-05', 1200.00, 'Delivered'),
(4, '2024-03-15', 350.00, 'Cancelled');

INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
(1, 1, 1, 1200.00), (1, 2, 2, 25.00),
(2, 3, 1, 350.00), (2, 4, 1, 45.00),
(3, 4, 1, 45.00), (3, 5, 1, 15.00),
(4, 1, 1, 1200.00);

-- ============================================================================
-- PART 2: BASIC SELECT QUERIES - FOUNDATION
-- ============================================================================

-- 2.1 Basic SELECT with filtering
SELECT * FROM customers;
SELECT first_name, email FROM customers;
SELECT * FROM customers WHERE country = 'USA';
SELECT * FROM products WHERE price > 100;

-- 2.2 Using comparison operators
SELECT * FROM customers WHERE age >= 35;
SELECT * FROM products WHERE price BETWEEN 20 AND 500;
SELECT * FROM customers WHERE country IN ('USA', 'UK', 'Canada');
SELECT * FROM customers WHERE email LIKE '%@email.com';

-- 2.3 Sorting results
SELECT * FROM products ORDER BY price DESC;
SELECT * FROM customers ORDER BY registration_date ASC, age DESC;

-- 2.4 Limiting results
SELECT * FROM products ORDER BY price DESC LIMIT 3;
SELECT * FROM orders ORDER BY order_date DESC LIMIT 5;

-- ============================================================================
-- PART 3: AGGREGATE FUNCTIONS - STATISTICAL ANALYSIS
-- ============================================================================

-- 3.1 COUNT - Counting records
SELECT COUNT(*) AS total_customers FROM customers;
SELECT COUNT(DISTINCT country) AS unique_countries FROM customers;
SELECT COUNT(*) AS usa_customers FROM customers WHERE country = 'USA';

-- 3.2 SUM - Total calculations
SELECT SUM(total_amount) AS total_revenue FROM orders;
SELECT SUM(total_amount) AS delivered_revenue 
FROM orders WHERE status = 'Delivered';

-- 3.3 AVG - Average calculations
SELECT AVG(price) AS avg_product_price FROM products;
SELECT AVG(age) AS avg_customer_age FROM customers;
SELECT AVG(total_amount) AS avg_order_value FROM orders;

-- 3.4 MIN and MAX
SELECT MIN(price) AS cheapest_product, MAX(price) AS most_expensive 
FROM products;
SELECT MIN(order_date) AS first_order, MAX(order_date) AS latest_order 
FROM orders;

-- 3.5 Multiple aggregates together
SELECT 
    COUNT(*) AS total_orders,
    SUM(total_amount) AS total_revenue,
    AVG(total_amount) AS avg_order_value,
    MIN(total_amount) AS min_order,
    MAX(total_amount) AS max_order
FROM orders;

-- ============================================================================
-- PART 4: GROUP BY - CATEGORICAL ANALYSIS
-- ============================================================================

-- 4.1 Basic GROUP BY
SELECT country, COUNT(*) AS customer_count
FROM customers
GROUP BY country;

-- 4.2 GROUP BY with multiple aggregates
SELECT 
    category,
    COUNT(*) AS product_count,
    AVG(price) AS avg_price,
    MIN(price) AS min_price,
    MAX(price) AS max_price,
    SUM(stock_quantity) AS total_stock
FROM products
GROUP BY category;

-- 4.3 GROUP BY with HAVING (filtering groups)
-- HAVING is used to filter after aggregation (WHERE filters before)
SELECT 
    country,
    COUNT(*) AS customer_count,
    AVG(age) AS avg_age
FROM customers
GROUP BY country
HAVING COUNT(*) >= 2;

-- 4.4 GROUP BY with ORDER BY
SELECT 
    status,
    COUNT(*) AS order_count,
    SUM(total_amount) AS total_revenue
FROM orders
GROUP BY status
ORDER BY total_revenue DESC;

-- 4.5 Multiple columns in GROUP BY
SELECT 
    country,
    gender,
    COUNT(*) AS count,
    AVG(age) AS avg_age
FROM customers
GROUP BY country, gender
ORDER BY country, gender;

-- ============================================================================
-- PART 5: JOINS - COMBINING TABLES
-- ============================================================================

-- 5.1 INNER JOIN - Only matching records
SELECT 
    o.order_id,
    o.order_date,
    c.first_name,
    c.last_name,
    c.email,
    o.total_amount
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id;

-- 5.2 LEFT JOIN - All from left table, matching from right
SELECT 
    c.customer_id,
    c.first_name,
    c.last_name,
    COUNT(o.order_id) AS order_count,
    COALESCE(SUM(o.total_amount), 0) AS total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name;

-- 5.3 Multiple JOIN
SELECT 
    o.order_id,
    o.order_date,
    c.first_name,
    c.last_name,
    p.product_name,
    oi.quantity,
    oi.unit_price,
    (oi.quantity * oi.unit_price) AS line_total
FROM orders o
INNER JOIN customers c ON o.customer_id = c.customer_id
INNER JOIN order_items oi ON o.order_id = oi.order_id
INNER JOIN products p ON oi.product_id = p.product_id
ORDER BY o.order_id, p.product_name;

-- 5.4 Self JOIN - Compare records within same table
SELECT 
    c1.customer_id AS customer1_id,
    c1.first_name AS customer1_name,
    c2.customer_id AS customer2_id,
    c2.first_name AS customer2_name,
    c1.city
FROM customers c1
INNER JOIN customers c2 ON c1.city = c2.city AND c1.customer_id < c2.customer_id;

-- ============================================================================
-- PART 6: SUBQUERIES - NESTED QUERIES
-- ============================================================================

-- 6.1 Subquery in WHERE clause
SELECT * FROM products
WHERE price > (SELECT AVG(price) FROM products);

-- 6.2 Subquery with IN
SELECT * FROM customers
WHERE customer_id IN (
    SELECT customer_id FROM orders WHERE total_amount > 1000
);

-- 6.3 Correlated subquery
SELECT 
    c.customer_id,
    c.first_name,
    c.last_name,
    (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.customer_id) AS order_count,
    (SELECT SUM(total_amount) FROM orders o WHERE o.customer_id = c.customer_id) AS total_spent
FROM customers c;

-- 6.4 Subquery in FROM clause (Derived table)
SELECT 
    avg_category_price.category,
    avg_category_price.avg_price
FROM (
    SELECT category, AVG(price) AS avg_price
    FROM products
    GROUP BY category
) AS avg_category_price
WHERE avg_category_price.avg_price > 100;

-- ============================================================================
-- PART 7: WINDOW FUNCTIONS - ADVANCED ANALYTICS
-- ============================================================================

-- 7.1 ROW_NUMBER - Assign unique row numbers
SELECT 
    product_id,
    product_name,
    price,
    ROW_NUMBER() OVER (ORDER BY price DESC) AS price_rank
FROM products;

-- 7.2 RANK and DENSE_RANK
SELECT 
    product_id,
    product_name,
    category,
    price,
    RANK() OVER (PARTITION BY category ORDER BY price DESC) AS rank_in_category,
    DENSE_RANK() OVER (PARTITION BY category ORDER BY price DESC) AS dense_rank
FROM products;

-- 7.3 Running totals with SUM
SELECT 
    order_date,
    total_amount,
    SUM(total_amount) OVER (ORDER BY order_date) AS running_total
FROM orders
ORDER BY order_date;

-- 7.4 Moving averages
SELECT 
    order_date,
    total_amount,
    AVG(total_amount) OVER (
        ORDER BY order_date 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg_3_orders
FROM orders
ORDER BY order_date;

-- 7.5 LAG and LEAD - Access previous/next rows
SELECT 
    order_id,
    order_date,
    total_amount,
    LAG(total_amount, 1) OVER (ORDER BY order_date) AS previous_order_amount,
    LEAD(total_amount, 1) OVER (ORDER BY order_date) AS next_order_amount
FROM orders
ORDER BY order_date;

-- ============================================================================
-- PART 8: DATE AND TIME FUNCTIONS - TEMPORAL ANALYSIS
-- ============================================================================

-- 8.1 Extract date parts
SELECT 
    order_date,
    YEAR(order_date) AS year,
    MONTH(order_date) AS month,
    DAY(order_date) AS day,
    DAYNAME(order_date) AS day_name,
    QUARTER(order_date) AS quarter
FROM orders;

-- 8.2 Date calculations
SELECT 
    customer_id,
    registration_date,
    DATEDIFF(CURDATE(), registration_date) AS days_since_registration,
    DATE_ADD(registration_date, INTERVAL 1 YEAR) AS one_year_anniversary
FROM customers;

-- 8.3 Time-based aggregations
SELECT 
    YEAR(order_date) AS year,
    MONTH(order_date) AS month,
    COUNT(*) AS order_count,
    SUM(total_amount) AS monthly_revenue
FROM orders
GROUP BY YEAR(order_date), MONTH(order_date)
ORDER BY year, month;

-- 8.4 Current date/time functions
SELECT 
    NOW() AS current_datetime,
    CURDATE() AS current_date,
    CURTIME() AS current_time,
    UNIX_TIMESTAMP() AS unix_timestamp;

-- ============================================================================
-- PART 9: STRING FUNCTIONS - TEXT MANIPULATION
-- ============================================================================

-- 9.1 Basic string functions
SELECT 
    first_name,
    last_name,
    CONCAT(first_name, ' ', last_name) AS full_name,
    UPPER(first_name) AS uppercase_name,
    LOWER(email) AS lowercase_email,
    LENGTH(email) AS email_length
FROM customers;

-- 9.2 Substring and position
SELECT 
    email,
    SUBSTRING(email, 1, LOCATE('@', email) - 1) AS username,
    SUBSTRING_INDEX(email, '@', -1) AS domain
FROM customers;

-- 9.3 Trimming and replacing
SELECT 
    product_name,
    TRIM(product_name) AS trimmed_name,
    REPLACE(product_name, 'Pro', 'Professional') AS replaced_name
FROM products;

-- ============================================================================
-- PART 10: CASE STATEMENTS - CONDITIONAL LOGIC
-- ============================================================================

-- 10.1 Simple CASE statement
SELECT 
    product_name,
    price,
    CASE 
        WHEN price < 50 THEN 'Budget'
        WHEN price BETWEEN 50 AND 500 THEN 'Mid-range'
        ELSE 'Premium'
    END AS price_category
FROM products;

-- 10.2 CASE for business logic
SELECT 
    customer_id,
    first_name,
    age,
    CASE 
        WHEN age < 25 THEN 'Young Adult'
        WHEN age BETWEEN 25 AND 40 THEN 'Adult'
        WHEN age BETWEEN 41 AND 60 THEN 'Middle Age'
        ELSE 'Senior'
    END AS age_group
FROM customers;

-- 10.3 CASE with aggregation
SELECT 
    status,
    COUNT(*) AS total_orders,
    SUM(CASE WHEN total_amount > 500 THEN 1 ELSE 0 END) AS high_value_orders,
    SUM(CASE WHEN total_amount <= 500 THEN 1 ELSE 0 END) AS low_value_orders
FROM orders
GROUP BY status;

-- ============================================================================
-- PART 11: COMMON TABLE EXPRESSIONS (CTE) - READABLE QUERIES
-- ============================================================================

-- 11.1 Basic CTE
WITH customer_orders AS (
    SELECT 
        customer_id,
        COUNT(*) AS order_count,
        SUM(total_amount) AS total_spent
    FROM orders
    GROUP BY customer_id
)
SELECT 
    c.first_name,
    c.last_name,
    co.order_count,
    co.total_spent
FROM customers c
LEFT JOIN customer_orders co ON c.customer_id = co.customer_id;

-- 11.2 Multiple CTEs
WITH 
monthly_sales AS (
    SELECT 
        DATE_FORMAT(order_date, '%Y-%m') AS month,
        SUM(total_amount) AS monthly_revenue
    FROM orders
    WHERE status = 'Delivered'
    GROUP BY DATE_FORMAT(order_date, '%Y-%m')
),
avg_monthly AS (
    SELECT AVG(monthly_revenue) AS avg_revenue
    FROM monthly_sales
)
SELECT 
    ms.month,
    ms.monthly_revenue,
    am.avg_revenue,
    ms.monthly_revenue - am.avg_revenue AS variance_from_avg
FROM monthly_sales ms
CROSS JOIN avg_monthly am
ORDER BY ms.month;

-- ============================================================================
-- PART 12: REAL-WORLD ANALYTICAL QUERIES
-- ============================================================================

-- 12.1 Customer Lifetime Value (CLV)
SELECT 
    c.customer_id,
    c.first_name,
    c.last_name,
    COUNT(o.order_id) AS total_orders,
    SUM(o.total_amount) AS lifetime_value,
    AVG(o.total_amount) AS avg_order_value,
    MAX(o.order_date) AS last_order_date,
    DATEDIFF(CURDATE(), MAX(o.order_date)) AS days_since_last_order
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
ORDER BY lifetime_value DESC;

-- 12.2 Product Performance Analysis
SELECT 
    p.product_id,
    p.product_name,
    p.category,
    p.price,
    COUNT(oi.order_item_id) AS times_ordered,
    SUM(oi.quantity) AS total_quantity_sold,
    SUM(oi.quantity * oi.unit_price) AS total_revenue,
    AVG(oi.quantity) AS avg_quantity_per_order
FROM products p
LEFT JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.product_name, p.category, p.price
ORDER BY total_revenue DESC;

-- 12.3 Cohort Analysis - Customer retention by registration month
SELECT 
    DATE_FORMAT(c.registration_date, '%Y-%m') AS cohort_month,
    COUNT(DISTINCT c.customer_id) AS total_customers,
    COUNT(DISTINCT CASE WHEN o.order_date IS NOT NULL THEN c.customer_id END) AS active_customers,
    ROUND(COUNT(DISTINCT CASE WHEN o.order_date IS NOT NULL THEN c.customer_id END) * 100.0 / COUNT(DISTINCT c.customer_id), 2) AS retention_rate
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id 
GROUP BY DATE_FORMAT(c.registration_date, '%Y-%m')
ORDER BY cohort_month;

-- 12.4 RFM Analysis (Recency, Frequency, Monetary)
WITH rfm_calc AS (
    SELECT 
        c.customer_id,
        c.first_name,
        c.last_name,
        DATEDIFF(CURDATE(), MAX(o.order_date)) AS recency,
        COUNT(o.order_id) AS frequency,
        SUM(o.total_amount) AS monetary
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, c.first_name, c.last_name
)
SELECT 
    customer_id,
    first_name,
    last_name,
    recency,
    frequency,
    monetary,
    CASE 
        WHEN recency <= 30 AND frequency >= 3 AND monetary >= 1000 THEN 'VIP'
        WHEN recency <= 60 AND frequency >= 2 AND monetary >= 500 THEN 'Loyal'
        WHEN recency <= 90 THEN 'Active'
        WHEN recency > 90 AND frequency >= 1 THEN 'At Risk'
        ELSE 'Lost'
    END AS customer_segment
FROM rfm_calc
ORDER BY monetary DESC;

-- 12.5 Sales funnel analysis
SELECT 
    'Total Customers' AS stage,
    COUNT(*) AS count,
    100.0 AS percentage
FROM customers
UNION ALL
SELECT 
    'Customers with Orders',
    COUNT(DISTINCT customer_id),
    ROUND(COUNT(DISTINCT customer_id) * 100.0 / (SELECT COUNT(*) FROM customers), 2)
FROM orders
UNION ALL
SELECT 
    'Delivered Orders',
    COUNT(DISTINCT customer_id),
    ROUND(COUNT(DISTINCT customer_id) * 100.0 / (SELECT COUNT(*) FROM customers), 2)
FROM orders
WHERE status = 'Delivered';

-- 12.6 Year-over-Year growth
WITH yearly_sales AS (
    SELECT 
        YEAR(order_date) AS year,
        SUM(total_amount) AS annual_revenue
    FROM orders
    WHERE status = 'Delivered'
    GROUP BY YEAR(order_date)
)
SELECT 
    current_year.year,
    current_year.annual_revenue AS current_revenue,
    previous_year.annual_revenue AS previous_revenue,
    ROUND((current_year.annual_revenue - previous_year.annual_revenue) * 100.0 / previous_year.annual_revenue, 2) AS yoy_growth_percentage
FROM yearly_sales current_year
LEFT JOIN yearly_sales previous_year ON current_year.year = previous_year.year + 1
ORDER BY current_year.year;

-- ============================================================================
-- PART 13: PERFORMANCE OPTIMIZATION
-- ============================================================================

-- 13.1 Create indexes for faster queries
CREATE INDEX idx_customer_country ON customers(country);
CREATE INDEX idx_order_date ON orders(order_date);
CREATE INDEX idx_order_customer ON orders(customer_id);
CREATE INDEX idx_product_category ON products(category);

-- 13.2 Analyze query performance
EXPLAIN SELECT * FROM customers WHERE country = 'USA';
EXPLAIN SELECT * FROM orders WHERE order_date > '2024-01-01';

-- 13.3 Use EXPLAIN to see query execution plan
EXPLAIN SELECT 
    c.first_name,
    COUNT(o.order_id) AS order_count
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.first_name;

-- ============================================================================
-- PART 14: DATA QUALITY CHECKS
-- ============================================================================

-- 14.1 Find NULL values
SELECT 
    COUNT(*) AS total_records,
    SUM(CASE WHEN email IS NULL THEN 1 ELSE 0 END) AS null_emails,
    SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END) AS null_ages
FROM customers;

-- 14.2 Find duplicates
SELECT email, COUNT(*) AS duplicate_count
FROM customers
GROUP BY email
HAVING COUNT(*) > 1;

-- 14.3 Data validation
SELECT 
    'Invalid Email' AS issue,
    COUNT(*) AS count
FROM customers
WHERE email NOT LIKE '%@%.%'
UNION ALL
SELECT 
    'Negative Price',
    COUNT(*)
FROM products
WHERE price < 0
UNION ALL
SELECT 
    'Future Order Date',
    COUNT(*)
FROM orders
WHERE order_date > CURDATE();

-- ============================================================================
-- PART 15: EXPORT AND REPORTING
-- ============================================================================

-- 15.1 Create summary report
SELECT 
    'Total Revenue' AS metric,
    CONCAT('$', FORMAT(SUM(total_amount), 2)) AS value
FROM orders WHERE status = 'Delivered'
UNION ALL
SELECT 
    'Total Orders',
    FORMAT(COUNT(*), 0)
FROM orders
UNION ALL
SELECT 
    'Average Order Value',
    CONCAT('$', FORMAT(AVG(total_amount), 2))
FROM orders
UNION ALL
SELECT 
    'Total Customers',
    FORMAT(COUNT(*), 0)
FROM customers;

-- ============================================================================
-- KEY TAKEAWAYS FOR DATA ANALYSIS
-- ============================================================================

/*
BEST PRACTICES:

1. ALWAYS start with exploratory queries (COUNT, SUM, AVG)
2. Use JOINS to combine related data
3. GROUP BY for categorical analysis
4. Use CASE for segmentation
5. CTEs make complex queries readable
6. Window functions for rankings and running totals
7. Index frequently queried columns
8. Use EXPLAIN to optimize slow queries
9. Validate data quality regularly
10. Document your queries with comments

COMMON ANALYSIS TYPES:
- Trend Analysis: Time-based aggregations
- Cohort Analysis: Group performance over time
- Funnel Analysis: Conversion rates
- RFM Analysis: Customer segmentation
- Product Performance: Sales by product/category
- Customer Lifetime Value: Total customer worth

PERFORMANCE TIPS:
- Limit rows when testing (LIMIT)
- Use WHERE before GROUP BY
- Avoid SELECT * in production
- Create indexes on foreign keys
- Use EXPLAIN for optimization
- Consider partitioning large tables
*/