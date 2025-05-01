CREATE DATABASE sales_db;
USE sales_db;
CREATE TABLE orders (
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_name VARCHAR(100),
    product VARCHAR(50),
    quantity INT,
    order_date DATE
);
INSERT INTO orders (customer_name, product, quantity, order_date) VALUES
('Alice Brown', 'Laptop', 2, '2025-01-15'),
('Bob Wilson', 'Phone', 5, '2025-02-10'),
('Charlie Lee', 'Tablet', 3, '2025-03-20');