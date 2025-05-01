-- Create database
CREATE DATABASE inventory_db;
GO

-- Use the database
USE inventory_db;
GO

-- Create products table
CREATE TABLE products (
    product_id INT PRIMARY KEY IDENTITY(1,1),
    name VARCHAR(100),
    category VARCHAR(50),
    stock INT,
    price DECIMAL(10, 2)
);
GO

-- Insert sample data
INSERT INTO products (name, category, stock, price) VALUES
('Laptop', 'Electronics', 25, 999.99),
('Desk Chair', 'Furniture', 50, 149.99),
('Notebook', 'Stationery', 100, 2.99),
('Smartphone', 'Electronics', 0, 699.99);
GO

SELECT * FROM products;
GO