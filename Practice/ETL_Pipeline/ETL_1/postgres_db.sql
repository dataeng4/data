-- Create database
CREATE DATABASE test_db;

-- Connect to the database
\c test_db

-- Create a sample table
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    department VARCHAR(50),
    salary DECIMAL(10, 2)
);

-- Insert sample data
INSERT INTO employees (first_name, last_name, department, salary) VALUES
('John', 'Doe', 'Engineering', 75000.00),
('Jane', 'Smith', 'Marketing', 65000.00),
('Mike', 'Johnson', 'Sales', 70000.00),
('Emily', 'Davis', 'HR', 60000.00);