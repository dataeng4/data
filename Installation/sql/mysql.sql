-- (Optional) 0. List all databases on the server
SHOW DATABASES;
-- This command displays all databases available on your MySQL server.

-- 1. Create a new database named 'TestDB'
CREATE DATABASE TestDB;
-- Creates a new database called TestDB.

-- 2. List all databases again to verify that 'TestDB' has been created
SHOW DATABASES;

-- 3. Select the database 'TestDB' for subsequent operations
USE TestDB;
-- Directs subsequent commands to operate within TestDB.

-- 4. List tables in 'TestDB' (should be empty initially)
SHOW TABLES;
-- Since no tables exist yet, this will return an empty result set.

----------------------------------------------------------------------
-- Note: MySQL does not support a direct RENAME DATABASE command.
-- To rename a database, you typically create a new database,
-- migrate the data/objects, and then drop the old database.
----------------------------------------------------------------------

-- 5. Create a table called 'employees'
CREATE TABLE employees (
    id INT AUTO_INCREMENT PRIMARY KEY,  -- Unique auto-incrementing ID
    name VARCHAR(100),                   -- Employee name
    age INT                             -- Employee age
);
-- This creates the 'employees' table with three columns.

-- 6. List tables in 'TestDB' to confirm 'employees' was created
SHOW TABLES;
-- Confirms the creation of the 'employees' table.

-- 7. Insert multiple records into the 'employees' table
INSERT INTO employees (name, age) VALUES 
    ('Alice', 30),
    ('Bob', 25),
    ('Charlie', 28);
-- Inserts three employee records.

-- 8. Alter the table by adding a new column 'department'
ALTER TABLE employees ADD COLUMN department VARCHAR(50);
-- Adds a column to store each employee's department information.

-- 9. Update multiple records: set the department to 'Sales' for employees under age 30
UPDATE employees
SET department = 'Sales'
WHERE age < 30;
-- Updates the department for qualifying employees.

-- 10. Update records to "rename" a data value: change the name 'Bob' to 'Robert'
UPDATE employees
SET name = 'Robert'
WHERE name = 'Bob';
-- Changes 'Bob' to 'Robert' in the employees table.

-- 11. Display all records from the 'employees' table
SELECT * FROM employees;
-- Shows the current data in the employees table.

-- 12. Rename the table 'employees' to 'staff'
RENAME TABLE employees TO staff;
-- Renames the table while preserving its data.

-- 13. List tables again to verify the table has been renamed to 'staff'
SHOW TABLES;
-- Confirms that 'staff' now exists instead of 'employees'.

-- 14. Display records from the renamed table 'staff'
SELECT * FROM staff;
-- Verifies the data in the renamed table.

-- 15. Delete multiple records: remove rows where the age is greater than 30
DELETE FROM staff
WHERE age > 30;
-- Deletes any records with an age greater than 30.

-- 16. Display the remaining records from 'staff'
SELECT * FROM staff;
-- Shows the data after deletion.

-- 17. Drop the table 'staff'
DROP TABLE staff;
-- Permanently removes the 'staff' table and its data.

-- 18. Drop the entire database 'TestDB'
DROP DATABASE TestDB;
-- Deletes the database and all its contained objects.
