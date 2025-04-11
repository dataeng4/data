-- (Optional) List all databases on the PostgreSQL server.
-- In psql you can use the meta command:
\l
-- Or, using an SQL query:
SELECT datname FROM pg_database;

-- 1. Create a new database named 'testdb'
CREATE DATABASE testdb;
-- This command creates the database 'testdb'.

-- 2. Connect to the new database (in psql use the following meta command):
\c testdb
-- Now you are connected to 'testdb' and subsequent commands run in its context.

-- 3. List tables in the current database (it should be empty initially)
\dt
-- Alternatively, you can query:
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public';

-- 4. Create a table called 'employees'
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,   -- The SERIAL type auto-increments and serves as a primary key.
    name VARCHAR(100),       -- Employee name column.
    age INT                  -- Employee age column.
);
-- The 'employees' table is now created in the 'public' schema.

-- 5. Confirm the table creation by listing tables
\dt

-- 6. Insert multiple records into the 'employees' table.
INSERT INTO employees (name, age)
VALUES 
    ('Alice', 30),
    ('Bob', 25),
    ('Charlie', 28);
-- Inserts three rows into the table.

-- 7. Alter the table to add a new column 'department'
ALTER TABLE employees ADD COLUMN department VARCHAR(50);
-- The 'department' column will be used to store additional information.

-- 8. Update records: Set the department to 'Sales' for employees under age 30
UPDATE employees
SET department = 'Sales'
WHERE age < 30;
-- Updates the 'department' field for the qualifying employees.

-- 9. Update a record value: Change the name 'Bob' to 'Robert'
UPDATE employees
SET name = 'Robert'
WHERE name = 'Bob';
-- Renames 'Bob' to 'Robert'.

-- 10. Display all records from the 'employees' table
SELECT * FROM employees;
-- This query shows the current data in the table.

-- 11. Rename the table 'employees' to 'staff'
ALTER TABLE employees RENAME TO staff;
-- This renames the table while preserving its data.

-- 12. List tables again to verify the new table name
\dt

-- 13. Display records from the renamed table 'staff'
SELECT * FROM staff;
-- Confirms that the data still exists after renaming.

-- 14. Delete records from 'staff' where the employee age is greater than 30
DELETE FROM staff
WHERE age > 30;
-- This deletes any rows meeting the condition (if any).

-- 15. Display the remaining records from 'staff'
SELECT * FROM staff;
-- Shows the data after deletion.

-- 16. Drop the table 'staff'
DROP TABLE staff;
-- Permanently removes the 'staff' table from the database.

-- 17. Disconnect from 'testdb' and connect to the default 'postgres' database to drop 'testdb'
\c postgres
-- Now you're connected to the 'postgres' database.

-- 18. Drop the previously created database 'testdb'
DROP DATABASE testdb;
-- This removes the entire 'testdb' database and all its objects.
