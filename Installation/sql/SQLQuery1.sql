-- 1. List all databases on the server
SELECT name FROM sys.databases;
-- This query lists all databases available on this SQL Server instance.

-- 2. Create a new database named 'TestDB'
CREATE DATABASE TestDB;
-- This command creates the new database 'TestDB'.

-- 3. Verify database creation by listing the databases again and filtering for 'TestDB'
SELECT name FROM sys.databases WHERE name = 'TestDB';
-- Confirms that 'TestDB' now exists.

-- 4. Change the context to the new database 'TestDB'
USE TestDB;
-- Directs subsequent operations to occur in 'TestDB'.

-- 5. List all tables in the current database (will be empty initially)
SELECT name FROM sys.tables;
-- This query shows that no tables exist yet in 'TestDB'.

---------------------------------------------------------------------
-- Note on Renaming:
-- SQL Server supports table renaming but does not have a direct RENAME DATABASE command.
---------------------------------------------------------------------

-- 6. Create a new table called 'Employees'
CREATE TABLE Employees (
    ID INT IDENTITY(1,1) PRIMARY KEY,  -- Auto-incrementing primary key
    Name NVARCHAR(100),               -- Employee name column
    Age INT                         -- Employee age column
);
-- This command creates the 'Employees' table with three columns.

-- 7. List tables in 'TestDB' to confirm the creation of 'Employees'
SELECT name FROM sys.tables;
-- The result should now show the 'Employees' table exists.

-- 8. Insert multiple records into the 'Employees' table
INSERT INTO Employees (Name, Age)
VALUES 
    ('Alice', 30),
    ('Bob', 25),
    ('Charlie', 28);
-- This inserts three records into the table.

-- 9. Alter the 'Employees' table by adding a new column 'Department'
ALTER TABLE Employees
ADD Department NVARCHAR(50);
-- The new column 'Department' will later store the employee's department.

-- 10. Update multiple records: Set 'Department' to 'Sales' for employees younger than 30
UPDATE Employees
SET Department = 'Sales'
WHERE Age < 30;
-- Updates the Department for employees with Age less than 30.

-- 11. Update records to "rename" a data value: Change 'Bob' to 'Robert'
UPDATE Employees
SET Name = 'Robert'
WHERE Name = 'Bob';
-- This command changes the Name from 'Bob' to 'Robert'.

-- 12. Display all records from the 'Employees' table to verify the changes
SELECT * FROM Employees;
-- Shows the current data stored in the 'Employees' table.

-- 13. Rename the table 'Employees' to 'Staff'
EXEC sp_rename 'Employees', 'Staff';
-- Renames the table from 'Employees' to 'Staff' while retaining all data.

-- 14. List the tables again to verify the table has been renamed
SELECT name FROM sys.tables;
-- Confirms that 'Staff' now exists and 'Employees' is no longer listed.

-- 15. Display all records from the renamed table 'Staff'
SELECT * FROM Staff;
-- Verifies the data in the renamed table.

-- 16. Delete multiple records: Remove rows where Age is greater than 30
DELETE FROM Staff
WHERE Age > 30;
-- Deletes records in 'Staff' that fulfill the specified condition.

-- 17. Display the resulting records from 'Staff'
SELECT * FROM Staff;
-- Shows the remaining data after deletion.

-- 18. Drop the table 'Staff'
DROP TABLE Staff;
-- Permanently removes the 'Staff' table and its data from the database.

-- 19. Switch back to the master database before dropping 'TestDB'
USE master;
-- Changes the context from 'TestDB' to the master database.

-- 20. Drop the entire database 'TestDB'
DROP DATABASE TestDB;
-- This command permanently drops the 'TestDB' database and all its objects.
