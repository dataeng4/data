# Simple ETL Pipeline for Multiple Databases

This project implements a simple ETL (Extract, Load) pipeline in Python (`etl_pipeline.py`) that connects to MSSQL, Oracle, PostgreSQL, and MySQL databases, extracts data using SQL queries defined in a single `config.yaml` file, and saves results to Excel files without transformations.

## Features
- Supports MSSQL, Oracle, PostgreSQL, and MySQL databases.
- Single YAML configuration (`config.yaml`) for all database settings, queries, and output paths.
- Retry logic for robust connection and query execution.
- Command-line interface to run a single database or all databases.
- Logging to file (`etl_pipeline.log`) and console for debugging.

## Prerequisites
- Python 3.9+
- Oracle Instant Client (for Oracle)
- ODBC Driver 17 for SQL Server (for MSSQL)
- PostgreSQL and MySQL servers
- Conda environment (`etl_env`)

## Installation

1. Create and activate a Conda environment:
   ```bash
   conda create -n etl_env python=3.9
   conda activate etl_env
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Configuration
Edit `config.yaml` to specify database connections, SQL queries, and output Excel file paths. The file consolidates settings for all databases in a single YAML structure.

Example `config.yaml`:
```yaml
databases:
  mssql:
    database:
      type: mssql
      host: INSTANCE-202504\SQLEXPRESS
      port: ''
      name: inventory_db
      user: sa
      password: 1234
    query: SELECT * FROM products WHERE stock > 0
    output:
      file_path: output/products_mssql.xlsx
  oracle:
    database:
      type: oracle
      host: localhost
      port: 1521
      name: FREEPDB1
      user: analytics_db
      password: analytics123
    query: SELECT * FROM customers WHERE total_purchases > 1000
    output:
      file_path: output/high_value_customers_oracle.xlsx
  postgresql:
    database:
      type: postgresql
      host: localhost
      port: 5432
      name: test_db
      user: postgres
      password: 1234
    query: SELECT * FROM employees
    output:
      file_path: output/employees_postgresql.xlsx
  mysql:
    database:
      type: mysql
      host: localhost
      port: 3306
      name: sales_db
      user: root
      password: 1234
    query: SELECT * FROM orders
    output:
      file_path: output/orders_mysql.xlsx
```

## Usage
Run `etl_pipeline.py` using command-line arguments to specify the config file and database.

1. **Run for a single database** (e.g., MSSQL):
   ```bash
   python etl_pipeline.py --config config.yaml --database mssql
   ```
   Replace `mssql` with `oracle`, `postgresql`, or `mysql` as needed.

2. **Run for all databases**:
   ```bash
   python etl_pipeline.py --config config.yaml
   ```

## Output
- Excel files in the `output/` directory:
  - `products_mssql.xlsx`: MSSQL `products` data.
  - `high_value_customers_oracle.xlsx`: Oracle `customers` data.
  - `employees_postgresql.xlsx`: PostgreSQL `employees` data.
  - `orders_mysql.xlsx`: MySQL `orders` data.
- Logs in `etl_pipeline.log` for execution details.

## Troubleshooting
- **MySQL Error**: If `'cryptography' package is required`, ensure it’s installed:
  ```bash
  pip install cryptography
  ```
  Alternatively, set the MySQL `root` user to `mysql_native_password`:
  ```sql
  ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '1234';
  FLUSH PRIVILEGES;
  ```
- **Table Errors**: Verify `employees` (PostgreSQL) and `orders` (MySQL) exist. Create if needed:
  ```sql
  -- PostgreSQL
  CREATE TABLE employees (
      employee_id SERIAL PRIMARY KEY,
      name VARCHAR(100),
      department VARCHAR(50),
      salary NUMERIC
  );
  -- MySQL
  CREATE TABLE orders (
      order_id INT PRIMARY KEY,
      customer_id INT,
      amount DECIMAL(10,2),
      order_date DATE
  );
  ```
- **Connection Issues**: Check server status, credentials, and drivers.
- **Logs**: Review `etl_pipeline.log` for errors.

## Project Structure
```
etl-pipeline/
│
├── etl_pipeline.py          # Main ETL Python script
├── logging_config.yaml      # Logging configuration
├── config.yaml              # Consolidated config for all databases
├── output/                  # Output Excel files
├── etl_pipeline.log         # Log file
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
```

