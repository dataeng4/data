**ETL PipeLine 1**
ETL Pipeline for Multiple Databases
This project implements an ETL (Extract, Transform, Load) pipeline that connects to MSSQL, Oracle, PostgreSQL, and MySQL databases, extracts data using SQL queries, and saves results to Excel files.
Features

Supports MSSQL, Oracle, PostgreSQL, and MySQL databases.
Configurable via YAML files for each database
Logging to file and console.

Prerequisites

Python 3.9+
Oracle Instant Client (for Oracle)
ODBC Driver 17 for SQL Server (for MSSQL)
PostgreSQL and MySQL servers
Conda environment (recommended)



Create and activate a Conda environment:conda create -n etl_env python=3.9
conda activate etl_env


Configuration
Edit the YAML config files in the project root:

mssql_config.yaml: MSSQL settings.
oracle_config.yaml: Oracle settings.
postgresql_config.yaml: PostgreSQL settings.
mysql_config.yaml: MySQL settings.

Example (mssql_config.yaml):
database:
  type: mssql
  host: INSTANCE-202504\SQLEXPRESS
  port: ''
  name: inventory_db
  user: sa
  password: 1234
query: SELECT * FROM products WHERE stock > 0
output:
  file_path: output/products_transformed.xlsx

Usage

Run ETL pipeline:python etl_pipeline.py


Uses the config file specified in etl_pipeline.py (Eg; oracle_config.yaml).




Output

Excel files in the output/ directory (e.g., products_transformed.xlsx, high_value_customers_transformed.xlsx).
Logs in etl_pipeline.log.

Project Structure
etl-pipeline/
│
├── etl_pipeline.py           # Main ETL script
├── logging_config.yaml      # Logging configuration
├── mssql_config.yaml        # MSSQL config
├── oracle_config.yaml       # Oracle config
├── postgresql_config.yaml   # PostgreSQL config
├── mysql_config.yaml        # MySQL config
├── output/                  # Output Excel files
├── etl_pipeline.log         # Log file
├── README.md                # Project documentation

