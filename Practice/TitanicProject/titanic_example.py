import pandas as pd
import requests
import io
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col as pyspark_col, when
from sqlalchemy import create_engine
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table

# Create output directories
base_dir = "C:/Users/lokesh/Desktop/data/Practice/TitanicProject"
os.makedirs(f"{base_dir}/Pandas", exist_ok=True)
os.makedirs(f"{base_dir}/PySpark", exist_ok=True)
os.makedirs(f"{base_dir}/SQL", exist_ok=True)

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# Download raw CSV
response = requests.get(url)
with open(f"{base_dir}/titanic_raw.csv", "w", encoding="utf-8") as f:
    f.write(response.text)
print("Downloaded titanic_raw.csv")

# --- Pandas Section ---
df = pd.read_csv(io.StringIO(response.text))
print("Pandas: Loaded DataFrame from URL")
print(df.head())

print("Pandas: Null counts per column:")
print(df.isna().sum())
print("Pandas: Checking for ?, -, empty strings...")
for col in df.columns:
    if df[col].dtype == "object":
        questionable_values = df[col].isin(["?", "-", ""]).sum()
        if questionable_values > 0:
            print(f"Column {col}: {questionable_values} ?, -, or empty values")
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
string_cols = df.select_dtypes(include=["object"]).columns
df[numeric_cols] = df[numeric_cols].fillna(0).replace(["?", "-"], 0)
df[string_cols] = df[string_cols].fillna("Unknown").replace(["?", "-", ""], "Unknown")
print("Pandas: Replaced nulls, ?, -, empty with 0 (numeric), Unknown (string)")
print("Pandas: Null counts after cleaning:")
print(df.isna().sum())

# Export Pandas files
df.to_csv(f"{base_dir}/Pandas/titanic_pandas_cleaned.csv", index=False)
print("Pandas: Exported cleaned data to Pandas/titanic_pandas_cleaned.csv")
df.to_json(f"{base_dir}/Pandas/titanic_pandas_cleaned.json", orient="records", lines=True)
print("Pandas: Exported cleaned data to Pandas/titanic_pandas_cleaned.json")
df.to_excel(f"{base_dir}/Pandas/titanic_pandas_cleaned.xlsx", index=False)
print("Pandas: Exported cleaned data to Pandas/titanic_pandas_cleaned.xlsx")
pdf = SimpleDocTemplate(f"{base_dir}/Pandas/titanic_pandas_cleaned.pdf", pagesize=letter)
table = Table([df.columns.tolist()] + df.values.tolist())
pdf.build([table])
print("Pandas: Exported cleaned data to Pandas/titanic_pandas_cleaned.pdf")

# --- PySpark Section ---
os.environ["HADOOP_HOME"] = "C:\\hadoop"
spark = SparkSession.builder.appName("TitanicExample").getOrCreate()

spark_df = spark.read.csv(f"{base_dir}/titanic_raw.csv", header=True, inferSchema=True)
print("PySpark: Loaded DataFrame from local file")
spark_df.show(5)

print("PySpark: Null counts per column:")
null_counts = spark_df.select([pyspark_col(c).isNull().cast("int").alias(c) for c in spark_df.columns]).groupBy().sum()
null_counts.show()

print("PySpark: Checking for ?, -, empty strings...")
for c in spark_df.columns:
    if spark_df.schema[c].dataType.simpleString() in ["string"]:
        counts = spark_df.filter(pyspark_col(c).isin("?", "-", "")).count()
        if counts > 0:
            print(f"Column {c}: {counts} ?, -, or empty values")

numeric_cols = [c for c, t in spark_df.dtypes if t in ["int", "double"]]
string_cols = [c for c, t in spark_df.dtypes if t == "string"]
for c in numeric_cols:
    spark_df = spark_df.fillna({c: 0})
for c in string_cols:
    spark_df = spark_df.fillna({c: "Unknown"})
    spark_df = spark_df.withColumn(c, when(pyspark_col(c).isin("?", "-", ""), "Unknown").otherwise(pyspark_col(c)))
print("PySpark: Replaced nulls, ?, -, empty with 0 (numeric), Unknown (string)")

print("PySpark: Null counts after cleaning:")
null_counts_cleaned = spark_df.select([pyspark_col(c).isNull().cast("int").alias(c) for c in spark_df.columns]).groupBy().sum()
null_counts_cleaned.show()

# Export PySpark files
pandas_df = spark_df.toPandas()
pandas_df.to_csv(f"{base_dir}/PySpark/titanic_spark_cleaned.csv", index=False)
print("PySpark: Exported cleaned data to PySpark/titanic_spark_cleaned.csv")
pandas_df.to_json(f"{base_dir}/PySpark/titanic_spark_cleaned.json", orient="records", lines=True)
print("PySpark: Exported cleaned data to PySpark/titanic_spark_cleaned.json")
pandas_df.to_parquet(f"{base_dir}/PySpark/titanic_spark_cleaned.parquet")
print("PySpark: Exported cleaned data to PySpark/titanic_spark_cleaned.parquet")
pdf = SimpleDocTemplate(f"{base_dir}/PySpark/titanic_spark_cleaned.pdf", pagesize=letter)
table = Table([pandas_df.columns.tolist()] + pandas_df.values.tolist())
pdf.build([table])
print("PySpark: Exported cleaned data to PySpark/titanic_spark_cleaned.pdf")

spark.stop()
print("PySpark: Session stopped")

# --- MS SQL Server Section ---
server = "INSTANCE-202504\\SQLEXPRESS"
database = "TitanicDB"
connection_string = f"mssql+pyodbc://{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes&encrypt=no"
engine = create_engine(connection_string)

# Load raw data as Titanic table
df_raw = pd.read_csv(f"{base_dir}/titanic_raw.csv")
df_raw.to_sql("Titanic", engine, if_exists="replace", index=False)
print("SQL Server: Loaded raw data into 'Titanic' table")

# Load cleaned data as TitanicCleaned table
df.to_sql("TitanicCleaned", engine, if_exists="replace", index=False)
print("SQL Server: Loaded cleaned data into 'TitanicCleaned' table")

null_query = """
SELECT
    SUM(CASE WHEN PassengerId IS NULL THEN 1 ELSE 0 END) AS PassengerId_nulls,
    SUM(CASE WHEN Survived IS NULL THEN 1 ELSE 0 END) AS Survived_nulls,
    SUM(CASE WHEN Pclass IS NULL THEN 1 ELSE 0 END) AS Pclass_nulls,
    SUM(CASE WHEN Name IS NULL THEN 1 ELSE 0 END) AS Name_nulls,
    SUM(CASE WHEN Sex IS NULL THEN 1 ELSE 0 END) AS Sex_nulls,
    SUM(CASE WHEN Age IS NULL THEN 1 ELSE 0 END) AS Age_nulls,
    SUM(CASE WHEN SibSp IS NULL THEN 1 ELSE 0 END) AS SibSp_nulls,
    SUM(CASE WHEN Parch IS NULL THEN 1 ELSE 0 END) AS Parch_nulls,
    SUM(CASE WHEN Ticket IS NULL THEN 1 ELSE 0 END) AS Ticket_nulls,
    SUM(CASE WHEN Fare IS NULL THEN 1 ELSE 0 END) AS Fare_nulls,
    SUM(CASE WHEN Cabin IS NULL THEN 1 ELSE 0 END) AS Cabin_nulls,
    SUM(CASE WHEN Embarked IS NULL THEN 1 ELSE 0 END) AS Embarked_nulls
FROM TitanicCleaned
"""
null_counts = pd.read_sql(null_query, engine)
print("SQL Server: Null counts in 'TitanicCleaned':")
print(null_counts)

check_query = """
SELECT
    SUM(CASE WHEN Name IN ('?', '-', '') THEN 1 ELSE 0 END) AS Name_issues,
    SUM(CASE WHEN Sex IN ('?', '-', '') THEN 1 ELSE 0 END) AS Sex_issues,
    SUM(CASE WHEN Ticket IN ('?', '-', '') THEN 1 ELSE 0 END) AS Ticket_issues,
    SUM(CASE WHEN Cabin IN ('?', '-', '') THEN 1 ELSE 0 END) AS Cabin_issues,
    SUM(CASE WHEN Embarked IN ('?', '-', '') THEN 1 ELSE 0 END) AS Embarked_issues
FROM TitanicCleaned
"""
issue_counts = pd.read_sql(check_query, engine)
print("SQL Server: Counts of ?, -, empty strings:")
print(issue_counts)

# Export SQL files
sql_df = pd.read_sql("SELECT * FROM TitanicCleaned", engine)
sql_df.to_csv(f"{base_dir}/SQL/titanic_sql_cleaned.csv", index=False)
print("SQL Server: Exported cleaned data to SQL/titanic_sql_cleaned.csv")
sql_df.to_json(f"{base_dir}/SQL/titanic_sql_cleaned.json", orient="records", lines=True)
print("SQL Server: Exported cleaned data to SQL/titanic_sql_cleaned.json")
sql_df.to_excel(f"{base_dir}/SQL/titanic_sql_cleaned.xlsx", index=False)
print("SQL Server: Exported cleaned data to SQL/titanic_sql_cleaned.xlsx")
pdf = SimpleDocTemplate(f"{base_dir}/SQL/titanic_sql_cleaned.pdf", pagesize=letter)
table = Table([sql_df.columns.tolist()] + sql_df.values.tolist())
pdf.build([table])
print("SQL Server: Exported cleaned data to SQL/titanic_sql_cleaned.pdf")

# --- Bonus: Import Cleaned Formats ---
print("\n--- Importing Cleaned Formats ---")

df_csv = pd.read_csv(f"{base_dir}/Pandas/titanic_pandas_cleaned.csv")
print("Pandas: Imported Pandas/titanic_pandas_cleaned.csv")
df_json = pd.read_json(f"{base_dir}/Pandas/titanic_pandas_cleaned.json", lines=True)
print("Pandas: Imported Pandas/titanic_pandas_cleaned.json")
df_excel = pd.read_excel(f"{base_dir}/Pandas/titanic_pandas_cleaned.xlsx")
print("Pandas: Imported Pandas/titanic_pandas_cleaned.xlsx")

spark = SparkSession.builder.appName("TitanicImport").getOrCreate()
spark_csv = spark.read.csv(f"{base_dir}/PySpark/titanic_spark_cleaned.csv", header=True, inferSchema=True)
print("PySpark: Imported PySpark/titanic_spark_cleaned.csv")
spark_json = spark.read.json(f"{base_dir}/PySpark/titanic_spark_cleaned.json")  # Spark handles JSON Lines natively
print("PySpark: Imported PySpark/titanic_spark_cleaned.json")
spark_parquet = spark.read.parquet(f"{base_dir}/PySpark/titanic_spark_cleaned.parquet")
print("PySpark: Imported PySpark/titanic_spark_cleaned.parquet")
spark.stop()
print("PySpark: Import session stopped")

df_csv.to_sql("TitanicCleanedCSV", engine, if_exists="replace", index=False)
print("SQL Server: Imported SQL/titanic_sql_cleaned.csv to 'TitanicCleanedCSV'")
df_json.to_sql("TitanicCleanedJSON", engine, if_exists="replace", index=False)
print("SQL Server: Imported SQL/titanic_sql_cleaned.json to 'TitanicCleanedJSON'")
df_excel.to_sql("TitanicCleanedExcel", engine, if_exists="replace", index=False)
print("SQL Server: Imported SQL/titanic_sql_cleaned.xlsx to 'TitanicCleanedExcel'")