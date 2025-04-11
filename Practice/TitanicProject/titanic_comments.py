# Importing libraries (tools) we need for the script
import pandas as pd  # Pandas is a tool for working with data tables in Python, like Excel spreadsheets
import requests  # Requests lets us download files from the internet, like getting the Titanic data
import io  # io helps us handle data as if it’s a file in memory, without saving it first
import os  # os lets us work with files and folders on our computer, like creating directories
from pyspark.sql import SparkSession  # SparkSession is the starting point for PySpark, a tool for big data processing
from pyspark.sql.functions import col as pyspark_col, when  # pyspark_col and when help us manipulate data in PySpark; renamed col to avoid confusion
from sqlalchemy import create_engine  # create_engine connects Python to SQL Server to save or load data
from reportlab.lib.pagesizes import letter  # letter sets the PDF page size (8.5x11 inches) for our exports
from reportlab.platypus import SimpleDocTemplate, Table  # SimpleDocTemplate and Table help us create PDFs with data tables

# Setting up the base directory where all our files will live
base_dir = "C:/Users/lokesh/Desktop/data/Practice/TitanicProject"  # This is the folder path where we’ll save everything

# Creating folders to keep our files organized
os.makedirs(f"{base_dir}/Pandas", exist_ok=True)  # Makes a Pandas folder; exist_ok=True means no error if it already exists
os.makedirs(f"{base_dir}/PySpark", exist_ok=True)  # Makes a PySpark folder for Spark-related files
os.makedirs(f"{base_dir}/SQL", exist_ok=True)  # Makes an SQL folder for SQL Server exports

# Defining the URL where the raw Titanic data lives online
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"  # This is the web address of the Titanic dataset

# Downloading the raw Titanic data from the internet
response = requests.get(url)  # Sends a request to the URL and gets the data as a response
with open(f"{base_dir}/titanic_raw.csv", "w", encoding="utf-8") as f:  # Opens a file called titanic_raw.csv in write mode ("w")
    f.write(response.text)  # Writes the downloaded data (as text) into the file
print("Downloaded titanic_raw.csv")  # Tells us the download worked

# --- Pandas Section: Working with data using Pandas ---
df = pd.read_csv(io.StringIO(response.text))  # Loads the downloaded data into a Pandas table (DataFrame) from memory
print("Pandas: Loaded DataFrame from URL")  # Confirms the data is loaded
print(df.head())  # Shows the first 5 rows to check what the data looks like

print("Pandas: Null counts per column:")  # Announces we’re checking for missing (null) values
print(df.isna().sum())  # Counts how many missing values are in each column and shows the result

print("Pandas: Checking for ?, -, empty strings...")  # Announces we’re looking for weird values like "?", "-", or empty spaces
for col in df.columns:  # Loops through each column name in the DataFrame
    if df[col].dtype == "object":  # Checks if the column contains text (strings), not numbers
        questionable_values = df[col].isin(["?", "-", ""]).sum()  # Counts how many "?", "-", or empty strings are in this column
        if questionable_values > 0:  # If we find any, we’ll print a message
            print(f"Column {col}: {questionable_values} ?, -, or empty values")  # Shows the count for that column

# Cleaning the data by filling in missing or weird values
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns  # Finds all columns with numbers (integers or decimals)
string_cols = df.select_dtypes(include=["object"]).columns  # Finds all columns with text (strings)
df[numeric_cols] = df[numeric_cols].fillna(0).replace(["?", "-"], 0)  # Replaces missing numbers with 0 and "?" or "-" with 0
df[string_cols] = df[string_cols].fillna("Unknown").replace(["?", "-", ""], "Unknown")  # Replaces missing text with "Unknown" and fixes weird values
print("Pandas: Replaced nulls, ?, -, empty with 0 (numeric), Unknown (string)")  # Confirms the cleaning is done
print("Pandas: Null counts after cleaning:")  # Checks nulls again to make sure they’re gone
print(df.isna().sum())  # Shows the new null counts (should be all 0 now)

# Saving the cleaned Pandas data to files
df.to_csv(f"{base_dir}/Pandas/titanic_pandas_cleaned.csv", index=False)  # Saves the table as a CSV file without row numbers
print("Pandas: Exported cleaned data to Pandas/titanic_pandas_cleaned.csv")  # Confirms CSV save
df.to_json(f"{base_dir}/Pandas/titanic_pandas_cleaned.json", orient="records", lines=True)  # Saves as JSON, one record per line
print("Pandas: Exported cleaned data to Pandas/titanic_pandas_cleaned.json")  # Confirms JSON save
df.to_excel(f"{base_dir}/Pandas/titanic_pandas_cleaned.xlsx", index=False)  # Saves as an Excel file
print("Pandas: Exported cleaned data to Pandas/titanic_pandas_cleaned.xlsx")  # Confirms Excel save
pdf = SimpleDocTemplate(f"{base_dir}/Pandas/titanic_pandas_cleaned.pdf", pagesize=letter)  # Sets up a PDF file with standard letter size
table = Table([df.columns.tolist()] + df.values.tolist())  # Turns the DataFrame into a table (header + data rows) for the PDF
pdf.build([table])  # Creates the PDF with the table inside
print("Pandas: Exported cleaned data to Pandas/titanic_pandas_cleaned.pdf")  # Confirms PDF save

# --- PySpark Section: Processing data with PySpark for big data ---
os.environ["HADOOP_HOME"] = "C:\\hadoop"  # Tells PySpark where Hadoop is (needed for Windows compatibility)
spark = SparkSession.builder.appName("TitanicExample").getOrCreate()  # Starts a PySpark session named "TitanicExample"

spark_df = spark.read.csv(f"{base_dir}/titanic_raw.csv", header=True, inferSchema=True)  # Loads the raw CSV into a Spark table, guessing column types
print("PySpark: Loaded DataFrame from local file")  # Confirms the data is loaded into PySpark
spark_df.show(5)  # Shows the first 5 rows of the Spark table

print("PySpark: Null counts per column:")  # Announces we’re checking for missing values in PySpark
null_counts = spark_df.select([pyspark_col(c).isNull().cast("int").alias(c) for c in spark_df.columns]).groupBy().sum()  # Counts nulls per column
null_counts.show()  # Displays the null counts in a table

print("PySpark: Checking for ?, -, empty strings...")  # Announces we’re looking for odd text values
for c in spark_df.columns:  # Loops through each column name in the Spark table
    if spark_df.schema[c].dataType.simpleString() in ["string"]:  # Checks if the column is text (string type)
        counts = spark_df.filter(pyspark_col(c).isin("?", "-", "")).count()  # Counts rows with "?", "-", or empty strings
        if counts > 0:  # If any are found, we print a message
            print(f"Column {c}: {counts} ?, -, or empty values")  # Shows the count for that column

# Cleaning the Spark data
numeric_cols = [c for c, t in spark_df.dtypes if t in ["int", "double"]]  # Finds numeric columns (integers or decimals)
string_cols = [c for c, t in spark_df.dtypes if t == "string"]  # Finds text columns
for c in numeric_cols:  # Loops through numeric columns
    spark_df = spark_df.fillna({c: 0})  # Replaces missing values in this column with 0
for c in string_cols:  # Loops through text columns
    spark_df = spark_df.fillna({c: "Unknown"})  # Replaces missing text with "Unknown"
    spark_df = spark_df.withColumn(c, when(pyspark_col(c).isin("?", "-", ""), "Unknown").otherwise(pyspark_col(c)))  # Fixes "?", "-", or empty to "Unknown"
print("PySpark: Replaced nulls, ?, -, empty with 0 (numeric), Unknown (string)")  # Confirms cleaning is done

print("PySpark: Null counts after cleaning:")  # Checks nulls again to confirm cleaning worked
null_counts_cleaned = spark_df.select([pyspark_col(c).isNull().cast("int").alias(c) for c in spark_df.columns]).groupBy().sum()  # Recounts nulls
null_counts_cleaned.show()  # Shows the updated null counts (should be all 0)

# Saving the cleaned PySpark data to files
pandas_df = spark_df.toPandas()  # Converts the Spark table to a Pandas table for easier saving
pandas_df.to_csv(f"{base_dir}/PySpark/titanic_spark_cleaned.csv", index=False)  # Saves as CSV
print("PySpark: Exported cleaned data to PySpark/titanic_spark_cleaned.csv")  # Confirms CSV save
pandas_df.to_json(f"{base_dir}/PySpark/titanic_spark_cleaned.json", orient="records", lines=True)  # Saves as JSON, one record per line
print("PySpark: Exported cleaned data to PySpark/titanic_spark_cleaned.json")  # Confirms JSON save
pandas_df.to_parquet(f"{base_dir}/PySpark/titanic_spark_cleaned.parquet")  # Saves as Parquet (a compact format for big data)
print("PySpark: Exported cleaned data to PySpark/titanic_spark_cleaned.parquet")  # Confirms Parquet save
pdf = SimpleDocTemplate(f"{base_dir}/PySpark/titanic_spark_cleaned.pdf", pagesize=letter)  # Sets up a PDF file
table = Table([pandas_df.columns.tolist()] + pandas_df.values.tolist())  # Turns the data into a table for the PDF
pdf.build([table])  # Creates the PDF
print("PySpark: Exported cleaned data to PySpark/titanic_spark_cleaned.pdf")  # Confirms PDF save

spark.stop()  # Stops the PySpark session to free up memory
print("PySpark: Session stopped")  # Confirms PySpark is done

# --- MS SQL Server Section: Storing data in a database ---
server = "INSTANCE-202504\\SQLEXPRESS"  # The name of your SQL Server instance
database = "TitanicDB"  # The name of the database we’ll use
connection_string = f"mssql+pyodbc://{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes&encrypt=no"  # How we connect to SQL Server
engine = create_engine(connection_string)  # Creates a connection tool (engine) to talk to SQL Server

# Loading raw data into SQL Server
df_raw = pd.read_csv(f"{base_dir}/titanic_raw.csv")  # Loads the raw CSV into a Pandas table
df_raw.to_sql("Titanic", engine, if_exists="replace", index=False)  # Saves it as a table called "Titanic" in SQL Server, replacing if it exists
print("SQL Server: Loaded raw data into 'Titanic' table")  # Confirms raw data is saved

# Loading cleaned data into SQL Server
df.to_sql("TitanicCleaned", engine, if_exists="replace", index=False)  # Saves the cleaned Pandas data as "TitanicCleaned" in SQL Server
print("SQL Server: Loaded cleaned data into 'TitanicCleaned' table")  # Confirms cleaned data is saved

# Checking nulls in the cleaned SQL table
null_query = """
SELECT
    SUM(CASE WHEN PassengerId IS NULL THEN 1 ELSE 0 END) AS PassengerId_nulls,  -- Counts nulls in PassengerId
    SUM(CASE WHEN Survived IS NULL THEN 1 ELSE 0 END) AS Survived_nulls,      -- Counts nulls in Survived
    SUM(CASE WHEN Pclass IS NULL THEN 1 ELSE 0 END) AS Pclass_nulls,          -- Counts nulls in Pclass
    SUM(CASE WHEN Name IS NULL THEN 1 ELSE 0 END) AS Name_nulls,              -- Counts nulls in Name
    SUM(CASE WHEN Sex IS NULL THEN 1 ELSE 0 END) AS Sex_nulls,                -- Counts nulls in Sex
    SUM(CASE WHEN Age IS NULL THEN 1 ELSE 0 END) AS Age_nulls,                -- Counts nulls in Age
    SUM(CASE WHEN SibSp IS NULL THEN 1 ELSE 0 END) AS SibSp_nulls,            -- Counts nulls in SibSp
    SUM(CASE WHEN Parch IS NULL THEN 1 ELSE 0 END) AS Parch_nulls,            -- Counts nulls in Parch
    SUM(CASE WHEN Ticket IS NULL THEN 1 ELSE 0 END) AS Ticket_nulls,          -- Counts nulls in Ticket
    SUM(CASE WHEN Fare IS NULL THEN 1 ELSE 0 END) AS Fare_nulls,              -- Counts nulls in Fare
    SUM(CASE WHEN Cabin IS NULL THEN 1 ELSE 0 END) AS Cabin_nulls,            -- Counts nulls in Cabin
    SUM(CASE WHEN Embarked IS NULL THEN 1 ELSE 0 END) AS Embarked_nulls       -- Counts nulls in Embarked
FROM TitanicCleaned  -- Looks at the TitanicCleaned table
"""
null_counts = pd.read_sql(null_query, engine)  # Runs the query and loads the results into a Pandas table
print("SQL Server: Null counts in 'TitanicCleaned':")  # Announces the null check
print(null_counts)  # Shows the null counts (should be all 0)

# Checking for weird values in the cleaned SQL table
check_query = """
SELECT
    SUM(CASE WHEN Name IN ('?', '-', '') THEN 1 ELSE 0 END) AS Name_issues,      -- Counts "?" or "-" or empty in Name
    SUM(CASE WHEN Sex IN ('?', '-', '') THEN 1 ELSE 0 END) AS Sex_issues,        -- Counts "?" or "-" or empty in Sex
    SUM(CASE WHEN Ticket IN ('?', '-', '') THEN 1 ELSE 0 END) AS Ticket_issues,  -- Counts "?" or "-" or empty in Ticket
    SUM(CASE WHEN Cabin IN ('?', '-', '') THEN 1 ELSE 0 END) AS Cabin_issues,    -- Counts "?" or "-" or empty in Cabin
    SUM(CASE WHEN Embarked IN ('?', '-', '') THEN 1 ELSE 0 END) AS Embarked_issues  -- Counts "?" or "-" or empty in Embarked
FROM TitanicCleaned  -- Looks at the TitanicCleaned table
"""
issue_counts = pd.read_sql(check_query, engine)  # Runs the query and loads results into a Pandas table
print("SQL Server: Counts of ?, -, empty strings:")  # Announces the weird value check
print(issue_counts)  # Shows the counts (should be all 0)

# Saving the SQL cleaned data to files
sql_df = pd.read_sql("SELECT * FROM TitanicCleaned", engine)  # Loads all data from TitanicCleaned into a Pandas table
sql_df.to_csv(f"{base_dir}/SQL/titanic_sql_cleaned.csv", index=False)  # Saves as CSV
print("SQL Server: Exported cleaned data to SQL/titanic_sql_cleaned.csv")  # Confirms CSV save
sql_df.to_json(f"{base_dir}/SQL/titanic_sql_cleaned.json", orient="records", lines=True)  # Saves as JSON, one record per line
print("SQL Server: Exported cleaned data to SQL/titanic_sql_cleaned.json")  # Confirms JSON save
sql_df.to_excel(f"{base_dir}/SQL/titanic_sql_cleaned.xlsx", index=False)  # Saves as Excel
print("SQL Server: Exported cleaned data to SQL/titanic_sql_cleaned.xlsx")  # Confirms Excel save
pdf = SimpleDocTemplate(f"{base_dir}/SQL/titanic_sql_cleaned.pdf", pagesize=letter)  # Sets up a PDF file
table = Table([sql_df.columns.tolist()] + sql_df.values.tolist())  # Turns the data into a table for the PDF
pdf.build([table])  # Creates the PDF
print("SQL Server: Exported cleaned data to SQL/titanic_sql_cleaned.pdf")  # Confirms PDF save

# --- Bonus: Importing Cleaned Formats: Loading files back to check they work ---
print("\n--- Importing Cleaned Formats ---")  # Announces the import section

df_csv = pd.read_csv(f"{base_dir}/Pandas/titanic_pandas_cleaned.csv")  # Loads the Pandas CSV back into a table
print("Pandas: Imported Pandas/titanic_pandas_cleaned.csv")  # Confirms CSV load
df_json = pd.read_json(f"{base_dir}/Pandas/titanic_pandas_cleaned.json", lines=True)  # Loads the Pandas JSON (line-by-line format)
print("Pandas: Imported Pandas/titanic_pandas_cleaned.json")  # Confirms JSON load
df_excel = pd.read_excel(f"{base_dir}/Pandas/titanic_pandas_cleaned.xlsx")  # Loads the Pandas Excel file
print("Pandas: Imported Pandas/titanic_pandas_cleaned.xlsx")  # Confirms Excel load

spark = SparkSession.builder.appName("TitanicImport").getOrCreate()  # Starts a new PySpark session for importing
spark_csv = spark.read.csv(f"{base_dir}/PySpark/titanic_spark_cleaned.csv", header=True, inferSchema=True)  # Loads the PySpark CSV
print("PySpark: Imported PySpark/titanic_spark_cleaned.csv")  # Confirms CSV load
spark_json = spark.read.json(f"{base_dir}/PySpark/titanic_spark_cleaned.json")  # Loads the PySpark JSON (handles line format automatically)
print("PySpark: Imported PySpark/titanic_spark_cleaned.json")  # Confirms JSON load
spark_parquet = spark.read.parquet(f"{base_dir}/PySpark/titanic_spark_cleaned.parquet")  # Loads the PySpark Parquet file
print("PySpark: Imported PySpark/titanic_spark_cleaned.parquet")  # Confirms Parquet load
spark.stop()  # Stops the PySpark session
print("PySpark: Import session stopped")  # Confirms PySpark is done

# Saving the imported files into SQL Server as new tables
df_csv.to_sql("TitanicCleanedCSV", engine, if_exists="replace", index=False)  # Saves the imported CSV as a new SQL table
print("SQL Server: Imported SQL/titanic_sql_cleaned.csv to 'TitanicCleanedCSV'")  # Confirms save
df_json.to_sql("TitanicCleanedJSON", engine, if_exists="replace", index=False)  # Saves the imported JSON as a new SQL table
print("SQL Server: Imported SQL/titanic_sql_cleaned.json to 'TitanicCleanedJSON'")  # Confirms save
df_excel.to_sql("TitanicCleanedExcel", engine, if_exists="replace", index=False)  # Saves the imported Excel as a new SQL table
print("SQL Server: Imported SQL/titanic_sql_cleaned.xlsx to 'TitanicCleanedExcel'")  # Confirms save