import pyodbc
try:
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=INSTANCE-202504\\SQLEXPRESS;'
        'DATABASE=inventory_db;'
        'UID=sa;'
        'PWD=1234'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products WHERE stock > 0")
    for row in cursor.fetchall():
        print(row)
    conn.close()
except Exception as e:
    print(f"Connection failed: {str(e)}")