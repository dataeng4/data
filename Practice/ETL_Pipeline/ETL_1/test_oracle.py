import cx_Oracle
try:
    conn = cx_Oracle.connect(
        user="analytics_db",
        password="analytics123",
        dsn="localhost:1521/FREEPDB1"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM customers WHERE total_purchases > 1000")
    for row in cursor.fetchall():
        print(row)
    conn.close()
except Exception as e:
    print(f"Connection failed: {str(e)}")