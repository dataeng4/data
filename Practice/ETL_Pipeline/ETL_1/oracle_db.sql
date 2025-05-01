sqlplus system/your_system_password@localhost:1521/FREEPDB1

CREATE USER analytics_db IDENTIFIED BY analytics123;
GRANT CONNECT, RESOURCE, CREATE SESSION TO analytics_db;
GRANT UNLIMITED TABLESPACE TO analytics_db;

ssqlplus analytics_db/analytics123@localhost:1521/FREEPDB1

CREATE TABLE customers (
    customer_id NUMBER PRIMARY KEY,
    name VARCHAR2(100),
    email VARCHAR2(100),
    total_purchases NUMBER
);

INSERT INTO customers VALUES (1, 'John Doe', 'john.doe@email.com', 1500.00);
INSERT INTO customers VALUES (2, 'Jane Smith', 'jane.smith@email.com', 800.00);
INSERT INTO customers VALUES (3, 'Alice Johnson', 'alice.j@email.com', 2000.00);
INSERT INTO customers VALUES (4, 'Bob Brown', 'bob.brown@email.com', 500.00);
COMMIT;