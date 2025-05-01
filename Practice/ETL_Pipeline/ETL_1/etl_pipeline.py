import yaml
import pandas as pd
from sqlalchemy import create_engine
import logging
import logging.config
from typing import Dict, Any
import sys
import os

def setup_logging(logging_config_path: str = "logging_config.yaml"):
    """Set up logging configuration from a YAML file."""
    try:
        with open(logging_config_path, 'r') as file:
            log_config = yaml.safe_load(file)
        logging.config.dictConfig(log_config)
        logger = logging.getLogger(__name__)
        logger.info("Logging configured successfully")
        return logger
    except Exception as e:
        print(f"Failed to configure logging: {str(e)}")
        sys.exit(1)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate the YAML configuration file."""
    logger = logging.getLogger(__name__)
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Validate required fields
        required_fields = ['database', 'query', 'output']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in config file")
        
        required_db_fields = ['type', 'host', 'port', 'name', 'user', 'password']
        for field in required_db_fields:
            if field not in config['database']:
                raise ValueError(f"Missing required database field '{field}' in config file")
        
        if 'file_path' not in config['output']:
            raise ValueError("Missing required output field 'file_path' in config file")
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise

def get_connection_string(db_config: Dict[str, Any]) -> str:
    """Generate SQLAlchemy connection string based on database type."""
    logger = logging.getLogger(__name__)
    db_type = db_config['type'].lower()
    host = db_config['host'].replace('\\', '\\\\')
    port = db_config['port']
    db_name = db_config['name']
    user = db_config['user']
    password = db_config['password']
    
    db_drivers = {
        'postgresql': f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}",
        'mysql': f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}",
        'mssql': f"mssql+pyodbc://{user}:{password}@{host}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server" if port == '' else f"mssql+pyodbc://{user}:{password}@{host}:{port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server",
        'oracle': f"oracle+cx_oracle://{user}:{password}@{host}:{port}/{db_name}"
    }
    
    if db_type not in db_drivers:
        raise ValueError(f"Unsupported database type: {db_type}")
    
    logger.info(f"Generated connection string for {db_type}")
    return db_drivers[db_type]

def connect_to_database(db_config: Dict[str, Any]):
    """Establish a database connection using SQLAlchemy with pyodbc or cx_Oracle."""
    logger = logging.getLogger(__name__)
    try:
        db_type = db_config['type'].lower()
        if db_type == 'mssql':
            import pyodbc
            conn_str = (
                'DRIVER={ODBC Driver 17 for SQL Server};'
                f'SERVER={db_config["host"]};'
                f'DATABASE={db_config["name"]};'
                f'UID={db_config["user"]};'
                f'PWD={db_config["password"]}'
            )
            conn = pyodbc.connect(conn_str)
            engine = create_engine(f"mssql+pyodbc://", creator=lambda: conn)
        elif db_type == 'oracle':
            import cx_Oracle
            dsn = f"{db_config['host']}:{db_config['port']}/{db_config['name']}"
            conn = cx_Oracle.connect(
                user=db_config['user'],
                password=db_config['password'],
                dsn=dsn
            )
            engine = create_engine(f"oracle+cx_oracle://", creator=lambda: conn)
        else:
            connection_string = get_connection_string(db_config)
            engine = create_engine(connection_string)
        logger.info(f"Successfully connected to {db_config['type']} database")
        return engine
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise

def execute_query(engine, query: str) -> pd.DataFrame:
    """Execute the SQL query and return results as a pandas DataFrame."""
    logger = logging.getLogger(__name__)
    try:
        df = pd.read_sql(query, engine)
        logger.info(f"Query executed successfully. Retrieved {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Failed to execute query: {str(e)}")
        raise

def save_to_excel(df: pd.DataFrame, output_path: str):
    """Save the DataFrame to an Excel file."""
    logger = logging.getLogger(__name__)
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        df.to_excel(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save data to Excel: {str(e)}")
        raise

def run_etl(config_path: str):
    """Main function to run the ETL pipeline."""
    logger = setup_logging()
    try:
        config = load_config(config_path)
        engine = connect_to_database(config['database'])
        df = execute_query(engine, config['query'])
        save_to_excel(df, config['output']['file_path'])
        logger.info("ETL pipeline completed successfully")
    except Exception as e:
        logger.error(f"ETL pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    #config_file = "postgres_config.yaml"
    #config_file = "mysql_config.yaml"
    #config_file = "mssql_config.yaml"
    config_file = "oracle_config.yaml"
    run_etl(config_file)