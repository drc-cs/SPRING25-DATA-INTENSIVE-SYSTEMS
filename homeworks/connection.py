import snowflake.connector
from dotenv import load_dotenv
import os

load_dotenv()

def connect_to_snowflake(
    account: str = os.getenv("SNOWFLAKE_ACCOUNT"),
    user: str = os.getenv("SNOWFLAKE_USER"),
    password: str = os.getenv("SNOWFLAKE_PASSWORD"),
    database: str = None,
    warehouse: str = None,
    schema: str = None,
) -> object:
    """
    Connect to Snowflake using the provided credentials.

    Args:

        database (str): The name of the Snowflake database.
        warehouse (str): The name of the Snowflake warehouse.
        schema (str): The name of the Snowflake schema.
        account (str, optional): The Snowflake account name. Defaults to the environment variable SNOWFLAKE_ACCOUNT.
        user (str, optional): The Snowflake username. Defaults to the environment variable SNOWFLAKE_USER.
        password (str, optional): The Snowflake password. Defaults to the environment variable SNOWFLAKE_PASSWORD.

    Returns:
        object: Snowflake connection object.
    """

    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        database=database,
        warehouse=warehouse,
        schema=schema,
    )
    return conn