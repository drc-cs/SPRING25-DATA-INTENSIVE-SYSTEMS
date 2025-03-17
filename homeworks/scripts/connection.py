"""Snowflake connection module.

Written by Joshua D'Arcy, 2025.
"""

from snowflake import connector
from dotenv import load_dotenv
import os
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

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

    conn = connector.connect(
        user=user,
        password=password,
        account=account,
        database=database,
        warehouse=warehouse,
        schema=schema,
    )
    return conn

def as_dataframe(rows: list, cur: object) -> pd.DataFrame:
    """
    Convert the result set to a pandas DataFrame.
    Args:
        rows (list): The result set to convert.
        cur (object): The Snowflake cursor object.
    Returns:
        pd.DataFrame: The result set as a pandas DataFrame.
    """
    return pd.DataFrame(rows, columns=[col[0] for col in cur.description])

