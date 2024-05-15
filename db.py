import psycopg2
from contextlib import closing

def create_connection(db_name, db_user, db_password, db_host, db_port):
    connection = psycopg2.connect(
        database=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port,
    )
    print("Connection to PostgreSQL DB successful")
    return connection

def execute_query(connection, query):
    with closing(connection.cursor()) as cursor:
        cursor.execute(query)
        connection.commit()
        print("Query executed successfully")

def execute_read_query(connection, query):
    with closing(connection.cursor()) as cursor:
        cursor.execute(query)
        return cursor.fetchall()
