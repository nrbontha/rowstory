from openai import OpenAI

import db

import json
from io import StringIO
from pprint import pprint


OPENAI_API_KEY = ""

client = OpenAI(api_key=OPENAI_API_KEY)

# Database parameters
db_name = "mydatabase"
db_user = "myuser"
db_password = "mypassword"
db_host = "localhost"  # Use "localhost" if the Docker container exposes the port on localhost
db_port = "5432"       # The port forwarded by your Docker container

customers = [
    {
        "id": 0,
        "first_name": "John",
        "last_name": "Doe",
        "age": 30,
        "income": 50000,
        "email": "john.doe@yahoo.com"
    },
    {
        "id": 1,
        "first_name": "Jane",
        "last_name": "Smith",
        "age": 28,
        "income": 55000,
        "email": "jane.smith@gmail.com"
    },
    {
        "id": 2,
        "first_name": "Michael",
        "last_name": "Johnson",
        "age": 35,
        "income": 60000,
        "email": "michael.johnson@gmail.com"
    },
    {
        "id": 3,
        "first_name": "Emily",
        "last_name": "Williams",
        "age": 26,
        "income": 48000,
        "email": "emily.williams@outlook.com"
    },
    {
        "id": 4,
        "first_name": "David",
        "last_name": "Brown",
        "age": 40,
        "income": 200000,
        "email": "david.brown@gmail.com"
    }
]

purchases = [
    {
        "id": 0,
        "customer_id": 1,
        "item": "Laptop",
        "price": 1500,
        "quantity": 1,
        "date": "2024-05-06"
    },
    {
        "id": 1,
        "customer_id": 2,
        "item": "Smartphone",
        "price": 800,
        "quantity": 1,
        "date": "2024-05-05"
    },
    {
        "id": 2,
        "customer_id": 2,
        "item": "Laptop",
        "price": 2000,
        "quantity": 1,
        "date": "2024-05-10"
    },
    {
        "id": 3,
        "customer_id": 3,
        "item": "Headphones",
        "price": 200,
        "quantity": 2,
        "date": "2024-05-04"
    },
    {
        "id": 4,
        "customer_id": 4,
        "item": "Tablet",
        "price": 600,
        "quantity": 1,
        "date": "2024-05-03"
    },
    {
        "id": 5,
        "customer_id": 4,
        "item": "Smartphone",
        "price": 1000,
        "quantity": 1,
        "date": "2024-05-05"
    },
    {
        "id": 6,
        "customer_id": 5,
        "item": "TV",
        "price": 1000,
        "quantity": 1,
        "date": "2024-05-02"
    }
]

loyalty = [
    {
        "id": 0,
        "customer_id": 1,
        "loyalty_points": 500,
        "loyalty_level": "Gold"
    },
    {
        "id": 1,
        "customer_id": 2,
        "loyalty_points": 300,
        "loyalty_level": "Silver"
    }
]

schema_generation_prompt = """
    You are tasked with generating a Postgres-compatible schema from structured data provided in formats such as JSON, CSV, or YAML. Your output should precisely follow a specified JSON structure that includes types of data fields, and optionally identifies primary and foreign keys. Here are your detailed instructions:
    
    1. **Data Analysis**:
       - Analyze the provided data to understand its structure and the types of data it contains.
       - Infer the most appropriate Postgres data type for each field (e.g., 'int' for integers, 'varchar' for strings, 'numeric' for floating numbers).
    
    2. **Schema Generation**:
       - Construct a JSON object named 'schema' where each key corresponds to a field in the provided data, and each value is the inferred Postgres data type.
    
    3. **Primary Key Identification**:
       - Determine if there is a primary key (PK) in the dataset, commonly indicated by fields named 'id' or other unique identifiers.
       - If a primary key is identified, include it in the output under the key 'primary_key'.
       - If no clear PK can be determined, do not include the 'primary_key' field.
    
    4. **Foreign Key Identification**:
       - Look for potential foreign keys, typically indicated by naming conventions (e.g., suffix '_id' that could refer to identifiers in other tables).
       - Include these in the output under the key 'foreign_keys' as a list, if any are found.
       - If no foreign keys are detected, do not include the 'foreign_keys' field.
    
    5. **Error Handling**:
       - If the data format is unsupported or critical information for schema generation is missing, return a JSON object with an 'error' key explaining the issue.
       - Example error message:
         ```json
         {"error": "Unsupported data format provided"}
         ```
    
    6. **Output**:
       - Ensure the final output is always in JSON format, adhering to the specified schema structure.
       - Do NOT output anything other than the schema structure.
       - Example schema structure:
         ```json
         {
           "columns": {
             "id": "uuid",
             "customer_id": "int",
             "first_name": "varchar",
             "last_name": "varchar",
             "age": "int",
             "income": "numeric",
             "email": "varchar"
           },
           "primary_key": "id",
           "foreign_keys": ["customer_id"]
         }
         ```
"""

table_name_generation_prompt = """
    You are tasked with generating a table name based on a given record and its associated schema. The table name should intuitively reflect the purpose and content of the data. Here are your detailed instructions:

    1. **Context Analysis**:
       - Examine the keys in the 'data' dictionary to understand the common themes or purposes these fields represent.
       - Consider the general function of the data, such as whether it pertains to transactions, customer information, or other specific activities.

    2. **Key Field Identification**:
       - Identify key fields that may indicate the nature of the data (e.g., 'transaction_id', 'order_date', 'customer_id') which suggest functionalities like orders, purchases, or user management.
       - Prioritize fields that suggest a broader category or function rather than specific details (e.g., 'items' or 'prices').

    3. **Word Choice**:
       - Select clear, simple language that directly reflects the identified function.
       - Avoid technical jargon unless it is commonly understood in the context of the data (e.g., using 'invoice' for financial transactions).

    4. **Formatting**:
       - Use lowercase letters for the table name.
       - If combining words, use underscores to separate them (e.g., 'financial_transactions').
       - Ensure the final name is a single string without spaces.

    5. **Output**:
       - Ensure the name is broad enough to encompass the type of records in the table but specific enough to provide meaningful context.
       - The output should be a JSON response with the following schema:
            {
                "table_name": "{table_name}"
            }

    6. **Example Decision Process**:
       - If the record includes 'customer_id', 'date', 'item', 'price', consider names like 'sales', 'transactions', or 'purchases'.
       - Review the frequency and context of these key terms across different industries to ensure appropriateness.

    7. **Final Selection**:
       - Choose the name that best captures the essence of the data while being concise and descriptive.
       - Example output: 'purchases'
"""


def escape_sql_string(value):
    """Escape single quotes in a string for SQL statements."""
    return value.replace("'", "''")


#TODO: add FKs if the foreign table exists
def generate_create_table_statement(schema, table_name):
    sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"

    columns = schema['columns']
    column_definitions = []
    for column_name, data_type in columns.items():
        column_definitions.append(f"{column_name} {data_type}")

    if schema['primary_key']:
        column_definitions.append(f"PRIMARY KEY ({schema['primary_key']})")

    sql += ",\n".join(column_definitions)
    sql += "\n);"

    return sql

def generate_insert_statement(record, table_name):
    columns = list(record['schema']['columns'].keys())
    values = [record['data'][key] for key in columns if key in record['data']]

    formatted_values = []
    for value in values:
        if isinstance(value, str):
            escaped_value = escape_sql_string(value)
            formatted_values.append(f"'{escaped_value}'")
        else:
            formatted_values.append(str(value))

    values_str = ", ".join(formatted_values)
    insert_statement = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({values_str});"

    return insert_statement


data = customers[0]

schema = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  response_format={"type": "json_object"},
  messages=[
    {"role": "system", "content": f"{schema_generation_prompt}"},
    {"role": "user", "content": f"{data}"},
  ]
)

record = {"schema": json.loads(schema.choices[0].message.content), "data": data}

pprint(record)

table_name = client.chat.completions.create(
  model="gpt-4-turbo",
  response_format={"type": "json_object"},
  messages=[
    {"role": "system", "content": f"{table_name_generation_prompt}"},
    {"role": "user", "content": f"{record}"},
  ]
)

table_name = json.loads(table_name.choices[0].message.content)
pprint(table_name)

insert = generate_insert_statement(record, table_name)
print(insert)


with db.create_connection(db_name, db_user, db_password, db_host, db_port) as connection:
    if connection:
        try:
            # Create new table if it doesn't exist
            create_table_query = generate_create_table_statement(record["schema"], table_name["table_name"])
            db.execute_query(connection, create_table_query)

            # Insert data
            with db.closing(connection.cursor()) as cursor:
                insert_query = generate_insert_statement(record, table_name["table_name"])
                cursor.execute(insert_query)

            # Read records from the table
            select_query = f"SELECT * FROM {table_name['table_name']}"
            records = db.execute_read_query(connection, select_query)
            for record in records:
                print(record)
        except Exception as e:
            print(f"An error occurred: {e}")
            connection.rollback()

# TODO: write function that inserts the new table details in a metadata table
#   * we need to check this table whenever we extract a new schema to see if we've seen this schema before
#   * we also need to see if the data might fit into an existing table even if the schema isn't exactly the same as the existing one (e.g. new columns, etc.)

# TODO: write function that creates the first part of the COPY statement using the schema (above):
# COPY customers (id, customer_id, first_name, last_name, age, income, email)

# TODO: write function that creates the second part of the COPY statement using the record:
# FROM stdin WITH (FORMAT json);
# {"id": 1, "customer_id": 2, "first_name": "John", "last_name": "Doe", "age": 30, "income": 50000, "email": "john.doe@yahoo.com"}
# \.


# TODO: check Postgres DB if the record already exists (and add rules for updating data if necessary)


# TODO: try using fine-tuned models


