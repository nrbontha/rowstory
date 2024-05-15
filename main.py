import db
import json
from pprint import pprint

import ai
import config
import sql
import prompts

from openai import OpenAI


# Initialize OpenAI client
client = OpenAI(api_key=config.OPENAI_API_KEY)


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


def main():
    data = loyalty[1]

    # Generate schema for data item
    schema = ai.generate_schema(client, prompts.schema_generation_prompt, data)
    record = {"schema": schema, "data": data}
    pprint(record)

    # Generate table name
    table_name = ai.generate_table_name(client, prompts.table_name_generation_prompt, record)["table_name"]
    pprint(table_name)

    conn = db.create_connection(
        config.DB_NAME,
        config.DB_USER,
        config.DB_PASSWORD,
        config.DB_HOST,
        config.DB_PORT
    )


    with conn as cursor:
        try:
            # Create schema registry table if it doesn't exist
            create_schema_registry_query = sql.create_schema_registry_table()
            db.execute_query(cursor, create_schema_registry_query)

            # Check if the schema already exists in the registry
            schema_hash = sql.hash_schema(schema)
            select_schema_registry_query = f"SELECT table_name FROM schema_registry WHERE schema_hash = '{schema_hash}'"
            schema_registry_record = db.execute_read_query(cursor, select_schema_registry_query)

            if not schema_registry_record:
                schema_json = json.dumps(schema)
                insert_schema_registry_query = f"""
                    INSERT INTO schema_registry (table_name, schema_json, schema_hash)
                    VALUES ('{table_name}', '{schema_json}', '{schema_hash}');
                """
                db.execute_query(cursor, insert_schema_registry_query)
                print("Schema inserted successfully.")

                # Create new table
                create_table_query = sql.generate_create_table_statement(schema, table_name)
                db.execute_query(cursor, create_table_query)
                print("New table created successfully.")

            # Insert data into the table
            insert_query = sql.generate_insert_statement(record, table_name)
            db.execute_query(cursor, insert_query)
            print("New data inserted successfully.")

        except Exception as e:
            print(f"An error occurred: {e}")
            conn.rollback()
        finally:
            conn.commit()

if __name__ == "__main__":
    main()


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

