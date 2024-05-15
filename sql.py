import hashlib
import json

def create_schema_registry_table():
    sql = """
        CREATE TABLE IF NOT EXISTS schema_registry (
            table_name VARCHAR(255) PRIMARY KEY,
            schema_json JSONB NOT NULL,
            schema_hash CHAR(64) NOT NULL
        );

        -- Create a GIN index on the schema_json column to improve JSONB operations
        CREATE INDEX IF NOT EXISTS idx_schema_json ON schema_registry USING GIN (schema_json);

        -- Create an index on the schema_hash column for fast lookup
        CREATE INDEX IF NOT EXISTS idx_schema_hash ON schema_registry (schema_hash);
    """
    return sql


def generate_create_table_statement(schema, table_name):
    sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"

    columns = schema['columns']
    column_definitions = []
    for column_name, data_type in columns.items():
        column_definitions.append(f"{column_name} {data_type}")

    if schema.get('primary_key'):
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


def hash_schema(schema):
    canonical_schema = json.dumps(schema, sort_keys=True)
    schema_hash = hashlib.sha256(canonical_schema.encode('utf-8')).hexdigest()
    return schema_hash

def escape_sql_string(value):
    """Escape single quotes in a string for SQL statements."""
    return value.replace("'", "''")
