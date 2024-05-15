schema_generation_prompt = """
    You are tasked with generating a Postgres-compatible schema from structured data provided in formats such as JSON, CSV, or YAML.

    Your output should precisely follow a specified JSON structure that includes types of data fields, and optionally identifies primary and foreign keys.

    Here are your detailed instructions:

    1. **Data Analysis**:
       - Analyze the provided data to understand its structure and the types of data it contains.
       - Infer the most appropriate Postgres data type for each field. Use "least restrictive" data types where appropriate (e.g., 'text' for textual data, 'numeric' for numbers to ensure precision).

    2. **Schema Generation**:
       - Construct a JSON object named 'schema' where each key corresponds to a field in the provided data, and each value is the inferred Postgres data type. Avoid using data types that create sequences or have dependencies.

    3. **Primary Key Identification**:
       - Determine if there is a primary key (PK) in the dataset, commonly indicated by fields named 'id' or other unique identifiers.
       - Use 'numeric' for the PK to maintain maximum flexibility and precision, accommodating for very large or decimal numbers if necessary.
       - Include the primary key in the output under the key 'primary_key' if identified. If a PK is detected in later records, update the schema in the registry accordingly.

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
             "id": "numeric",
             "customer_id": "numeric",
             "first_name": "text",
             "last_name": "text",
             "age": "numeric",
             "income": "numeric",
             "email": "text"
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
