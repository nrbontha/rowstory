import json

def generate_schema(client, prompt, data):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(data)},
        ]
    )
    return json.loads(response.choices[0].message.content)

def generate_table_name(client, prompt, record):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(record)},
        ]
    )
    return json.loads(response.choices[0].message.content)
