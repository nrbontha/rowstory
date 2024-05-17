from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import openai

documents = SimpleDirectoryReader("data").load_data()
llm = openai.OpenAI(model="gpt-4")
service_context = ServiceContext.from_defaults(llm=llm)
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("I am the CEO of the company that produced this data. Which products do you think I should sell more of?")
print(response)
