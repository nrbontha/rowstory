import os
import shutil
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from mirascope.core import openai, prompt_template
from typing import List, Dict, Any

# Load environment variables
os.environ["OPENAI_API_KEY"] = "your-api-key-here" #TODO: manage this better

# Pydantic models for input validation
class DocumentModel(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

class UpdateIndexModel(BaseModel):
    topic: str
    documents: List[DocumentModel]

class QueryModel(BaseModel):
    topic: str
    query: str
    top_k: int = Field(default=10, ge=1)

class DeleteIndexModel(BaseModel):
    topic: str

class IndexSizeModel(BaseModel):
    topic: str

class UpsertIndexModel(BaseModel):
    documents: List[DocumentModel]

# Retriever class
class Retriever:
    def __init__(self, base_storage_path: str):
        self.base_storage_path = base_storage_path
        self.indices: Dict[str, VectorStoreIndex] = {}

    def get_or_create_index(self, topic: str) -> VectorStoreIndex:
        if topic not in self.indices:
            index_storage_path = os.path.join(self.base_storage_path, topic)
            if os.path.exists(index_storage_path):
                storage_context = StorageContext.from_defaults(persist_dir=index_storage_path)
                self.indices[topic] = load_index_from_storage(storage_context)
            else:
                self.indices[topic] = VectorStoreIndex([])
        return self.indices[topic]

    def update_index(self, topic: str, documents: List[Document]):
        index = self.get_or_create_index(topic)
        for doc in documents:
            index.insert(doc)
        index_storage_path = os.path.join(self.base_storage_path, topic)
        index.storage_context.persist(persist_dir=index_storage_path)

    def query_index(self, topic: str, query: str, top_k: int = 10) -> Dict[str, Any]:
        index = self.get_or_create_index(topic)
        retriever = index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query)
        return {
            "results": [node.get_content() for node in results],
            "total_docs": len(index.docstore.docs),
            "returned_docs": len(results)
        }

    def delete_index(self, topic: str):
        if topic in self.indices:
            del self.indices[topic]
        
        index_storage_path = os.path.join(self.base_storage_path, topic)
        if os.path.exists(index_storage_path):
            shutil.rmtree(index_storage_path)

    def get_index_size(self, topic: str) -> int:
        index = self.get_or_create_index(topic)
        return len(index.docstore.docs)

# FastAPI app and routes
app = FastAPI()
retriever = Retriever("./index_storage")

@app.put("/index/{topic}")
async def update_index(topic: str, documents: List[DocumentModel]):
    try:
        docs = [Document(text=doc.content, metadata=doc.metadata) for doc in documents]
        retriever.update_index(topic, docs)
        return {"message": f"Index for topic '{topic}' updated successfully with {len(documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/index/{topic}/query")
async def query_index(topic: str, query: str, top_k: int = 10):
    try:
        query_result = retriever.query_index(topic, query, top_k)
        excerpts = "\n".join(query_result["results"])
        response = generate_response(excerpts=excerpts, query=query)
        return {
            "response": response,
            "total_docs": query_result["total_docs"],
            "returned_docs": query_result["returned_docs"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/index/{topic}")
async def delete_index(topic: str):
    try:
        retriever.delete_index(topic)
        return {"message": f"Index for topic '{topic}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/index/{topic}/size")
async def index_size(topic: str):
    try:
        size = retriever.get_index_size(topic)
        return {"topic": topic, "size": size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index/{topic}")
async def upsert_index(topic: str, upsert_model: UpsertIndexModel):
    try:
        docs = [Document(text=doc.content, metadata=doc.metadata) for doc in upsert_model.documents]
        retriever.update_index(topic, docs)
        return {"message": f"Index for topic '{topic}' upserted successfully with {len(docs)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# OpenAI response generation
@openai.call("gpt-4o-mini")
@prompt_template(
    """
    SYSTEM:
    You are an intelligent data querying service designed to help users understand and interpret their data.
    The user will provide a query, and your task is to analyze the relevant data provided in the excerpts.
    Use this data to generate the most accurate and helpful response to the user's query.
    
    If the query directly relates to the data, provide a detailed answer, highlighting key insights and patterns.
    If the query does not correspond to the data or cannot be answered based on the available information, please 
    acknowledge this and provide a response indicating that the data is insufficient to answer the query.
    
    Maintain a clear, concise, and professional tone in your response.
    
    USER QUERY: {query}
    
    DATA EXCERPTS:
    {excerpts}
    
    YOUR RESPONSE:
    """
)
def generate_response(excerpts: str, query: str):
    pass
