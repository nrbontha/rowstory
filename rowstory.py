import os
import shutil
import pickle
import numpy as np
from abc import ABC, abstractmethod
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from mirascope.core import openai, prompt_template
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import hnswlib

# Load environment variables
os.environ["OPENAI_API_KEY"] = "your-api-key-here" #TODO: manage this better

# Pydantic models for input validation
class DocumentModel(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

class UpsertIndexModel(BaseModel):
    documents: List[DocumentModel]

# Abstract Retriever class
class RetrieverInterface(ABC):
    @abstractmethod
    def get_or_create_index(self, topic: str) -> Any:
        pass

    @abstractmethod
    def update_index(self, topic: str, documents: List[DocumentModel]):
        pass

    @abstractmethod
    def query_index(self, topic: str, query: str, top_k: int = 10) -> Dict[str, Any]:
        pass

    @abstractmethod
    def delete_index(self, topic: str):
        pass

    @abstractmethod
    def get_index_size(self, topic: str) -> int:
        pass

# HNSW Retriever class
class HNSWRetriever(RetrieverInterface):
    def __init__(self, base_storage_path: str):
        self.base_storage_path = base_storage_path
        os.makedirs(self.base_storage_path, exist_ok=True)  # Add this line
        self.indices = {}
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dim = 384  # Dimension of the embeddings

    def get_or_create_index(self, topic: str) -> hnswlib.Index:
        if topic not in self.indices:
            index_path = os.path.join(self.base_storage_path, f"{topic}_index.bin")
            if os.path.exists(index_path):
                # Load existing index
                index = hnswlib.Index(space='cosine', dim=self.dim)
                index.load_index(index_path)
            else:
                # Create new index
                index = hnswlib.Index(space='cosine', dim=self.dim)
                index.init_index(max_elements=100000, ef_construction=200, M=16)
            self.indices[topic] = index
        return self.indices[topic]

    def update_index(self, topic: str, documents: List[DocumentModel]):
        index = self.get_or_create_index(topic)
        contents = [doc.content for doc in documents]
        embeddings = self.embedder.encode(contents)
        
        # Get the current size of the index
        current_size = index.get_current_count()
        
        # Add new elements
        index.add_items(embeddings, list(range(current_size, current_size + len(embeddings))))

        # Save metadata
        metadata_path = os.path.join(self.base_storage_path, f"{topic}_metadata.pkl")
        metadata = self.load_metadata(topic)
        for i, doc in enumerate(documents):
            metadata[current_size + i] = doc.model_dump()
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        # Save index
        index_path = os.path.join(self.base_storage_path, f"{topic}_index.bin")
        index.save_index(index_path)

    def query_index(self, topic: str, query: str, top_k: int = 10) -> Dict[str, Any]:
        index = self.get_or_create_index(topic)
        query_vector = self.embedder.encode([query])[0]
        
        # Start with a higher ef value
        ef_search = 100
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                index.set_ef(ef_search)
                labels, distances = index.knn_query(query_vector, k=min(top_k, index.get_current_count()))
                
                # Load metadata
                metadata = self.load_metadata(topic)
                
                # Prepare results
                results = [metadata[label].get('content', '') for label in labels[0] if label in metadata]
                
                return {
                    "results": results,
                    "total_docs": index.get_current_count(),
                    "returned_docs": len(results)
                }
            except RuntimeError as e:
                if "Cannot return the results in a contiguous 2D array" in str(e):
                    ef_search *= 2  # Double the ef_search value
                    if attempt == max_attempts - 1:
                        raise HTTPException(status_code=500, detail=f"Failed to query index after {max_attempts} attempts with ef={ef_search}")
                else:
                    raise HTTPException(status_code=500, detail=str(e))

    def delete_index(self, topic: str):
        if topic in self.indices:
            del self.indices[topic]
        
        index_path = os.path.join(self.base_storage_path, f"{topic}_index.bin")
        metadata_path = os.path.join(self.base_storage_path, f"{topic}_metadata.pkl")
        
        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

    def get_index_size(self, topic: str) -> int:
        index = self.get_or_create_index(topic)
        return index.get_current_count()

    def load_metadata(self, topic: str) -> Dict[int, Dict]:
        metadata_path = os.path.join(self.base_storage_path, f"{topic}_metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                return pickle.load(f)
        return {}

# FastAPI app and routes
app = FastAPI()
hnsw_retriever = HNSWRetriever("./hnsw_index_storage")

@app.post("/index/{topic}")
async def upsert_index(topic: str, upsert_model: UpsertIndexModel):
    try:
        hnsw_retriever.update_index(topic, upsert_model.documents)
        return {"message": f"Index for topic '{topic}' upserted successfully with {len(upsert_model.documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/index/{topic}/query")
async def query_index(topic: str, query: str, top_k: int = 10):
    try:
        result = hnsw_retriever.query_index(topic, query, top_k)
        
        excerpts = "\n".join(result["results"])
        response = generate_response(excerpts=excerpts, query=query)
        return {
            "response": response,
            "total_docs": result["total_docs"],
            "returned_docs": result["returned_docs"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/index/{topic}")
async def delete_index(topic: str):
    try:
        hnsw_retriever.delete_index(topic)
        return {"message": f"Index for topic '{topic}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/index/{topic}/size")
async def index_size(topic: str):
    try:
        hnsw_size = hnsw_retriever.get_index_size(topic)
        return {"topic": topic, "hnsw_size": hnsw_size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# OpenAI response generation (unchanged)
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
