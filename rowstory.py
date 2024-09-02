#!/usr/bin/env python3

import os
import pickle
import hashlib
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import List, Dict, Any

import hnswlib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mirascope.core import openai, prompt_template
from sentence_transformers import SentenceTransformer


# Environment variables
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # TODO: manage this securely


class VectorIndex(ABC):
    @abstractmethod
    def insert(self, documents: List[Dict[str, Any]]):
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete(self):
        pass

    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def save(self, file_path: str):
        pass

    @abstractmethod
    def load(self, file_path: str):
        pass

class Embedder(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        pass

class RetrieverInterface(ABC):
    @abstractmethod
    def get_or_create_index(self, topic: str) -> Any:
        pass

    @abstractmethod
    def update_index(self, topic: str, documents: List[Dict[str, Any]]):
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

class HNSWIndex(VectorIndex):
    def __init__(self, dim: int, max_elements: int, ef_construction: int = 200, M: int = 16):
        self.dim = dim
        self.max_elements = max_elements
        self.ef_construction = ef_construction
        self.M = M
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        self.documents = {}
        self.current_id = 0

    def insert(self, documents: List[Dict[str, Any]]):
        vectors = [np.array(doc['embedding']) for doc in documents]
        ids = list(range(self.current_id, self.current_id + len(documents)))
        self.index.add_items(vectors, ids)
        for i, doc in zip(ids, documents):
            self.documents[i] = doc
        self.current_id += len(documents)

    def search(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        labels, distances = self.index.knn_query(query_vector, k=min(top_k, len(self.documents)))
        results = []
        for label, distance in zip(labels[0], distances[0]):
            doc = self.documents[label].copy()
            doc['distance'] = float(distance)  # Convert numpy.float32 to Python float
            results.append(doc)
        return results

    def delete(self):
        self.index = hnswlib.Index(space='cosine', dim=self.dim)
        self.index.init_index(max_elements=self.max_elements, ef_construction=self.ef_construction, M=self.M)
        self.documents.clear()
        self.current_id = 0

    def size(self) -> int:
        return len(self.documents)

    def save(self, file_path: str):
        self.index.save_index(file_path)
        with open(file_path + '.metadata', 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'current_id': self.current_id,
                'dim': self.dim,
                'max_elements': self.max_elements,
                'ef_construction': self.ef_construction,
                'M': self.M
            }, f)

    def load(self, file_path: str):
        self.index.load_index(file_path)
        with open(file_path + '.metadata', 'rb') as f:
            metadata = pickle.load(f)
        self.documents = metadata['documents']
        self.current_id = metadata['current_id']
        self.dim = metadata['dim']
        self.max_elements = metadata['max_elements']
        self.ef_construction = metadata['ef_construction']
        self.M = metadata['M']

class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

class HNSWRetriever(RetrieverInterface):
    def __init__(self, vector_index: HNSWIndex, embedder: Embedder, base_path: str):
        self.vector_index = vector_index
        self.embedder = embedder
        self.base_path = base_path
        self.indices = {}
        os.makedirs(self.base_path, exist_ok=True)

    def get_or_create_index(self, topic: str) -> VectorIndex:
        if topic not in self.indices:
            index_path = os.path.join(self.base_path, f"{topic}_index.bin")
            if os.path.exists(index_path):
                index = HNSWIndex(dim=self.vector_index.dim, max_elements=self.vector_index.max_elements)
                index.load(index_path)
                self.indices[topic] = index
            else:
                self.indices[topic] = self.vector_index.__class__(
                    dim=self.vector_index.dim,
                    max_elements=self.vector_index.max_elements,
                    ef_construction=self.vector_index.ef_construction,
                    M=self.vector_index.M
                )
        return self.indices[topic]

    def update_index(self, topic: str, documents: List[Dict[str, Any]]):
        index = self.get_or_create_index(topic)
        contents = [doc['content'] for doc in documents]
        embeddings = self.embedder.encode(contents)
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding
        index.insert(documents)
        self._save_index(topic)

    def query_index(self, topic: str, query: str, top_k: int = 10) -> Dict[str, Any]:
        index = self.get_or_create_index(topic)
        query_vector = self.embedder.encode([query])[0]
        results = index.search(query_vector, top_k)
        return {
            "results": results,
            "total_docs": self.get_index_size(topic),
            "returned_docs": len(results)
        }

    def delete_index(self, topic: str):
        if topic in self.indices:
            self.indices[topic].delete()
            del self.indices[topic]
            index_path = os.path.join(self.base_path, f"{topic}_index.bin")
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(index_path + '.metadata'):
                os.remove(index_path + '.metadata')

    def get_index_size(self, topic: str) -> int:
        index = self.get_or_create_index(topic)
        return index.size()

    def _save_index(self, topic: str):
        index_path = os.path.join(self.base_path, f"{topic}_index.bin")
        self.indices[topic].save(index_path)

class DocumentModel(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

class UpsertIndexModel(BaseModel):
    documents: List[DocumentModel]

class QueryModel(BaseModel):
    query: str
    top_k: int = 10

app = FastAPI()

vector_index = HNSWIndex(dim=384, max_elements=100000)
embedder = SentenceTransformerEmbedder('all-MiniLM-L6-v2')
retriever = HNSWRetriever(vector_index, embedder, base_path="./index_storage")

@app.post("/index/{topic}")
async def upsert_index(topic: str, upsert_model: UpsertIndexModel):
    try:
        documents = [doc.model_dump() for doc in upsert_model.documents]
        retriever.update_index(topic, documents)
        return {"message": f"Index for topic '{topic}' upserted successfully with {len(documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/index/{topic}/query")
async def query_index(topic: str, query: str, top_k: int = 10):
    try:
        result = retriever.query_index(topic, query, top_k)
        
        if not result["results"]:
            return {"message": "No results found for the given query."}
        
        excerpts = "\n".join([doc.get("content", "") for doc in result["results"]])
        response = generate_response(excerpts=excerpts, query=query)
        
        return {
            "response": response,
            "total_docs": result["total_docs"],
            "returned_docs": result["returned_docs"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

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

@openai.call("gpt-4o-mini")
@prompt_template(
    """
    SYSTEM:
    You are an intelligent data querying service designed to help users understand and interpret their data.
    The user will provide a query, and your task is to analyze the relevant data provided in the excerpts.
    
    Important Notes:
    1. The excerpts contain not only directly relevant information but also related data that might provide additional context or insights.
    2. Pay special attention to relationships between different types of data (e.g., users, transactions, products, orders).
    3. Consider all provided information, even if it doesn't seem immediately relevant to the query.
    4. If the data set is large, prioritize the most relevant information but don't ignore potential patterns in the broader dataset.
    5. Look for insights or patterns that might not be immediately obvious from the most directly relevant data.

    Use this data to generate the most accurate, helpful, and insightful response to the user's query.
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
