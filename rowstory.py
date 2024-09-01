#!/usr/bin/env python3

import os
import pickle
import hashlib
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import List, Dict, Any

import hnswlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mirascope.core import openai, prompt_template
from sentence_transformers import SentenceTransformer


# Environment variables
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # TODO: manage this securely


class DocumentModel(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}

class UpsertIndexModel(BaseModel):
    documents: List[DocumentModel]

class QueryModel(BaseModel):
    query: str
    top_k: int = 10


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


class HNSWRetriever(RetrieverInterface):
    def __init__(self, base_storage_path: str):
        self.base_storage_path = base_storage_path
        os.makedirs(self.base_storage_path, exist_ok=True)
        self.indices = {}
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dim = 384  # Dimension of the embeddings
        self.key_frequency = Counter()
        self.key_value_index = defaultdict(lambda: defaultdict(set))
        self.ef_search = 50

    def get_or_create_index(self, topic: str) -> hnswlib.Index:
        if topic not in self.indices:
            index_path = os.path.join(self.base_storage_path, f"{topic}_index.bin")
            if os.path.exists(index_path):
                index = hnswlib.Index(space='cosine', dim=self.dim)
                index.load_index(index_path)
            else:
                index = hnswlib.Index(space='cosine', dim=self.dim)
                index.init_index(max_elements=100000, ef_construction=200, M=16)
            self.indices[topic] = index
        return self.indices[topic]

    def update_index(self, topic: str, documents: List[DocumentModel]):
        index = self.get_or_create_index(topic)
        contents = [doc.content for doc in documents]
        embeddings = self.embedder.encode(contents)
        
        current_size = index.get_current_count()
        index.add_items(embeddings, list(range(current_size, current_size + len(embeddings))))

        metadata_path = os.path.join(self.base_storage_path, f"{topic}_metadata.pkl")
        metadata = self.load_metadata(topic)
        for i, doc in enumerate(documents):
            doc_data = doc.model_dump()
            generated_metadata = self.generate_metadata(doc_data['content'])
            doc_data['metadata'] = generated_metadata
            doc_id = current_size + i
            metadata[doc_id] = doc_data
            self.update_key_frequency(generated_metadata)
            self.update_key_value_index(doc_id, generated_metadata)
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        index_path = os.path.join(self.base_storage_path, f"{topic}_index.bin")
        index.save_index(index_path)

    def query_index(self, topic: str, query: str, top_k: int = 10) -> Dict[str, Any]:
        try:
            index = self.get_or_create_index(topic)
            query_vector = self.embedder.encode([query])[0]

            max_ef_search = 200
            while self.ef_search <= max_ef_search:
                try:
                    index.set_ef(self.ef_search)
                    labels, distances = index.knn_query(query_vector, k=min(top_k, index.get_current_count()))
                    break
                except RuntimeError:
                    self.ef_search *= 2
                    if self.ef_search > max_ef_search:
                        raise HTTPException(status_code=500, detail="Unable to perform the query. The index might be empty or the query parameters are unsuitable.")

            metadata = self.load_metadata(topic)
            results = [metadata[label] for label in labels[0] if label in metadata]
            related_data = self.gather_related_data(metadata, results)
            
            return {
                "results": related_data,
                "total_docs": index.get_current_count(),
                "returned_docs": len(related_data)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

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

    def generate_metadata(self, content: str) -> Dict[str, Any]:
        metadata = {}
        metadata['id'] = hashlib.md5(content.encode()).hexdigest()
        key_value_pairs = re.findall(r'(\w+)[:\s]+(\S+)', content)
        for key, value in key_value_pairs:
            metadata[key.lower()] = value.rstrip(',')
        return metadata

    def update_key_frequency(self, metadata: Dict[str, Any]):
        for key in metadata.keys():
            self.key_frequency[key] += 1

    def update_key_value_index(self, doc_id: int, metadata: Dict[str, Any]):
        for key, value in metadata.items():
            self.key_value_index[key][value].add(doc_id)

    def get_potential_keys(self, threshold: int = 5) -> List[str]:
        return [key for key, count in self.key_frequency.items() if count >= threshold]

    def gather_related_data(self, metadata: Dict[int, Dict], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Discover and collect documents related to the initial query results.
        This method performs a depth-first search to find documents that share common
        metadata values with the initial results. It builds a network of related
        information based on shared attributes.
        Args:
            metadata (Dict[int, Dict]): Map of all document metadata, keyed by document ID.
            results (List[Dict[str, Any]]): The initial list of documents returned by the query.
        Returns:
            List[Dict[str, Any]]: List of related documents.
        """
        related_data = []
        explored_docs = set()
        potential_keys = self.get_potential_keys()
        
        def explore_document(doc):
            if not isinstance(doc, dict):
                return
            
            doc_id = doc['metadata'].get('id')
            if doc_id is None or doc_id in explored_docs:
                return
            
            explored_docs.add(doc_id)
            related_data.append(doc)
            
            for key in potential_keys:
                if key in doc['metadata']:
                    value = doc['metadata'][key]
                    related_docs = [
                        metadata[id] for id in metadata
                        if metadata[id]['metadata'].get(key) == value and metadata[id]['metadata'].get('id') not in explored_docs
                    ]
                    for related_doc in related_docs:
                        explore_document(related_doc)

        for result in results:
            explore_document(result)
        
        return related_data

    def load_metadata(self, topic: str) -> Dict[int, Dict]:
        metadata_path = os.path.join(self.base_storage_path, f"{topic}_metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                return pickle.load(f)
        return {}

# FastAPI app and routes
app = FastAPI()
retriever = HNSWRetriever("./hnsw_index_storage")

@app.post("/index/{topic}")
async def upsert_index(topic: str, upsert_model: UpsertIndexModel):
    try:
        retriever.update_index(topic, upsert_model.documents)
        return {"message": f"Index for topic '{topic}' upserted successfully with {len(upsert_model.documents)} documents"}
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
        hnsw_size = retriever.get_index_size(topic)
        return {"topic": topic, "hnsw_size": hnsw_size}
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
