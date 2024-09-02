import lmdb
import json
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod

class StorageBackend(ABC):
    @abstractmethod
    def save_bucket(self, key, vectors, metadata):
        pass

    @abstractmethod
    def load_bucket(self, key):
        pass

    @abstractmethod
    def bucket_exists(self, key):
        pass

    @abstractmethod
    def list_buckets(self):
        pass

    @abstractmethod
    def delete_bucket(self, key):
        pass

class LMDBStorage(StorageBackend):
    def __init__(self, path):
        self.env = lmdb.open(path, map_size=1024*1024*1024)

    def save_bucket(self, key, vectors, metadata):
        with self.env.begin(write=True) as txn:
            txn.put(key.encode(), json.dumps({'vectors': vectors.tolist(), 'metadata': metadata}).encode())

    def load_bucket(self, key):
        with self.env.begin() as txn:
            data = json.loads(txn.get(key.encode()).decode())
            return np.array(data['vectors']), data['metadata']

    def bucket_exists(self, key):
        with self.env.begin() as txn:
            return txn.get(key.encode()) is not None

    def list_buckets(self):
        with self.env.begin() as txn:
            return [key.decode() for key, _ in txn.cursor()]

    def delete_bucket(self, key):
        with self.env.begin(write=True) as txn:
            txn.delete(key.encode())

class Streamhouse:
    def __init__(self, storage_backend: StorageBackend, dim: int = 128, bucket_size: timedelta = timedelta(hours=1)):
        self.storage = storage_backend
        self.dim = dim
        self.bucket_size = bucket_size

    def _get_bucket_key(self, timestamp: datetime) -> str:
        return timestamp.replace(minute=0, second=0, microsecond=0).isoformat()

    def insert(self, timestamp: datetime, vector: np.ndarray, metadata: Dict[str, Any]):
        bucket_key = self._get_bucket_key(timestamp)
        vectors, existing_metadata = self.storage.load_bucket(bucket_key) if self.storage.bucket_exists(bucket_key) else (np.empty((0, self.dim)), [])
        
        vectors = np.vstack([vectors, vector])
        existing_metadata.append({**metadata, 'timestamp': timestamp.isoformat()})
        
        self.storage.save_bucket(bucket_key, vectors, existing_metadata)

    def search(self, query_vector: np.ndarray, start_time: datetime, end_time: datetime, top_k: int = 10) -> List[Dict[str, Any]]:
        results = []
        current_time = start_time
        while current_time <= end_time:
            bucket_key = self._get_bucket_key(current_time)
            if self.storage.bucket_exists(bucket_key):
                vectors, metadata = self.storage.load_bucket(bucket_key)
                distances = np.linalg.norm(vectors - query_vector, axis=1)
                top_indices = np.argsort(distances)[:top_k]
                for idx in top_indices:
                    results.append({
                        'distance': float(distances[idx]),
                        'metadata': metadata[idx],
                        'bucket': bucket_key
                    })
            current_time += self.bucket_size
        
        results.sort(key=lambda x: x['distance'])
        return results[:top_k]

    def get_time_range(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        results = []
        current_time = start_time
        while current_time <= end_time:
            bucket_key = self._get_bucket_key(current_time)
            if self.storage.bucket_exists(bucket_key):
                _, metadata = self.storage.load_bucket(bucket_key)
                results.extend(metadata)
            current_time += self.bucket_size
        return results

    def implement_retention_policy(self, retention_period: timedelta):
        current_time = datetime.now()
        for bucket_key in self.storage.list_buckets():
            bucket_time = datetime.fromisoformat(bucket_key)
            if current_time - bucket_time > retention_period:
                self.storage.delete_bucket(bucket_key)

# Usage example
if __name__ == "__main__":
    storage = LMDBStorage("/tmp/streamhouse_db")
    sh = Streamhouse(storage)

    # Insert some time series data
    base_time = datetime.now()
    for i in range(100):
        timestamp = base_time + timedelta(minutes=i)
        vector = np.random.rand(128).astype(np.float32)
        metadata = {"id": i, "text": f"Document {i}"}
        sh.insert(timestamp, vector, metadata)

    # Perform a search over a time range
    query = np.random.rand(128).astype(np.float32)
    start_time = base_time
    end_time = base_time + timedelta(hours=1)
    results = sh.search(query, start_time, end_time, top_k=5)

    print("Search Results:")
    for result in results:
        print(f"Distance: {result['distance']:.4f}, Bucket: {result['bucket']}, Metadata: {result['metadata']}")

    # Get all data in a time range
    time_range_data = sh.get_time_range(start_time, end_time)
    print(f"\nData points in time range: {len(time_range_data)}")

    # Implement retention policy
    sh.implement_retention_policy(timedelta(days=7))
