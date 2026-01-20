"""
OpenSearch Vector Store Implementation
Replaces FAISS for vector similarity search
"""

import json
import pickle
import os
from typing import List, Dict, Optional, Any
import numpy as np
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import NotFoundError
import logging

logger = logging.getLogger(__name__)


class OpenSearchVectorStore:
    """OpenSearch-based vector store for similarity search"""
    
    def __init__(self, config: Dict[str, Any], base_path: str):
        """
        Initialize OpenSearch vector store
        
        Args:
            config: Configuration dictionary
            base_path: Path to vector store directory
        """
        self.config = config
        self.base_path = base_path
        
        # OpenSearch configuration
        opensearch_config = config.get('opensearch', {})
        self.host = opensearch_config.get('host', 'localhost')
        self.port = opensearch_config.get('port', 9200)
        self.username = opensearch_config.get('username')
        self.password = opensearch_config.get('password')
        self.use_ssl = opensearch_config.get('use_ssl', False)
        self.verify_certs = opensearch_config.get('verify_certs', False)
        self.index_name = opensearch_config.get('index_name', 'document_embeddings')
        
        # Initialize OpenSearch client
        self.client = self._create_client()
        
        # Load BM25 index and chunks (still needed)
        self.load_traditional_indexes()
    
    def _create_client(self) -> OpenSearch:
        """Create OpenSearch client"""
        auth = None
        if self.username and self.password:
            auth = (self.username, self.password)
        
        client = OpenSearch(
            hosts=[{'host': self.host, 'port': self.port}],
            http_auth=auth,
            use_ssl=self.use_ssl,
            verify_certs=self.verify_certs,
            connection_class=RequestsHttpConnection,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
        
        logger.info(f"OpenSearch client created: {self.host}:{self.port}")
        return client
    
    def load_traditional_indexes(self):
        """Load BM25 index and chunk metadata (still needed)"""
        # Load BM25 index
        bm25_path = os.path.join(self.base_path, 'bm25_index', 'bm25.pkl')
        if not os.path.exists(bm25_path):
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}")
        with open(bm25_path, 'rb') as f:
            self.bm25_index = pickle.load(f)
        
        # Load chunk metadata
        metadata_path = os.path.join(self.base_path, 'chunks', 'chunk_metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Chunk metadata not found at {metadata_path}")
        with open(metadata_path, 'r') as f:
            self.chunks = json.load(f)
        
        logger.info("BM25 index and chunk metadata loaded successfully")
    
    def create_index(self, dimension: int = 768):
        """
        Create OpenSearch index with vector field
        
        Args:
            dimension: Embedding dimension (default 768 for many models)
        """
        index_mapping = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    },
                    "entitlement": {"type": "keyword"},
                    "org_id": {"type": "keyword"},
                    "tags": {"type": "keyword"}
                }
            },
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            }
        }
        
        try:
            self.client.indices.create(index=self.index_name, body=index_mapping)
            logger.info(f"Created OpenSearch index: {self.index_name}")
        except Exception as e:
            if "resource_already_exists_exception" in str(e):
                logger.info(f"Index {self.index_name} already exists")
            else:
                raise
    
    def index_documents(self, documents: List[Dict]):
        """
        Index documents with embeddings
        
        Args:
            documents: List of documents with embeddings
                      Each doc should have: chunk_id, doc_id, title, content, embedding, etc.
        """
        for doc in documents:
            try:
                self.client.index(
                    index=self.index_name,
                    id=doc['chunk_id'],
                    body=doc
                )
            except Exception as e:
                logger.error(f"Failed to index document {doc.get('chunk_id')}: {e}")
        
        # Refresh index to make documents searchable immediately
        self.client.indices.refresh(index=self.index_name)
        logger.info(f"Indexed {len(documents)} documents")
    
    def vector_search(self, query_embedding: np.ndarray, top_k: int = 10, 
                     filters: Optional[Dict] = None) -> List[Dict]:
        """
        Perform vector similarity search
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filters: Optional filters (e.g., entitlement, org_id)
            
        Returns:
            List of similar documents with scores
        """
        # Build query
        knn_query = {
            "knn": {
                "embedding": {
                    "vector": query_embedding.tolist(),
                    "k": top_k
                }
            }
        }
        
        # Add filters if provided
        if filters:
            query_body = {
                "query": {
                    "bool": {
                        "must": [knn_query],
                        "filter": []
                    }
                }
            }
            
            # Add entitlement filter
            if 'entitlement' in filters:
                query_body["query"]["bool"]["filter"].append({
                    "terms": {"entitlement": filters['entitlement']}
                })
            
            # Add org_id filter
            if 'org_id' in filters:
                query_body["query"]["bool"]["filter"].append({
                    "term": {"org_id": filters['org_id']}
                })
            
            # Add tags filter
            if 'tags' in filters and filters['tags']:
                query_body["query"]["bool"]["filter"].append({
                    "terms": {"tags": filters['tags']}
                })
        else:
            query_body = {"query": knn_query}
        
        try:
            response = self.client.search(
                index=self.index_name,
                body=query_body,
                size=top_k
            )
            
            results = []
            for hit in response['hits']['hits']:
                doc = hit['_source']
                doc['score'] = hit['_score']
                results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, query_embedding: np.ndarray, 
                     entitlement: List[str], org_id: str = None,
                     tags: List[str] = None, top_k: int = 5,
                     vector_weight: float = 0.7, bm25_weight: float = 0.3) -> List[Dict]:
        """
        Perform hybrid search combining OpenSearch vectors and BM25
        
        Args:
            query: Text query
            query_embedding: Query embedding vector
            entitlement: User entitlements
            org_id: Organization ID
            tags: Tags filter
            top_k: Number of results
            vector_weight: Weight for vector scores
            bm25_weight: Weight for BM25 scores
            
        Returns:
            List of ranked documents
        """
        # Vector search with OpenSearch
        filters = {
            'entitlement': entitlement,
            'org_id': org_id,
            'tags': tags
        }
        
        retrieval_multiplier = 5
        initial_top_k = min(top_k * retrieval_multiplier, len(self.chunks))
        
        vector_results = self.vector_search(
            query_embedding, 
            top_k=initial_top_k, 
            filters=filters
        )
        
        # BM25 search (same as before)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Normalize scores
        def normalize(scores):
            if len(scores) == 0:
                return scores
            min_s, max_s = scores.min(), scores.max()
            if max_s - min_s < 1e-10:
                return np.zeros_like(scores)
            return (scores - min_s) / (max_s - min_s)
        
        # Get vector scores and indices
        vector_scores = np.array([r['score'] for r in vector_results])
        vector_indices = [self._get_chunk_index(r['chunk_id']) for r in vector_results]
        
        vector_scores_norm = normalize(vector_scores)
        bm25_scores_norm = normalize(bm25_scores)
        
        # Compute hybrid scores
        hybrid_scores = {}
        
        # Add vector scores
        for idx, score in zip(vector_indices, vector_scores_norm):
            if idx is not None:  # Valid chunk index
                hybrid_scores[idx] = score * vector_weight
        
        # Add BM25 scores
        for idx, score in enumerate(bm25_scores_norm):
            if idx in hybrid_scores:
                hybrid_scores[idx] += score * bm25_weight
            else:
                hybrid_scores[idx] = score * bm25_weight
        
        # Sort and return top results
        sorted_indices = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in sorted_indices[:top_k]:
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(score)
                results.append(chunk)
        
        return results
    
    def _get_chunk_index(self, chunk_id: str) -> Optional[int]:
        """Get chunk index from chunk_id"""
        for i, chunk in enumerate(self.chunks):
            if chunk.get('chunk_id') == chunk_id:
                return i
        return None
    
    def health_check(self) -> bool:
        """Check if OpenSearch is healthy"""
        try:
            health = self.client.cluster.health()
            return health['status'] in ['green', 'yellow']
        except Exception as e:
            logger.error(f"OpenSearch health check failed: {e}")
            return False