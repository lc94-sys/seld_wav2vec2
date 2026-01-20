"""
Native OpenSearch Hybrid Search Implementation
Uses OpenSearch's built-in hybrid search capabilities
"""

import json
import os
from typing import List, Dict, Optional, Any
import numpy as np
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import NotFoundError
import logging

logger = logging.getLogger(__name__)


class NativeHybridOpenSearch:
    """OpenSearch with native hybrid search (vector + text in single query)"""
    
    def __init__(self, config: Dict[str, Any], base_path: str):
        """
        Initialize OpenSearch with hybrid search capabilities
        
        Args:
            config: Configuration dictionary
            base_path: Path to vector store directory (for chunk metadata if needed)
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
        
        logger.info("Native Hybrid OpenSearch initialized")
    
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
    
    def create_hybrid_index(self, dimension: int = 768):
        """
        Create OpenSearch index optimized for hybrid search
        
        Args:
            dimension: Embedding dimension
        """
        index_mapping = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
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
                    "tags": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"}
                }
            },
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            }
        }
        
        try:
            self.client.indices.create(index=self.index_name, body=index_mapping)
            logger.info(f"Created hybrid search index: {self.index_name}")
        except Exception as e:
            if "resource_already_exists_exception" in str(e):
                logger.info(f"Index {self.index_name} already exists")
            else:
                raise
    
    def index_documents(self, documents: List[Dict]):
        """
        Index documents with embeddings and full text content
        
        Args:
            documents: List of documents with embeddings and metadata
        """
        for doc in documents:
            try:
                # Prepare document for indexing
                index_doc = {
                    "chunk_id": doc["chunk_id"],
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "content": doc["content"],
                    "embedding": doc["embedding"],  # Should be list, not numpy array
                    "entitlement": doc.get("entitlement", []),
                    "org_id": doc.get("org_id"),
                    "tags": doc.get("tags", []),
                    "created_at": doc.get("created_at", "2024-01-01"),
                    "updated_at": doc.get("updated_at", "2024-01-01")
                }
                
                self.client.index(
                    index=self.index_name,
                    id=doc['chunk_id'],
                    body=index_doc
                )
            except Exception as e:
                logger.error(f"Failed to index document {doc.get('chunk_id')}: {e}")
        
        # Refresh index to make documents searchable immediately
        self.client.indices.refresh(index=self.index_name)
        logger.info(f"Indexed {len(documents)} documents")
    
    def hybrid_search(self, 
                     query: str, 
                     query_embedding: np.ndarray,
                     entitlement: List[str] = None,
                     org_id: str = None,
                     tags: List[str] = None,
                     top_k: int = 5,
                     vector_weight: float = 0.7,
                     text_weight: float = 0.3) -> List[Dict]:
        """
        Native hybrid search combining vector similarity and text matching in single query
        
        Args:
            query: Text query
            query_embedding: Query embedding vector
            entitlement: User entitlements for filtering
            org_id: Organization ID for filtering  
            tags: Tags for filtering
            top_k: Number of results to return
            vector_weight: Weight for vector similarity
            text_weight: Weight for text matching
            
        Returns:
            List of ranked documents with hybrid scores
        """
        # Build the hybrid search query
        search_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        # Vector similarity search
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding.tolist(),
                                    "k": top_k * 2,  # Get more candidates
                                    "boost": vector_weight
                                }
                            }
                        },
                        # Text matching (BM25-like)
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "title^2",    # Title has 2x weight
                                    "content^1"   # Content has 1x weight
                                ],
                                "type": "best_fields",
                                "boost": text_weight
                            }
                        }
                    ],
                    "minimum_should_match": 1  # At least one should clause must match
                }
            }
        }
        
        # Add filters if provided
        if entitlement or org_id or tags:
            search_body["query"]["bool"]["filter"] = []
            
            # Entitlement filter
            if entitlement:
                search_body["query"]["bool"]["filter"].append({
                    "terms": {"entitlement": entitlement}
                })
            
            # Organization filter
            if org_id:
                search_body["query"]["bool"]["filter"].append({
                    "term": {"org_id": org_id}
                })
            
            # Tags filter
            if tags:
                search_body["query"]["bool"]["filter"].append({
                    "terms": {"tags": tags}
                })
        
        try:
            # Execute search
            response = self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            # Process results
            results = []
            for hit in response['hits']['hits']:
                doc = hit['_source']
                doc['score'] = float(hit['_score'])  # OpenSearch's hybrid score
                results.append(doc)
            
            logger.info(f"Hybrid search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def pure_text_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Text-only search for comparison"""
        search_body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "content"],
                    "type": "best_fields"
                }
            }
        }
        
        response = self.client.search(index=self.index_name, body=search_body)
        return [hit['_source'] for hit in response['hits']['hits']]
    
    def pure_vector_search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Vector-only search for comparison"""
        search_body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding.tolist(),
                        "k": top_k
                    }
                }
            }
        }
        
        response = self.client.search(index=self.index_name, body=search_body)
        return [hit['_source'] for hit in response['hits']['hits']]
    
    def get_document_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Get a specific document by chunk_id"""
        try:
            response = self.client.get(index=self.index_name, id=chunk_id)
            return response['_source']
        except NotFoundError:
            return None
    
    def update_document(self, chunk_id: str, updates: Dict):
        """Update a specific document"""
        try:
            self.client.update(
                index=self.index_name,
                id=chunk_id,
                body={"doc": updates}
            )
        except Exception as e:
            logger.error(f"Failed to update document {chunk_id}: {e}")
    
    def delete_document(self, chunk_id: str):
        """Delete a specific document"""
        try:
            self.client.delete(index=self.index_name, id=chunk_id)
        except Exception as e:
            logger.error(f"Failed to delete document {chunk_id}: {e}")
    
    def get_index_stats(self) -> Dict:
        """Get index statistics"""
        try:
            stats = self.client.indices.stats(index=self.index_name)
            return {
                "total_docs": stats['indices'][self.index_name]['total']['docs']['count'],
                "index_size": stats['indices'][self.index_name]['total']['store']['size_in_bytes'],
                "search_queries": stats['indices'][self.index_name]['total']['search']['query_total']
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check if OpenSearch is healthy"""
        try:
            health = self.client.cluster.health()
            index_exists = self.client.indices.exists(index=self.index_name)
            return health['status'] in ['green', 'yellow'] and index_exists
        except Exception as e:
            logger.error(f"OpenSearch health check failed: {e}")
            return False