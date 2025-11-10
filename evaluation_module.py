"""
RAG Model Evaluation Module
Evaluates retrieval and generation quality
"""

import json
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
import re


class RAGEvaluator:
    """Evaluator for Retrieval-Augmented Generation models"""
    
    def __init__(self):
        self.results = []
    
    def evaluate_retrieval(self, 
                          query: str,
                          retrieved_chunks: List[Dict],
                          ground_truth_doc_ids: List[str]) -> Dict[str, float]:
        """
        Evaluate retrieval quality
        
        Args:
            query: User query
            retrieved_chunks: List of retrieved chunks with doc_ids
            ground_truth_doc_ids: List of relevant document IDs
            
        Returns:
            Dictionary with retrieval metrics
        """
        retrieved_doc_ids = [chunk['doc_id'] for chunk in retrieved_chunks]
        
        # Calculate metrics
        metrics = {
            'precision_at_k': self._precision_at_k(retrieved_doc_ids, ground_truth_doc_ids),
            'recall_at_k': self._recall_at_k(retrieved_doc_ids, ground_truth_doc_ids),
            'mrr': self._mean_reciprocal_rank(retrieved_doc_ids, ground_truth_doc_ids),
            'ndcg': self._ndcg(retrieved_chunks, ground_truth_doc_ids)
        }
        
        return metrics
    
    def evaluate_answer_quality(self,
                              query: str,
                              generated_answer: str,
                              reference_answer: str,
                              source_documents: List[Dict]) -> Dict[str, float]:
        """
        Evaluate answer generation quality
        
        Args:
            query: User query
            generated_answer: Model's generated answer
            reference_answer: Ground truth answer
            source_documents: Source documents used
            
        Returns:
            Dictionary with generation metrics
        """
        metrics = {
            'exact_match': self._exact_match(generated_answer, reference_answer),
            'token_f1': self._token_f1_score(generated_answer, reference_answer),
            'faithfulness': self._check_faithfulness(generated_answer, source_documents),
            'completeness': self._check_completeness(generated_answer, reference_answer),
            'relevance': self._check_relevance(generated_answer, query)
        }
        
        return metrics
    
    def _precision_at_k(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Precision@K"""
        if not retrieved:
            return 0.0
        relevant_set = set(relevant)
        relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant_set)
        return relevant_retrieved / len(retrieved)
    
    def _recall_at_k(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Recall@K"""
        if not relevant:
            return 0.0
        relevant_set = set(relevant)
        relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant_set)
        return relevant_retrieved / len(relevant)
    
    def _mean_reciprocal_rank(self, retrieved: List[str], relevant: List[str]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""
        relevant_set = set(relevant)
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                return 1.0 / i
        return 0.0
    
    def _ndcg(self, retrieved_chunks: List[Dict], relevant: List[str]) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        relevant_set = set(relevant)
        
        # Calculate DCG
        dcg = 0.0
        for i, chunk in enumerate(retrieved_chunks, 1):
            if chunk['doc_id'] in relevant_set:
                # Using binary relevance (1 if relevant, 0 otherwise)
                dcg += 1.0 / np.log2(i + 1)
        
        # Calculate IDCG (ideal DCG)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), len(retrieved_chunks))))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def _exact_match(self, generated: str, reference: str) -> float:
        """Check if answers match exactly (after normalization)"""
        # Normalize texts
        gen_normalized = generated.lower().strip()
        ref_normalized = reference.lower().strip()
        return 1.0 if gen_normalized == ref_normalized else 0.0
    
    def _token_f1_score(self, generated: str, reference: str) -> float:
        """Calculate token-level F1 score"""
        gen_tokens = set(generated.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not ref_tokens:
            return 0.0
        
        common = gen_tokens.intersection(ref_tokens)
        
        if not common:
            return 0.0
        
        precision = len(common) / len(gen_tokens) if gen_tokens else 0
        recall = len(common) / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def _check_faithfulness(self, answer: str, source_docs: List[Dict]) -> float:
        """
        Check if answer is faithful to source documents
        Simple heuristic: Check what percentage of answer tokens appear in sources
        """
        if not source_docs:
            return 0.0
        
        # Combine all source content
        source_text = " ".join([doc.get('content', '') for doc in source_docs]).lower()
        source_tokens = set(source_text.split())
        
        answer_tokens = answer.lower().split()
        if not answer_tokens:
            return 0.0
        
        # Count how many answer tokens appear in sources
        tokens_in_source = sum(1 for token in answer_tokens if token in source_tokens)
        
        return tokens_in_source / len(answer_tokens)
    
    def _check_completeness(self, generated: str, reference: str) -> float:
        """
        Check if generated answer covers key points from reference
        Simple heuristic: Check coverage of reference keywords
        """
        # Extract key terms from reference (simple approach)
        ref_words = reference.lower().split()
        # Filter out common words (simple stopword removal)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for'}
        key_terms = [w for w in ref_words if len(w) > 3 and w not in stopwords]
        
        if not key_terms:
            return 1.0
        
        generated_lower = generated.lower()
        covered = sum(1 for term in key_terms if term in generated_lower)
        
        return covered / len(key_terms)
    
    def _check_relevance(self, answer: str, query: str) -> float:
        """
        Check if answer is relevant to query
        Simple heuristic: Check if query keywords appear in answer
        """
        query_words = query.lower().split()
        # Filter out common words
        stopwords = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an'}
        query_keywords = [w for w in query_words if w not in stopwords and len(w) > 2]
        
        if not query_keywords:
            return 1.0
        
        answer_lower = answer.lower()
        relevant_terms = sum(1 for keyword in query_keywords if keyword in answer_lower)
        
        return relevant_terms / len(query_keywords)
    
    def run_evaluation(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """
        Run evaluation on multiple test cases
        
        Args:
            test_cases: List of test cases, each containing:
                - query: str
                - ground_truth_docs: List[str] (relevant doc IDs)
                - reference_answer: str
                - generated_answer: str
                - retrieved_chunks: List[Dict]
                
        Returns:
            Aggregated evaluation results
        """
        all_results = []
        
        for test_case in test_cases:
            # Evaluate retrieval
            retrieval_metrics = self.evaluate_retrieval(
                test_case['query'],
                test_case['retrieved_chunks'],
                test_case['ground_truth_docs']
            )
            
            # Evaluate generation
            generation_metrics = self.evaluate_answer_quality(
                test_case['query'],
                test_case['generated_answer'],
                test_case['reference_answer'],
                test_case['retrieved_chunks']
            )
            
            all_results.append({
                'retrieval': retrieval_metrics,
                'generation': generation_metrics
            })
        
        # Calculate averages
        avg_results = {
            'retrieval': {},
            'generation': {},
            'total_cases': len(test_cases)
        }
        
        # Average retrieval metrics
        for metric in ['precision_at_k', 'recall_at_k', 'mrr', 'ndcg']:
            avg_results['retrieval'][metric] = np.mean([r['retrieval'][metric] for r in all_results])
        
        # Average generation metrics
        for metric in ['exact_match', 'token_f1', 'faithfulness', 'completeness', 'relevance']:
            avg_results['generation'][metric] = np.mean([r['generation'][metric] for r in all_results])
        
        # Calculate overall F1 for retrieval
        avg_precision = avg_results['retrieval']['precision_at_k']
        avg_recall = avg_results['retrieval']['recall_at_k']
        if avg_precision + avg_recall > 0:
            avg_results['retrieval']['f1'] = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        else:
            avg_results['retrieval']['f1'] = 0.0
        
        return avg_results
    
    def create_test_suite(self, queries_file: str) -> List[Dict]:
        """
        Load test suite from JSON file
        
        Expected JSON format:
        [
            {
                "query": "What is machine learning?",
                "ground_truth_docs": ["doc1", "doc2"],
                "reference_answer": "Machine learning is..."
            },
            ...
        ]
        """
        with open(queries_file, 'r') as f:
            return json.load(f)
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")


# Example usage
if __name__ == "__main__":
    # Example test case
    evaluator = RAGEvaluator()
    
    test_cases = [
        {
            "query": "What is machine learning?",
            "ground_truth_docs": ["doc1", "doc2"],
            "reference_answer": "Machine learning is a subset of AI that enables systems to learn from data.",
            "generated_answer": "Machine learning is a type of artificial intelligence that allows systems to learn from data without being explicitly programmed.",
            "retrieved_chunks": [
                {"doc_id": "doc1", "content": "ML content...", "score": 0.9},
                {"doc_id": "doc3", "content": "Other content...", "score": 0.8}
            ]
        }
    ]
    
    results = evaluator.run_evaluation(test_cases)
    print("Evaluation Results:")
    print(json.dumps(results, indent=2))