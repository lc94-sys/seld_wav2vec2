# Evaluation Framework for RAG System
# Tests accuracy, relevance, and performance

import yaml
import json
import time
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
import pandas as pd

# Import retriever (assuming notebook 03 is available)
from notebook_03_retrieval import HybridRetriever

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize retriever
retriever = HybridRetriever(config)

print("="*70)
print("🎯 RAG SYSTEM EVALUATION FRAMEWORK")
print("="*70)

# ============================================
# 1. DEFINE TEST DATASET
# ============================================

# Create evaluation dataset with ground truth
# Format: {query, expected_docs, expected_answer_keywords, entitlement}

EVAL_DATASET = [
    {
        'id': 'Q1',
        'query': 'How do I cancel a booking?',
        'entitlement': 'agent_support',
        'org_id': 'org_123',
        'tags': ['cancellation'],
        'expected_docs': ['Cancellation Procedure'],  # Ground truth documents
        'expected_keywords': ['cancel', 'booking', 'procedure', 'steps'],  # Keywords that should appear
        'category': 'cancellation'
    },
    {
        'id': 'Q2',
        'query': 'What is the refund policy?',
        'entitlement': 'agent_support',
        'org_id': 'org_123',
        'tags': ['cancellation'],
        'expected_docs': ['Cancellation Procedure'],
        'expected_keywords': ['refund', 'policy', 'amount'],
        'category': 'cancellation'
    },
    {
        'id': 'Q3',
        'query': 'How do I create a new booking?',
        'entitlement': 'agent_sales',
        'org_id': 'org_123',
        'tags': ['booking'],
        'expected_docs': ['Booking Procedure'],
        'expected_keywords': ['create', 'booking', 'new', 'customer'],
        'category': 'booking'
    },
    {
        'id': 'Q4',
        'query': 'How to modify an existing reservation?',
        'entitlement': 'agent_support',
        'org_id': 'org_123',
        'tags': ['update'],
        'expected_docs': ['Update Procedure'],
        'expected_keywords': ['modify', 'update', 'change', 'reservation'],
        'category': 'update'
    },
    {
        'id': 'Q5',
        'query': 'What documents are required for cancellation?',
        'entitlement': 'agent_support',
        'org_id': 'org_123',
        'tags': ['cancellation'],
        'expected_docs': ['Cancellation Procedure'],
        'expected_keywords': ['documents', 'required', 'cancellation'],
        'category': 'cancellation'
    }
]

# Add more test cases as needed
print(f"\n✓ Loaded {len(EVAL_DATASET)} test cases")

# ============================================
# 2. RETRIEVAL METRICS
# ============================================

def calculate_retrieval_metrics(retrieved_docs: List[str], expected_docs: List[str]) -> Dict:
    """
    Calculate retrieval quality metrics
    
    Metrics:
    - Precision: What % of retrieved docs are relevant?
    - Recall: What % of relevant docs were retrieved?
    - F1 Score: Harmonic mean of precision and recall
    - MRR (Mean Reciprocal Rank): Position of first relevant doc
    """
    retrieved_set = set(retrieved_docs)
    expected_set = set(expected_docs)
    
    # True Positives: docs that are both retrieved and relevant
    tp = len(retrieved_set.intersection(expected_set))
    
    # Precision: TP / (TP + FP) = relevant retrieved / all retrieved
    precision = tp / len(retrieved_set) if retrieved_set else 0
    
    # Recall: TP / (TP + FN) = relevant retrieved / all relevant
    recall = tp / len(expected_set) if expected_set else 0
    
    # F1 Score: 2 * (P * R) / (P + R)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Mean Reciprocal Rank: 1 / position of first relevant doc
    mrr = 0
    for i, doc in enumerate(retrieved_docs, 1):
        if doc in expected_set:
            mrr = 1 / i
            break
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mrr': mrr,
        'retrieved_count': len(retrieved_set),
        'expected_count': len(expected_set),
        'correct_count': tp
    }


def calculate_ndcg(retrieved_docs: List[str], expected_docs: List[str], k: int = 5) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K)
    Measures ranking quality with position discount
    """
    # Create relevance scores (1 if relevant, 0 if not)
    relevance = [1 if doc in expected_docs else 0 for doc in retrieved_docs[:k]]
    
    # DCG: sum of (relevance / log2(position + 1))
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
    
    # Ideal DCG: best possible ranking
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
    
    # NDCG: DCG / IDCG
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return ndcg


# ============================================
# 3. ANSWER QUALITY METRICS
# ============================================

def calculate_answer_metrics(answer: str, expected_keywords: List[str]) -> Dict:
    """
    Calculate answer quality metrics
    
    Metrics:
    - Keyword Coverage: % of expected keywords present
    - Answer Length: Character count
    - Keyword Density: Keywords per 100 words
    """
    answer_lower = answer.lower()
    
    # Count how many expected keywords appear in answer
    found_keywords = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    keyword_coverage = found_keywords / len(expected_keywords) if expected_keywords else 0
    
    # Answer length
    answer_length = len(answer)
    word_count = len(answer.split())
    
    # Keyword density
    keyword_density = (found_keywords / word_count * 100) if word_count > 0 else 0
    
    return {
        'keyword_coverage': keyword_coverage,
        'keywords_found': found_keywords,
        'keywords_expected': len(expected_keywords),
        'answer_length': answer_length,
        'word_count': word_count,
        'keyword_density': keyword_density
    }


# ============================================
# 4. PERFORMANCE METRICS
# ============================================

def measure_performance(func, *args, **kwargs) -> Tuple[any, float]:
    """Measure execution time of a function"""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed


# ============================================
# 5. RUN EVALUATION
# ============================================

def run_evaluation(dataset: List[Dict], retriever: HybridRetriever) -> pd.DataFrame:
    """Run complete evaluation on dataset"""
    
    results = []
    
    print("\n" + "="*70)
    print("Running Evaluation...")
    print("="*70)
    
    for i, test_case in enumerate(dataset, 1):
        print(f"\n[{i}/{len(dataset)}] Evaluating: {test_case['id']}")
        print(f"Query: {test_case['query']}")
        
        # Create session
        session_id = retriever.create_session(
            user_id=f"eval_user_{i}",
            entitlement=test_case['entitlement'],
            org_id=test_case['org_id']
        )
        
        try:
            # Measure query time
            result, query_time = measure_performance(
                retriever.query_with_session,
                session_id=session_id,
                query=test_case['query'],
                tags=test_case.get('tags'),
                top_k=5,
                history_limit=0  # No history for eval
            )
            
            # Extract retrieved document titles
            retrieved_docs = [src['title'] for src in result['sources']]
            
            # Calculate retrieval metrics
            retrieval_metrics = calculate_retrieval_metrics(
                retrieved_docs,
                test_case['expected_docs']
            )
            
            # Calculate NDCG
            ndcg = calculate_ndcg(retrieved_docs, test_case['expected_docs'], k=5)
            
            # Calculate answer quality metrics
            answer_metrics = calculate_answer_metrics(
                result['answer'],
                test_case['expected_keywords']
            )
            
            # Check entitlement filtering
            entitlement_correct = all(
                src['title'] in test_case['expected_docs'] or 
                'universal' in test_case.get('allowed_entitlements', [test_case['entitlement']])
                for src in result['sources']
            )
            
            # Compile results
            eval_result = {
                'test_id': test_case['id'],
                'query': test_case['query'],
                'category': test_case['category'],
                'entitlement': test_case['entitlement'],
                
                # Retrieval Metrics
                'precision': retrieval_metrics['precision'],
                'recall': retrieval_metrics['recall'],
                'f1_score': retrieval_metrics['f1_score'],
                'mrr': retrieval_metrics['mrr'],
                'ndcg@5': ndcg,
                
                # Answer Quality
                'keyword_coverage': answer_metrics['keyword_coverage'],
                'answer_length': answer_metrics['answer_length'],
                'word_count': answer_metrics['word_count'],
                
                # Performance
                'query_time_sec': query_time,
                
                # Details
                'retrieved_docs': ', '.join(retrieved_docs),
                'expected_docs': ', '.join(test_case['expected_docs']),
                'answer_preview': result['answer'][:100] + '...',
                
                # Pass/Fail
                'retrieval_pass': retrieval_metrics['f1_score'] >= 0.5,
                'answer_pass': answer_metrics['keyword_coverage'] >= 0.5,
                'performance_pass': query_time < 5.0,
                'overall_pass': (
                    retrieval_metrics['f1_score'] >= 0.5 and 
                    answer_metrics['keyword_coverage'] >= 0.5 and 
                    query_time < 5.0
                )
            }
            
            results.append(eval_result)
            
            # Print summary
            print(f"  ✓ F1: {retrieval_metrics['f1_score']:.3f} | "
                  f"Keyword Coverage: {answer_metrics['keyword_coverage']:.1%} | "
                  f"Time: {query_time:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results.append({
                'test_id': test_case['id'],
                'query': test_case['query'],
                'error': str(e),
                'overall_pass': False
            })
    
    return pd.DataFrame(results)


# Run evaluation
eval_results = run_evaluation(EVAL_DATASET, retriever)

# ============================================
# 6. GENERATE EVALUATION REPORT
# ============================================

print("\n" + "="*70)
print("📊 EVALUATION REPORT")
print("="*70)

# Overall Statistics
print("\n--- Overall Performance ---")
print(f"Total Test Cases: {len(eval_results)}")
print(f"Passed: {eval_results['overall_pass'].sum()} ({eval_results['overall_pass'].mean():.1%})")
print(f"Failed: {(~eval_results['overall_pass']).sum()}")

# Retrieval Metrics Summary
print("\n--- Retrieval Quality (Average) ---")
print(f"Precision: {eval_results['precision'].mean():.3f}")
print(f"Recall: {eval_results['recall'].mean():.3f}")
print(f"F1 Score: {eval_results['f1_score'].mean():.3f}")
print(f"MRR: {eval_results['mrr'].mean():.3f}")
print(f"NDCG@5: {eval_results['ndcg@5'].mean():.3f}")

# Answer Quality Summary
print("\n--- Answer Quality (Average) ---")
print(f"Keyword Coverage: {eval_results['keyword_coverage'].mean():.1%}")
print(f"Average Answer Length: {eval_results['answer_length'].mean():.0f} characters")
print(f"Average Word Count: {eval_results['word_count'].mean():.0f} words")

# Performance Summary
print("\n--- Performance (Average) ---")
print(f"Query Time: {eval_results['query_time_sec'].mean():.2f}s")
print(f"Min Time: {eval_results['query_time_sec'].min():.2f}s")
print(f"Max Time: {eval_results['query_time_sec'].max():.2f}s")
print(f"95th Percentile: {eval_results['query_time_sec'].quantile(0.95):.2f}s")

# By Category
print("\n--- Performance by Category ---")
category_stats = eval_results.groupby('category').agg({
    'f1_score': 'mean',
    'keyword_coverage': 'mean',
    'query_time_sec': 'mean',
    'overall_pass': lambda x: f"{x.sum()}/{len(x)}"
}).round(3)
print(category_stats)

# Failed Cases
failed_cases = eval_results[~eval_results['overall_pass']]
if not failed_cases.empty:
    print("\n--- Failed Test Cases ---")
    for _, row in failed_cases.iterrows():
        print(f"\n{row['test_id']}: {row['query']}")
        if 'error' in row:
            print(f"  Error: {row['error']}")
        else:
            print(f"  F1: {row.get('f1_score', 'N/A'):.3f}")
            print(f"  Keyword Coverage: {row.get('keyword_coverage', 'N/A'):.1%}")
            print(f"  Time: {row.get('query_time_sec', 'N/A'):.2f}s")

# Save detailed results
output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
eval_results.to_csv(output_file, index=False)
print(f"\n✓ Detailed results saved to: {output_file}")

# ============================================
# 7. VISUALIZATIONS (Optional)
# ============================================

try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Retrieval Metrics
    metrics = ['precision', 'recall', 'f1_score', 'mrr', 'ndcg@5']
    values = [eval_results[m].mean() for m in metrics]
    axes[0, 0].bar(metrics, values, color='skyblue')
    axes[0, 0].set_title('Retrieval Metrics (Average)')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel('Score')
    
    # Plot 2: Query Time Distribution
    axes[0, 1].hist(eval_results['query_time_sec'], bins=10, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Query Time Distribution')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Frequency')
    
    # Plot 3: Pass Rate by Category
    category_pass = eval_results.groupby('category')['overall_pass'].mean()
    axes[1, 0].bar(category_pass.index, category_pass.values, color='coral')
    axes[1, 0].set_title('Pass Rate by Category')
    axes[1, 0].set_ylabel('Pass Rate')
    axes[1, 0].set_ylim(0, 1)
    
    # Plot 4: F1 Score vs Query Time
    axes[1, 1].scatter(eval_results['query_time_sec'], eval_results['f1_score'], 
                       c=eval_results['overall_pass'], cmap='RdYlGn', alpha=0.6)
    axes[1, 1].set_title('F1 Score vs Query Time')
    axes[1, 1].set_xlabel('Query Time (seconds)')
    axes[1, 1].set_ylabel('F1 Score')
    
    plt.tight_layout()
    plot_file = f"evaluation_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_file)
    print(f"✓ Plots saved to: {plot_file}")
    plt.show()
    
except ImportError:
    print("\n⚠️  matplotlib not available. Skipping visualizations.")

# ============================================
# 8. RECOMMENDATIONS
# ============================================

print("\n" + "="*70)
print("💡 RECOMMENDATIONS")
print("="*70)

avg_f1 = eval_results['f1_score'].mean()
avg_coverage = eval_results['keyword_coverage'].mean()
avg_time = eval_results['query_time_sec'].mean()

if avg_f1 < 0.7:
    print("\n⚠️  Retrieval Quality (F1 < 0.7)")
    print("  Suggestions:")
    print("  - Adjust hybrid weights (try 0.7 vector / 0.3 keyword)")
    print("  - Increase retrieval top_k")
    print("  - Review semantic chunking threshold")
    print("  - Add more training documents")

if avg_coverage < 0.6:
    print("\n⚠️  Answer Quality (Keyword Coverage < 60%)")
    print("  Suggestions:")
    print("  - Improve prompts to LLM")
    print("  - Increase context chunks passed to LLM")
    print("  - Review expected keywords (may be too strict)")

if avg_time > 3.0:
    print("\n⚠️  Performance (Average time > 3s)")
    print("  Suggestions:")
    print("  - Enable embedding caching")
    print("  - Reduce max_tokens (currently using 512)")
    print("  - Use smaller top_k value")
    print("  - Consider GPU inference")

if avg_f1 >= 0.7 and avg_coverage >= 0.6 and avg_time <= 3.0:
    print("\n✅ System performing well!")
    print("   F1 Score ≥ 0.7")
    print("   Keyword Coverage ≥ 60%")
    print("   Average Response Time ≤ 3s")

print("\n" + "="*70)
print("EVALUATION COMPLETE")
print("="*70)
