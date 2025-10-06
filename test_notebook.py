# Testing Session Management and Context Preservation
# Run this notebook after notebooks 01, 02, and 03

import yaml
import sys
sys.path.append('.')

# Import the retriever from notebook 03
from notebook_03_retrieval import HybridRetriever

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize base retriever
print("Initializing retriever...")
retriever = HybridRetriever(config)

# ============================================
# TEST 1: Basic Context Preservation
# ============================================
print("\n" + "="*60)
print("TEST 1: Context Preservation in Conversation")
print("="*60)

# Simulate a conversation with follow-up questions
conversation = [
    {"query": "How do I cancel a booking?", "tags": ["cancellation"]},
    {"query": "What documents do I need?", "tags": ["cancellation"]},  # Should understand context
    {"query": "How long does the process take?", "tags": ["cancellation"]}  # Should know "process" = cancellation
]

# Store conversation history manually
history = []

for i, turn in enumerate(conversation, 1):
    print(f"\n--- Turn {i} ---")
    print(f"User: {turn['query']}")
    
    # Build prompt with history
    if history:
        history_text = "\nPrevious conversation:\n"
        for h in history[-2:]:  # Last 2 turns
            history_text += f"User: {h['query']}\nAssistant: {h['answer'][:100]}...\n"
        
        # You'd need to modify the retriever to accept history
        # For now, just retrieve normally
        result = retriever.query(
            query=turn['query'],
            entitlement="agent_support",
            org_id="org_123",
            tags=turn.get('tags')
        )
    else:
        result = retriever.query(
            query=turn['query'],
            entitlement="agent_support",
            org_id="org_123",
            tags=turn.get('tags')
        )
    
    print(f"Assistant: {result['answer'][:200]}...")
    print(f"Sources used: {[s['title'] for s in result['sources']]}")
    
    # Store in history
    history.append({
        'query': turn['query'],
        'answer': result['answer']
    })

print("\n✓ Context preservation test completed")
print("Manual check: Did follow-up questions make sense given context?")


# ============================================
# TEST 2: Session Isolation
# ============================================
print("\n" + "="*60)
print("TEST 2: Session Isolation Between Users")
print("="*60)

# User 1 - Support agent asking about cancellations
print("\n--- User 1 (Support Agent) ---")
user1_query = "How do I process a refund?"
user1_result = retriever.query(
    query=user1_query,
    entitlement="agent_support",
    org_id="org_123",
    tags=["cancellation"]
)
print(f"Query: {user1_query}")
print(f"Answer: {user1_result['answer'][:150]}...")
print(f"Sources: {[s['title'] for s in user1_result['sources']]}")

# User 2 - Sales agent asking about bookings
print("\n--- User 2 (Sales Agent) ---")
user2_query = "How do I create a new booking?"
user2_result = retriever.query(
    query=user2_query,
    entitlement="agent_sales",
    org_id="org_123",
    tags=["booking"]
)
print(f"Query: {user2_query}")
print(f"Answer: {user2_result['answer'][:150]}...")
print(f"Sources: {[s['title'] for s in user2_result['sources']]}")

print("\n✓ Session isolation test completed")
print("Manual check: Did each user get appropriate documents for their role?")


# ============================================
# TEST 3: Entitlement Filtering
# ============================================
print("\n" + "="*60)
print("TEST 3: Entitlement-Based Access Control")
print("="*60)

# Try accessing the same topic with different entitlements
query = "What are the booking procedures?"

# Support agent
support_result = retriever.query(
    query=query,
    entitlement="agent_support",
    org_id="org_123"
)

# Sales agent
sales_result = retriever.query(
    query=query,
    entitlement="agent_sales",
    org_id="org_123"
)

print(f"\nQuery: {query}")
print(f"\nSupport Agent sees {len(support_result['sources'])} sources:")
for s in support_result['sources']:
    print(f"  - {s['title']}")

print(f"\nSales Agent sees {len(sales_result['sources'])} sources:")
for s in sales_result['sources']:
    print(f"  - {s['title']}")

print("\n✓ Entitlement filtering test completed")
print("Manual check: Do agents only see documents they're authorized for?")


# ============================================
# TEST 4: Retrieval Quality
# ============================================
print("\n" + "="*60)
print("TEST 4: Retrieval Quality & Relevance")
print("="*60)

test_cases = [
    {
        "query": "cancellation policy",
        "expected_docs": ["Cancellation Procedure"],
        "entitlement": "agent_support"
    },
    {
        "query": "how to book",
        "expected_docs": ["Booking Procedure"],
        "entitlement": "agent_sales"
    },
    {
        "query": "modify reservation",
        "expected_docs": ["Update Procedure"],
        "entitlement": "agent_support"
    }
]

for i, test in enumerate(test_cases, 1):
    print(f"\n--- Test Case {i} ---")
    print(f"Query: {test['query']}")
    print(f"Expected: {test['expected_docs']}")
    
    result = retriever.query(
        query=test['query'],
        entitlement=test['entitlement'],
        org_id="org_123",
        top_k=3
    )
    
    retrieved_titles = [s['title'] for s in result['sources']]
    print(f"Retrieved: {retrieved_titles}")
    
    # Check if expected docs are in top results
    found = any(exp in retrieved_titles for exp in test['expected_docs'])
    status = "✓ PASS" if found else "✗ FAIL"
    print(f"Status: {status}")

print("\n" + "="*60)
print("ALL TESTS COMPLETED")
print("="*60)
print("\nNext steps:")
print("1. Review answers for quality and relevance")
print("2. Check that context is preserved in conversations")
print("3. Verify entitlement filtering works correctly")
print("4. Test with your actual procedure documents")
