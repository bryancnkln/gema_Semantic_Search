# Semantic Search Bar - Learning & Convergence

## Overview

The semantic search bar is designed to **learn and converge over time** through user interactions. This is the core value proposition - a search bar that gets smarter with each query.

## Key Features

### 1. **Continuous Learning**
- **Every query updates centroids** - not just high-confidence matches
- Adaptive learning rate based on confidence (higher confidence = faster learning)
- Centroids move toward user queries, improving matching over time

### 2. **Semantic Matching**
- Embeddings use the **same algorithm** as centroid creation for proper alignment
- Keyword-based semantic groups ensure logical matching
- Cosine similarity for accurate semantic distance measurement

### 3. **Convergence Tracking**
- Each command tracks:
  - Similarity scores over time
  - Number of iterations
  - Convergence status (converged when similarity > 0.85 after 5+ iterations)

### 4. **GEMA Smoothing**
- Gated Exponential Moving Average prevents jitter
- Smooth momentum updates for stable learning
- Prevents catastrophic forgetting through contrastive learning

## How Learning Works

### Initial State
- Centroids start with semantic keyword-based positions
- Each command has a centroid trained on relevant keywords

### During Queries
1. **Query → Embedding**: Text is converted to embedding using semantic keyword matching
2. **Match Selection**: Best matching centroid is found using cosine similarity
3. **Learning Update**: 
   - Matched centroid moves toward query embedding
   - Learning rate adapts: `lr * max(0.3, confidence)`
   - Contrastive learning pushes away from incorrect centroids

### Convergence
- After multiple queries, centroids converge to optimal positions
- Similarity scores increase over time
- System becomes more accurate with usage

## Example Learning Progression

**Query 1**: "search for files"
- Initial match: Search (45% confidence)
- Centroid updates toward query
- Learning rate: 0.08 * 0.45 = 0.036

**Query 2**: "find documents"  
- Match: Search (52% confidence) ← Improved!
- Centroid continues converging
- Learning rate: 0.08 * 0.52 = 0.042

**Query 5**: "search database"
- Match: Search (78% confidence) ← Much better!
- Convergence approaching

**Query 10+**: "look for data"
- Match: Search (85%+ confidence) ← Converged!
- System has learned the semantic pattern

## Technical Details

### Embedding Algorithm
- Uses same hash-based feature generation as Python centroids
- Three seed functions (seed1, seed2, seed3) for rich features
- Keyword position weighting (first keywords more important)
- L2 normalization for consistent similarity calculations

### Learning Algorithm
- **GEMA momentum**: Smooth updates prevent jitter
- **Adaptive learning rate**: Scales with confidence
- **Contrastive learning**: Improves separation between commands
- **Quantization**: 4-bit quantization for efficiency

### Convergence Criteria
- Similarity > 0.85
- Minimum 5 iterations
- Tracks convergence state per command

## Usage

The search bar learns automatically when `isLearning` is enabled (default: true).

```javascript
const searchBar = new SemanticSearchBar(input, {
  // Learning is enabled by default
  // searchBar.enableLearning() // Explicitly enable
  // searchBar.disableLearning() // Disable if needed
});
```

## Monitoring Convergence

Check convergence state:
```javascript
const state = searchBar.getState();
console.log(state.convergenceState);
// Shows: { command: { converged: true/false, similarity: 0.85, iterations: 10 } }
```

## Best Practices

1. **Start with demo centroids** - Provides good initial semantic positions
2. **Let it learn** - Every query improves the system
3. **Monitor confidence** - Watch confidence scores increase over time
4. **Use consistent queries** - Similar queries help convergence

## Why This Matters

Unlike static search systems, this semantic search bar:
- ✅ **Adapts to your language** - Learns your query patterns
- ✅ **Improves over time** - Gets better with each use
- ✅ **Converges to optimal** - Finds best semantic representations
- ✅ **No manual tuning** - Self-optimizing through usage

This is the future of search interfaces - **intelligent, adaptive, and continuously improving**.

