# Semantic Search Bar - GEMA Powered (Trained Codebook Version)

A smart semantic search module that plugs into any input field, powered by Gated Exponential Moving Average (GEMA) for smooth momentum and convergence.

## Features

- **GEMA Smoothing**: Gated Exponential Moving Average for smooth momentum updates that prevent jitter
- **State Initialization**: Proper initialization of all momentum, gate, and embedding states
- **Advanced Controls**: UI controls for momentum alpha range (min/max) and gated momentum coupling checkbox
- **Catastrophic Forgetting Prevention**: Contrastive learning to prevent drift
- **Online Learning**: Centroids adapt based on user interactions
- **Plug & Play**: Works with any input field - just pass the element
- **Real Embeddings (Production)**: Uses Transformers.js (all-MiniLM-L6-v2) with your trained `centroids.json`

## Quick Start

```javascript
// 1) Include tempfile.js in your page
// 2) Serve over HTTP (not file://) so centroids.json can load
// 3) Use a module script to init embeddings and load centroids

// Example:
import {} from './tempfile.js';

await SemanticSearchBar.initializeEmbeddings('Xenova/all-MiniLM-L6-v2');
const centroidsData = await fetch('./centroids.json', { cache: 'no-store' }).then(r => r.json());
const centroids = centroidsData.centroids ?? centroidsData;

const searchBar = new SemanticSearchBar(document.getElementById('myInput'), {
  embeddingSize: 384,               // all-MiniLM-L6-v2 output size
  preTrainedCentroids: centroids,   // your trained codebook
  onCommandDetected: (command, confidence, query) => {
    console.log(`Command: ${command} (${(confidence * 100).toFixed(1)}%)`, query);
  }
});

searchBar.enableLearning();         // optional: let it adapt further
```

## Configuration Options

```javascript
{
  // GEMA Configuration
  momentumAlpha: 0.85,              // Range: 0.7 - 0.95
  momentumAlphaMin: 0.7,            // Minimum momentum alpha
  momentumAlphaMax: 0.95,           // Maximum momentum alpha
  gateMomentumCoupling: true,       // Enable gated momentum coupling
  momentumDecayOnConvergence: 0.2,  // Range: 0.0 - 0.5
  
  // Embedding Configuration
  embeddingSize: 384,               // Trained with all-MiniLM-L6-v2
  gateSize: 128,
  learningRate: 0.08,
  beta: 0.9,                        // For catastrophic forgetting prevention
  
  // Commands/Intents
  commands: [
    { id: 'search', label: 'Search', centroid: null },
    { id: 'navigate', label: 'Navigate', centroid: null },
    // ... more commands
  ],
  
  // Callbacks
  onCommandDetected: (command, confidence, query) => {},
  onConfidenceChange: (confidences) => {},
  
  // UI
  showAdvancedControls: true
}
```

## GEMA Algorithm

The module uses Gated Exponential Moving Average (GEMA) to smooth momentum updates:

```javascript
// Smooth momentum update with gating
if (gateMomentumCoupling) {
  const gateWeight = gateValues[i];
  const adaptiveAlpha = alpha * (0.5 + gateWeight);
  momentum[i] = adaptiveAlpha * momentum[i] + (1 - adaptiveAlpha) * delta;
} else {
  momentum[i] = alpha * momentum[i] + (1 - alpha) * delta;
}
```

## Advanced Controls

The module includes UI controls for:
- **Momentum Alpha Range**: Adjust min/max values for momentum smoothing
- **Gated Momentum Coupling**: Toggle to enable/disable gate-momentum coupling
- Real-time updates to convergence behavior

## API Methods

- `enableLearning()` - Enable online learning mode
- `disableLearning()` - Disable online learning mode
- `reset()` - Reset all state to initial values
- `getState()` - Get current state (momentum, gates, centroids, etc.)
- `findBestCommand(query)` - Find best matching command for a query
- `handleQuery(query, execute)` - Process a query and optionally execute

## Example

See `semantic-search-example.html` for a complete working example.

Notes:
- Serve with a local server (e.g. `python -m http.server 8000`) so `centroids.json` can be fetched.
- `centroids.json` can be either the raw map `{ "search": [...], ... }` or `{ centroids: { ... } }`.
- The embedding model is loaded once and cached by the browser.

## Technical Details

### Momentum Smoothing
- Uses GEMA to prevent jitter in centroid updates
- Adaptive alpha based on gate values when coupling is enabled
- Momentum decay on convergence to prevent drift

### Catastrophic Forgetting Prevention
- Contrastive learning updates positive and negative embeddings
- Beta parameter controls update strength (default: 0.9)
- Normalization prevents embedding drift

### State Initialization
- All momentum vectors initialized to zero
- Gate values initialized to 0.5 (neutral)
- Centroids initialized with small random values
- Convergence state tracked per command

## License

MIT

