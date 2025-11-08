# Quick Start: Using Trained Centroids with Semantic Search

This guide shows you how to train PQ codebooks and use them with the semantic search bar.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Generate Training Data

```bash
python prepare_training_data.py
```

This creates `training_data.json` with example embeddings for semantic commands.

## Step 3: Train PQ Codebooks

```bash
python train_pq_codebook.py --input-file training_data.json --output-dir ./models
```

This will:
- Train PQ codebooks on your training data
- Create semantic centroids for each command
- Save everything to the `models/` directory

## Step 4: Export Centroids to JSON

```bash
python export_centroids_to_json.py --input models/semantic_centroids.pkl --output centroids.json
```

This converts the Python pickle file to JSON that JavaScript can load.

## Step 5: Use in Semantic Search Bar

1. Make sure `centroids.json` is in the same directory as `semantic-search-example.html`
2. Open `semantic-search-example.html` in a web browser
3. The page will automatically load the trained centroids

## What You'll See

- **Status**: Shows "Ready (Pre-trained)" when centroids are loaded
- **Better Accuracy**: Pre-trained centroids provide more accurate semantic matching
- **Online Learning**: Centroids can still be fine-tuned based on user interactions
- **Smooth Convergence**: KeyStoneH-powered smoothing eliminates jitter during learning
- **Animated UI**: Progress bars and confidence indicators update smoothly at 60fps

## Troubleshooting

### Centroids not loading?

- Check browser console for errors
- Ensure `centroids.json` is in the correct location
- Verify the JSON file is valid (can open in a text editor)
- Check browser CORS settings if loading from a different origin

### Want to use your own embeddings?

1. Create a JSON file with your embeddings:
```json
{
  "command_1": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
  "command_2": [[0.5, 0.6, ...], [0.7, 0.8, ...], ...]
}
```

2. Train with your data:
```bash
python train_pq_codebook.py --input-file your_embeddings.json
```

3. Export and use as above

## Advanced Usage

### Programmatic Loading

```javascript
// Load centroids programmatically
const centroids = await SemanticSearchBar.loadCentroidsFromJSON('centroids.json');

const searchBar = new SemanticSearchBar(inputElement, {
  preTrainedCentroids: centroids,
  onCommandDetected: (command, confidence, query) => {
    console.log(`Detected: ${command} (${confidence})`);
  }
});
```

### Custom Commands

The system automatically detects command IDs from the loaded centroids. If you train centroids for custom commands, they'll be available immediately.

### Smoothing Controls (KeyStoneH Integration)

The semantic search bar includes advanced smoothing controls powered by KeyStoneH convergence mechanics:

```javascript
const searchBar = new SemanticSearchBar(inputElement, {
  // GEMA smoothing parameters
  momentumAlpha: 0.85,           // Base smoothing factor (0.7 - 0.95)
  momentumAlphaMin: 0.7,         // Minimum alpha for UI control
  momentumAlphaMax: 0.95,        // Maximum alpha for UI control
  gateMomentumCoupling: true,    // Enable adaptive gating
  
  // Learning parameters
  learningRate: 0.08,            // Base learning rate
  
  // Advanced: see KEYSTONEH_SMOOTHING.md for details
  showAdvancedControls: true     // Show UI controls for tuning
});
```

**Key Features:**
- **Adaptive Smoothing**: Automatically increases as convergence improves
- **Cubic Easing**: Ultra-smooth transitions using cubic-bezier curves
- **Progress-Based Alpha**: Momentum factor adapts based on convergence stage
- **RequestAnimationFrame**: Jitter-free 60fps UI updates

For detailed information on smoothing mechanics, see [`KEYSTONEH_SMOOTHING.md`](KEYSTONEH_SMOOTHING.md).

## Next Steps

- Fine-tune PQ parameters (`--num-subvectors`, `--num-codewords`)
- Add more training data for better accuracy
- Integrate with your own embedding model
- Experiment with different command sets

