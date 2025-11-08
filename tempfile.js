/**
 * SemanticSearchBar - A GEMA-powered semantic search module
 * Plugs into any input field for intelligent command/query understanding
 * Uses Gated Exponential Moving Average (GEMA) for smooth momentum and convergence
 */

class SemanticSearchBar {
  constructor(inputElement, options = {}) {
    // === STATE INITIALIZATION ===
    this.inputElement = inputElement;
    this.options = {
      // GEMA Configuration
      momentumAlpha: options.momentumAlpha ?? 0.85,        // Range: 0.7 - 0.95
      momentumAlphaMin: options.momentumAlphaMin ?? 0.7,
      momentumAlphaMax: options.momentumAlphaMax ?? 0.95,
      gateMomentumCoupling: options.gateMomentumCoupling ?? true,
      momentumDecayOnConvergence: options.momentumDecayOnConvergence ?? 0.2, // Range: 0.0 - 0.5
      
      // Embedding Configuration
      embeddingSize: options.embeddingSize ?? 384,  // Updated for all-MiniLM-L6-v2
      gateSize: options.gateSize ?? 128,
      learningRate: options.learningRate ?? 0.08,
      beta: options.beta ?? 0.9,  // For catastrophic forgetting prevention
      
      // Command/Intent Centroids
      commands: options.commands ?? [
        { id: 'search', label: 'Search', centroid: null },
        { id: 'navigate', label: 'Navigate', centroid: null },
        { id: 'execute', label: 'Execute', centroid: null },
        { id: 'help', label: 'Help', centroid: null }
      ],
      
      // Callbacks
      onCommandDetected: options.onCommandDetected ?? null,
      onConfidenceChange: options.onConfidenceChange ?? null,
      
      // UI
      showAdvancedControls: options.showAdvancedControls ?? true
    };
    
    // === GEMA STATE INITIALIZATION ===
    this.momentum = new Float32Array(this.options.embeddingSize);
    this.gateValues = new Float32Array(this.options.gateSize);
    this.centroids = new Map();
    this.posEmbed = new Float32Array(this.options.embeddingSize);
    this.negEmbed = new Float32Array(this.options.embeddingSize);
    this.lastEmbedding = new Float32Array(this.options.embeddingSize);
    this.contrastiveBuffer = [];
    this.convergenceState = new Map();
    
    // Initialize centroids for each command
    // If preTrainedCentroids provided, use them; otherwise initialize randomly
    if (options.preTrainedCentroids) {
      this.loadCentroids(options.preTrainedCentroids);
    } else {
      this.options.commands.forEach(cmd => {
        this.centroids.set(cmd.id, this.randomVector());
        this.convergenceState.set(cmd.id, {
          converged: false,
          similarity: 0,
          iterations: 0
        });
      });
    }
    
    // Initialize gate values
    for (let i = 0; i < this.options.gateSize; i++) {
      this.gateValues[i] = 0.5;
    }
    
    // Initialize momentum
    this.momentum.fill(0);
    
    // === UI SETUP ===
    this.setupUI();
    this.setupEventListeners();
    
    // === INTERNAL STATE ===
    this.iteration = 0;
    this.lastQuery = '';
    this.isLearning = false;
    this._renderFrameId = null; // For smooth requestAnimationFrame rendering
    
    // Set embedding extractor (use global if available, or custom from options)
    this._embeddingExtractor = options.embeddingExtractor || 
                                SemanticSearchBar._globalEmbeddingExtractor || 
                                null;
  }
  
  // === STATE INITIALIZATION HELPERS ===
  randomVector() {
    const vec = new Float32Array(this.options.embeddingSize);
    for (let i = 0; i < this.options.embeddingSize; i++) {
      vec[i] = (Math.random() * 2 - 1) * 0.1; // Small initial values
    }
    this.normalize(vec);
    return vec;
  }
  
  normalize(vec) {
    let sum = 0;
    for (let i = 0; i < vec.length; i++) {
      sum += vec[i] * vec[i];
    }
    const norm = Math.sqrt(sum);
    if (norm > 0) {
      for (let i = 0; i < vec.length; i++) {
        vec[i] /= norm;
      }
    }
  }
  
  /**
   * Blend two vectors smoothly (from KeyStoneH)
   * @param {Float32Array} a - First vector
   * @param {Float32Array} b - Second vector  
   * @param {number} t - Blend factor (0 = all a, 1 = all b)
   * @returns {Float32Array} - Blended and normalized vector
   */
  blendVectors(a, b, t) {
    const vec = new Float32Array(this.options.embeddingSize);
    // Apply easing to blend factor for smoother transitions
    const easedT = this.easeInOutCubic(t * 2 - 1) * 0.5 + 0.5;
    
    for (let i = 0; i < this.options.embeddingSize; i++) {
      vec[i] = a[i] * (1 - easedT) + b[i] * easedT;
    }
    this.normalize(vec);
    return vec;
  }
  
  // === GEMA SMOOTHED MOMENTUM (ENHANCED WITH KEYSTONEH CONVERGENCE) ===
  /**
   * Gated Exponential Moving Average (GEMA) for smooth momentum
   * Prevents jitter and catastrophic forgetting
   * Enhanced with KeyStoneH convergence mechanics for ultra-smooth updates
   */
  updateMomentumWithGEMA(delta, i) {
    const alpha = this.options.momentumAlpha;
    
    // Calculate progress-based adaptive alpha (KeyStoneH-style)
    // As we converge, increase smoothing to reduce jitter
    const convergenceProgress = this.getConvergenceProgress();
    const progressBasedAlpha = alpha + (0.95 - alpha) * convergenceProgress;
    
    // GEMA: Smooth momentum update with gating
    if (this.options.gateMomentumCoupling) {
      // Couple momentum with gate values for adaptive smoothing
      const gateWeight = i < this.options.gateSize ? this.gateValues[i] : 0.5;
      const adaptiveAlpha = progressBasedAlpha * (0.5 + gateWeight * 0.5);
      
      // Smooth momentum update with cubic easing for ultra-smooth transitions
      const easedDelta = this.easeInOutCubic(delta);
      this.momentum[i] = adaptiveAlpha * this.momentum[i] + (1 - adaptiveAlpha) * easedDelta;
    } else {
      // Standard EMA momentum with progress-based smoothing
      this.momentum[i] = progressBasedAlpha * this.momentum[i] + (1 - progressBasedAlpha) * delta;
    }
    
    // Apply momentum decay on convergence to prevent drift
    const convergenceDecay = this.getConvergenceDecay();
    if (convergenceDecay > 0) {
      this.momentum[i] *= (1 - convergenceDecay);
    }
    
    return this.momentum[i];
  }
  
  /**
   * Cubic easing for ultra-smooth transitions (from KeyStoneH)
   * For delta values (can be negative), applies easing while preserving sign
   */
  easeInOutCubic(x) {
    // For small delta values, preserve the sign and apply easing to magnitude
    if (Math.abs(x) < 0.001) return x;
    
    const absX = Math.abs(x);
    const sign = x < 0 ? -1 : 1;
    
    // Standard cubic easing: slow at start/end, fast in middle
    // Map to [0,1] range, apply easing, map back
    const normalized = Math.min(1, absX);
    const eased = normalized < 0.5 
      ? 4 * normalized * normalized * normalized 
      : 1 - Math.pow(-2 * normalized + 2, 3) / 2;
    
    return sign * eased * absX;
  }
  
  /**
   * Get overall convergence progress (0 = not converged, 1 = fully converged)
   */
  getConvergenceProgress() {
    let totalSimilarity = 0;
    let count = 0;
    
    this.convergenceState.forEach(state => {
      if (state.iterations > 0) {
        totalSimilarity += state.similarity;
        count++;
      }
    });
    
    if (count === 0) return 0;
    
    const avgSimilarity = totalSimilarity / count;
    // Map [0.5, 0.95] to [0, 1] for smooth progress
    return Math.max(0, Math.min(1, (avgSimilarity - 0.5) / 0.45));
  }
  
  getConvergenceDecay() {
    // Calculate decay based on convergence state
    let maxConvergence = 0;
    this.convergenceState.forEach(state => {
      if (state.converged) {
        maxConvergence = Math.max(maxConvergence, state.similarity);
      }
    });
    
    // Apply decay when highly converged
    if (maxConvergence > 0.95) {
      return this.options.momentumDecayOnConvergence;
    }
    return 0;
  }
  
  // === QUANTIZATION ===
  quantize4bit(value) {
    // Quantize to 4-bit range [-8, +7]
    const quantized = Math.round(value * 7);
    return Math.max(-8, Math.min(7, quantized)) / 7.0;
  }
  
  // === CENTROID UPDATE WITH GEMA (ENHANCED WITH KEYSTONEH CONVERGENCE) ===
  updateCentroid(commandId, feedbackEmbedding, learningRate = null) {
    const centroid = this.centroids.get(commandId);
    if (!centroid) return;
    
    const state = this.convergenceState.get(commandId);
    state.iterations++;
    
    // Use provided learning rate or default
    const lr = learningRate !== null ? learningRate : this.options.learningRate;
    
    // Calculate current similarity for adaptive convergence
    const currentSimilarity = this.cosineSimilarity(feedbackEmbedding, centroid);
    
    // Adaptive weight based on convergence (KeyStoneH-style)
    // Higher weight = faster convergence, reduces as we get closer
    const convergenceWeight = Math.max(0.2, 1.0 - currentSimilarity);
    const adaptiveLR = lr * convergenceWeight;
    
    // Option 1: GEMA with momentum (smooth but gradual)
    for (let i = 0; i < this.options.embeddingSize; i++) {
      const delta = feedbackEmbedding[i] - centroid[i];
      
      // Update momentum with GEMA smoothing
      const smoothedMomentum = this.updateMomentumWithGEMA(delta, i);
      
      // Apply gating
      const gateWeight = i < this.options.gateSize ? this.gateValues[i] : 1.0;
      const effectiveDelta = smoothedMomentum * gateWeight;
      
      // Update centroid with quantized, smoothed delta
      const quantizedDelta = this.quantize4bit(effectiveDelta * adaptiveLR);
      centroid[i] += quantizedDelta;
    }
    
    // Normalize to prevent drift
    this.normalize(centroid);
    
    // Update convergence state with exponential smoothing
    const smoothedSimilarity = 0.7 * state.similarity + 0.3 * currentSimilarity;
    state.similarity = smoothedSimilarity;
    
    // Convergence threshold with iteration requirement
    const convergenceThreshold = 0.85;
    const minIterations = 5;
    state.converged = smoothedSimilarity > convergenceThreshold && state.iterations > minIterations;
    
    // Log convergence progress (useful for debugging)
    if (state.iterations % 10 === 0) {
      console.log(`[${commandId}] Convergence: ${(smoothedSimilarity * 100).toFixed(1)}% (iter: ${state.iterations})`);
    }
  }
  
  // === PREVENT CATASTROPHIC FORGETTING ===
  /**
   * Update embeddings with contrastive learning to prevent drift
   */
  updateEmbeddingsWithContrastive(positiveGrad, negativeGrad) {
    for (let i = 0; i < this.options.embeddingSize; i++) {
      // Positive embedding update
      this.posEmbed[i] = (1 - this.options.beta) * this.posEmbed[i] + 
                        this.options.beta * (this.posEmbed[i] + positiveGrad[i]);
      
      // Negative embedding update
      this.negEmbed[i] = (1 - this.options.beta) * this.negEmbed[i] + 
                         this.options.beta * (this.negEmbed[i] + negativeGrad[i]);
    }
    
    // Normalize to prevent drift
    this.normalize(this.posEmbed);
    this.normalize(this.negEmbed);
  }
  
  // === CONTRASTIVE GRADIENT ===
  computeContrastiveGradient(positive, target, negative) {
    const grad = new Float32Array(this.options.embeddingSize);
    for (let i = 0; i < this.options.embeddingSize; i++) {
      grad[i] += (target[i] - positive[i]) * 0.5;
      grad[i] -= (negative[i] - positive[i]) * 0.3;
    }
    this.normalize(grad);
    return grad;
  }
  
  // === GATE UPDATE (ENHANCED WITH KEYSTONEH SMOOTHING) ===
  updateGate(intentEmbedding) {
    const scores = Array.from(this.centroids.entries()).map(([id, centroid]) => 
      this.cosineSimilarity(intentEmbedding, centroid)
    );
    
    // Adaptive exponential decay based on convergence (KeyStoneH-style)
    const convergenceProgress = this.getConvergenceProgress();
    const halfLife = 2000 * (1 + convergenceProgress * 2); // Slower decay as we converge
    const alpha = Math.pow(0.5, this.iteration / halfLife);
    
    // Smooth gate updates with easing
    for (let i = 0; i < this.options.gateSize && i < scores.length; i++) {
      // Calculate target gate value with exponential smoothing
      const targetGate = this.gateValues[i] * alpha + scores[i] * (1 - alpha);
      
      // Apply additional smoothing for jitter-free updates
      const smoothingFactor = 0.3 + convergenceProgress * 0.4; // More smoothing as we converge
      this.gateValues[i] = this.gateValues[i] * smoothingFactor + targetGate * (1 - smoothingFactor);
      
      // Clamp gate values with soft boundaries
      this.gateValues[i] = Math.max(0.1, Math.min(0.9, this.gateValues[i]));
    }
  }
  
  // === SIMILARITY ===
  cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    const denom = Math.sqrt(normA * normB);
    return denom === 0 ? 0 : dot / denom;
  }
  
  // === SEMANTIC SEARCH ===
  /**
   * Convert text query to embedding using Transformers.js
   * 
   * PRODUCTION-READY: Uses the same model as training (all-MiniLM-L6-v2)
   * This is automatically initialized when the page loads.
   * 
   * If you need a custom embedding function, override this method:
   * searchBar.textToEmbedding = async (text) => { ... }
   */
  async textToEmbedding(text) {
    // Check if embedding extractor is available
    if (!this._embeddingExtractor) {
      throw new Error(
        'Embedding model not loaded! ' +
        'Make sure to call SemanticSearchBar.initializeEmbeddings() ' +
        'or provide a custom textToEmbedding function.'
      );
    }
    
    try {
      // Generate embedding using Transformers.js
      const output = await this._embeddingExtractor(text, {
        pooling: 'mean',
        normalize: true
      });
      
      return new Float32Array(output.data);
    } catch (error) {
      console.error('Error generating embedding:', error);
      throw error;
    }
  }
  
  /**
   * Set a custom embedding extractor (from Transformers.js or other source)
   * @param {Function} extractor - Function that takes text and returns embeddings
   */
  setEmbeddingExtractor(extractor) {
    this._embeddingExtractor = extractor;
  }
  
  /**
   * Static method to initialize Transformers.js embeddings globally
   * Call this once before creating any SemanticSearchBar instances
   * 
   * @param {string} modelName - HuggingFace model name (default: all-MiniLM-L6-v2)
   * @returns {Promise<Function>} The embedding extractor function
   */
  static async initializeEmbeddings(modelName = 'Xenova/all-MiniLM-L6-v2') {
    if (typeof window === 'undefined') {
      throw new Error('Transformers.js requires a browser environment');
    }
    
    console.log(`Loading embedding model: ${modelName}...`);
    console.log('First load: ~50MB download, then cached for future use.');
    
    try {
      // Dynamically import Transformers.js
      const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0');
      
      // Create feature extraction pipeline
      const extractor = await pipeline('feature-extraction', modelName);
      
      console.log(`✅ Embedding model loaded: ${modelName}`);
      
      // Store globally for all instances
      SemanticSearchBar._globalEmbeddingExtractor = extractor;
      
      return extractor;
    } catch (error) {
      console.error('Failed to load embedding model:', error);
      throw error;
    }
  }
  
  /**
   * Hash a string to a number
   */
  hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash) + str.charCodeAt(i);
      hash = hash & hash;
    }
    return hash;
  }
  
  /**
   * Find best matching command for query
   */
  async findBestCommand(query) {
    const queryEmbedding = await this.textToEmbedding(query);
    this.lastEmbedding.set(queryEmbedding);
    
    // Calculate all similarities first
    const centroidIds = Array.from(this.centroids.keys());
    
    if (centroidIds.length === 0) {
      console.warn('No centroids available for matching');
      return {
        command: null,
        confidence: 0,
        allConfidences: []
      };
    }
    
    const allSimilarities = centroidIds.map(id => {
      const centroid = this.centroids.get(id);
      const rawSimilarity = this.cosineSimilarity(queryEmbedding, centroid);
      
      // Apply gating if enabled (for selection only, not for display)
      let gatedSimilarity = rawSimilarity;
      if (this.options.gateMomentumCoupling) {
        const gateIndex = centroidIds.indexOf(id);
        if (gateIndex >= 0 && gateIndex < this.options.gateSize) {
          gatedSimilarity = rawSimilarity * this.gateValues[gateIndex];
        }
      }
      
      return {
        id,
        rawSimilarity,
        gatedSimilarity
      };
    });
    
    // Find best matching command
    // Use raw similarity for selection to ensure semantic accuracy
    // Gating is primarily for momentum updates, not for selection
    let bestCommand = null;
    let bestSimilarity = -Infinity;
    let bestIndex = -1;
    
    for (let i = 0; i < allSimilarities.length; i++) {
      const sim = allSimilarities[i];
      // Use raw similarity for selection (semantic match quality)
      // Only use gated similarity if it's significantly different and gating is explicitly for selection
      const selectionSimilarity = sim.rawSimilarity;
      
      if (selectionSimilarity > bestSimilarity) {
        bestSimilarity = selectionSimilarity;
        bestCommand = sim.id;
        bestIndex = i;
      }
    }
    
    // Fallback: if no best command found (shouldn't happen), use first centroid
    if (!bestCommand || bestIndex < 0) {
      if (centroidIds.length > 0) {
        bestCommand = centroidIds[0];
        bestSimilarity = allSimilarities[0].rawSimilarity;
        bestIndex = 0;
      } else {
        return {
          command: null,
          confidence: 0,
          allConfidences: []
        };
      }
    }
    
    // Update gate (this happens after selection to avoid affecting current match)
    this.updateGate(queryEmbedding);
    
    // Online learning: ALWAYS learn from user interactions to improve over time
    // This is the key feature - the search bar learns and converges with each query
    if (bestCommand && this.isLearning) {
      // Learn from every interaction, not just high confidence ones
      // Lower learning rate for low confidence matches to avoid overfitting
      const adaptiveLearningRate = this.options.learningRate * Math.max(0.3, bestSimilarity);
      
      // Update the matched centroid to move toward the query
      this.updateCentroid(bestCommand, queryEmbedding, adaptiveLearningRate);
      
      // Contrastive learning: push away from other centroids to improve separation
      const otherCentroids = Array.from(this.centroids.entries())
        .filter(([id]) => id !== bestCommand)
        .map(([, centroid]) => centroid);
      
      if (otherCentroids.length > 0) {
        // Use the most similar incorrect centroid as negative sample
        const negativeCentroid = otherCentroids.reduce((best, current) => {
          const simCurrent = this.cosineSimilarity(queryEmbedding, current);
          const simBest = this.cosineSimilarity(queryEmbedding, best);
          return simCurrent > simBest ? current : best;
        });
        
        const grad = this.computeContrastiveGradient(
          queryEmbedding,
          this.centroids.get(bestCommand),
          negativeCentroid
        );
        this.updateEmbeddingsWithContrastive(grad, negativeCentroid);
      }
      
      // Update convergence state
      const state = this.convergenceState.get(bestCommand);
      if (state) {
        state.similarity = bestSimilarity;
        state.iterations++;
        // Mark as converged if similarity is consistently high
        state.converged = bestSimilarity > 0.85 && state.iterations > 5;
      }
    }
    
    this.iteration++;
    
    // Return with raw similarities for display (so users see actual semantic match quality)
    // Selection is based on raw similarity to ensure semantic accuracy
    return {
      command: bestCommand,
      confidence: bestSimilarity,
      allConfidences: allSimilarities.map(sim => ({
        id: sim.id,
        confidence: sim.rawSimilarity  // Show raw similarity for transparency
      }))
    };
  }
  
  // === UI SETUP ===
  setupUI() {
    // Create container for search bar and suggestions
    const container = document.createElement('div');
    container.className = 'semantic-search-container';
    container.style.cssText = `
      position: relative;
      width: 100%;
      max-width: 600px;
      margin: 0 auto;
    `;
    
    // Wrap input if needed
    if (this.inputElement.parentNode) {
      this.inputElement.parentNode.insertBefore(container, this.inputElement);
      container.appendChild(this.inputElement);
    }
    
    // Add styles
    this.inputElement.style.cssText += `
      width: 100%;
      padding: 12px 16px;
      font-size: 16px;
      border: 2px solid #3a3a4a;
      border-radius: 8px;
      background: rgba(25, 28, 38, 0.95);
      color: #d4d4d4;
      transition: all 0.3s ease;
    `;
    
    // Suggestions dropdown
    this.suggestionsDiv = document.createElement('div');
    this.suggestionsDiv.className = 'semantic-suggestions';
    this.suggestionsDiv.style.cssText = `
      position: absolute;
      top: 100%;
      left: 0;
      right: 0;
      background: rgba(25, 28, 38, 0.98);
      border: 2px solid #3a3a4a;
      border-top: none;
      border-radius: 0 0 8px 8px;
      max-height: 300px;
      overflow-y: auto;
      display: none;
      z-index: 1000;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    `;
    container.appendChild(this.suggestionsDiv);
    
    // Advanced controls
    if (this.options.showAdvancedControls) {
      this.setupAdvancedControls(container);
    }
  }
  
  setupAdvancedControls(container) {
    const advancedDiv = document.createElement('div');
    advancedDiv.className = 'semantic-advanced-controls';
    advancedDiv.style.cssText = `
      margin-top: 1rem;
      padding: 1rem;
      background: rgba(30, 30, 40, 0.7);
      border: 1px solid #3a3a4a;
      border-radius: 8px;
      display: none;
    `;
    
    // Momentum Alpha Range
    const momentumDiv = document.createElement('div');
    momentumDiv.style.cssText = 'margin-bottom: 1rem;';
    
    const momentumLabel = document.createElement('label');
    momentumLabel.textContent = 'Momentum Alpha: ';
    momentumLabel.style.cssText = 'color: #8894a4; display: block; margin-bottom: 0.5rem;';
    
    const momentumMinInput = document.createElement('input');
    momentumMinInput.type = 'number';
    momentumMinInput.min = '0.5';
    momentumMinInput.max = '0.95';
    momentumMinInput.step = '0.01';
    momentumMinInput.value = this.options.momentumAlphaMin;
    momentumMinInput.style.cssText = 'width: 80px; padding: 4px; margin-right: 0.5rem; background: #1e1e2e; color: #d4d4d4; border: 1px solid #3a3a4a; border-radius: 4px;';
    
    const momentumMaxInput = document.createElement('input');
    momentumMaxInput.type = 'number';
    momentumMaxInput.min = '0.7';
    momentumMaxInput.max = '0.99';
    momentumMaxInput.step = '0.01';
    momentumMaxInput.value = this.options.momentumAlphaMax;
    momentumMaxInput.style.cssText = 'width: 80px; padding: 4px; background: #1e1e2e; color: #d4d4d4; border: 1px solid #3a3a4a; border-radius: 4px;';
    
    const momentumSlider = document.createElement('input');
    momentumSlider.type = 'range';
    momentumSlider.min = this.options.momentumAlphaMin;
    momentumSlider.max = this.options.momentumAlphaMax;
    momentumSlider.step = '0.01';
    momentumSlider.value = this.options.momentumAlpha;
    momentumSlider.style.cssText = 'width: 100%; margin-top: 0.5rem;';
    
    const momentumValue = document.createElement('span');
    momentumValue.textContent = this.options.momentumAlpha.toFixed(2);
    momentumValue.style.cssText = 'color: #a0a8c0; margin-left: 0.5rem;';
    
    momentumSlider.addEventListener('input', (e) => {
      const value = parseFloat(e.target.value);
      this.options.momentumAlpha = value;
      momentumValue.textContent = value.toFixed(2);
    });
    
    momentumMinInput.addEventListener('change', (e) => {
      this.options.momentumAlphaMin = parseFloat(e.target.value);
      momentumSlider.min = this.options.momentumAlphaMin;
    });
    
    momentumMaxInput.addEventListener('change', (e) => {
      this.options.momentumAlphaMax = parseFloat(e.target.value);
      momentumSlider.max = this.options.momentumAlphaMax;
    });
    
    momentumDiv.appendChild(momentumLabel);
    momentumDiv.appendChild(momentumMinInput);
    momentumDiv.appendChild(document.createTextNode(' - '));
    momentumDiv.appendChild(momentumMaxInput);
    momentumDiv.appendChild(momentumSlider);
    momentumDiv.appendChild(momentumValue);
    
    // Gated Momentum Checkbox
    const gateCheckboxDiv = document.createElement('div');
    gateCheckboxDiv.style.cssText = 'margin-bottom: 1rem;';
    
    const gateCheckbox = document.createElement('input');
    gateCheckbox.type = 'checkbox';
    gateCheckbox.checked = this.options.gateMomentumCoupling;
    gateCheckbox.id = 'gate-momentum-coupling';
    gateCheckbox.style.cssText = 'margin-right: 0.5rem;';
    
    const gateLabel = document.createElement('label');
    gateLabel.htmlFor = 'gate-momentum-coupling';
    gateLabel.textContent = 'Enable Gated Momentum Coupling';
    gateLabel.style.cssText = 'color: #8894a4; cursor: pointer;';
    
    gateCheckbox.addEventListener('change', (e) => {
      this.options.gateMomentumCoupling = e.target.checked;
    });
    
    gateCheckboxDiv.appendChild(gateCheckbox);
    gateCheckboxDiv.appendChild(gateLabel);
    
    // Toggle button
    const toggleBtn = document.createElement('button');
    toggleBtn.textContent = '⚙️ Advanced Settings';
    toggleBtn.style.cssText = `
      padding: 0.5rem 1rem;
      background: #2a3040;
      color: #8894a4;
      border: 1px solid #3a3a4a;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
      margin-top: 0.5rem;
    `;
    
    toggleBtn.addEventListener('click', () => {
      const isVisible = advancedDiv.style.display !== 'none';
      advancedDiv.style.display = isVisible ? 'none' : 'block';
      toggleBtn.textContent = isVisible ? '⚙️ Advanced Settings' : '▲ Hide Settings';
    });
    
    advancedDiv.appendChild(momentumDiv);
    advancedDiv.appendChild(gateCheckboxDiv);
    container.appendChild(advancedDiv);
    container.appendChild(toggleBtn);
    
    this.advancedControlsDiv = advancedDiv;
  }
  
  // === EVENT LISTENERS ===
  setupEventListeners() {
    let debounceTimer;
    
    this.inputElement.addEventListener('input', (e) => {
      clearTimeout(debounceTimer);
      const query = e.target.value.trim();
      
      if (query.length === 0) {
        this.suggestionsDiv.style.display = 'none';
        return;
      }
      
      debounceTimer = setTimeout(() => {
        this.handleQuery(query);
      }, 150);
    });
    
    this.inputElement.addEventListener('focus', () => {
      if (this.inputElement.value.trim().length > 0) {
        this.suggestionsDiv.style.display = 'block';
      }
    });
    
    this.inputElement.addEventListener('blur', () => {
      // Delay to allow click events on suggestions
      setTimeout(() => {
        this.suggestionsDiv.style.display = 'none';
      }, 200);
    });
    
    this.inputElement.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        const query = this.inputElement.value.trim();
        if (query) {
          this.handleQuery(query, true);
        }
      }
    });
  }
  
  // === QUERY HANDLING (ENHANCED WITH SMOOTH RENDERING) ===
  async handleQuery(query, execute = false) {
    this.lastQuery = query;
    
    try {
      const result = await this.findBestCommand(query);
      
      // Schedule smooth UI update with requestAnimationFrame (KeyStoneH-style)
      if (this._renderFrameId) {
        cancelAnimationFrame(this._renderFrameId);
      }
      
      this._renderFrameId = requestAnimationFrame(() => {
        // Update suggestions with smooth rendering
        this.updateSuggestions(result);
        
        // Callback
        if (this.options.onCommandDetected) {
          this.options.onCommandDetected(result.command, result.confidence, query);
        }
        
        if (this.options.onConfidenceChange) {
          this.options.onConfidenceChange(result.allConfidences);
        }
        
        this._renderFrameId = null;
      });
      
      if (execute) {
        await this.executeCommand(result.command, query);
      }
    } catch (error) {
      console.error('Error handling query:', error);
      // Show error in UI
      this.suggestionsDiv.innerHTML = `
        <div style="padding: 12px 16px; color: #d46f6f;">
          Error: ${error.message}
        </div>
      `;
      this.suggestionsDiv.style.display = 'block';
    }
  }
  
  updateSuggestions(result) {
    // Smooth fade-out before update (if already displayed)
    const wasVisible = this.suggestionsDiv.style.display !== 'none';
    
    // Clear and update content
    this.suggestionsDiv.innerHTML = '';
    this.suggestionsDiv.style.display = 'block';
    
    // Apply smooth opacity transition for jitter-free updates
    if (wasVisible) {
      this.suggestionsDiv.style.transition = 'opacity 0.15s ease-out';
      this.suggestionsDiv.style.opacity = '0';
      
      // Force reflow
      this.suggestionsDiv.offsetHeight;
      
      // Restore opacity after content update
      requestAnimationFrame(() => {
        this.suggestionsDiv.style.opacity = '1';
      });
    }
    
    // Sort by confidence
    const sorted = result.allConfidences.sort((a, b) => b.confidence - a.confidence);
    
    sorted.forEach((item, index) => {
      const cmd = this.options.commands.find(c => c.id === item.id);
      if (!cmd) return;
      
      const suggestionDiv = document.createElement('div');
      suggestionDiv.className = 'semantic-suggestion';
      suggestionDiv.style.cssText = `
        padding: 12px 16px;
        cursor: pointer;
        border-bottom: 1px solid #3a3a4a;
        transition: background 0.2s ease-out, transform 0.2s ease-out;
        ${index === 0 ? 'background: rgba(136, 148, 164, 0.1);' : ''}
        transform: translateX(0);
      `;
      
      const label = document.createElement('div');
      label.textContent = cmd.label;
      label.style.cssText = 'color: #8894a4; font-weight: 500; margin-bottom: 4px;';
      
      const confidence = document.createElement('div');
      confidence.textContent = `${(item.confidence * 100).toFixed(1)}% confidence`;
      confidence.style.cssText = 'color: #70798c; font-size: 12px;';
      
      // Add progress bar for visual feedback (KeyStoneH-style)
      const progressBar = document.createElement('div');
      progressBar.style.cssText = `
        width: 100%;
        height: 2px;
        background: rgba(58, 58, 74, 0.5);
        margin-top: 8px;
        border-radius: 2px;
        overflow: hidden;
      `;
      
      const progressFill = document.createElement('div');
      progressFill.style.cssText = `
        height: 100%;
        background: linear-gradient(90deg, #5a6177, #8894a4);
        width: 0%;
        transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
      `;
      progressBar.appendChild(progressFill);
      
      // Animate progress bar
      requestAnimationFrame(() => {
        progressFill.style.width = `${item.confidence * 100}%`;
      });
      
      suggestionDiv.appendChild(label);
      suggestionDiv.appendChild(confidence);
      suggestionDiv.appendChild(progressBar);
      
      suggestionDiv.addEventListener('mouseenter', () => {
        suggestionDiv.style.background = 'rgba(136, 148, 164, 0.15)';
        suggestionDiv.style.transform = 'translateX(4px)';
      });
      
      suggestionDiv.addEventListener('mouseleave', () => {
        suggestionDiv.style.background = index === 0 ? 'rgba(136, 148, 164, 0.1)' : 'transparent';
        suggestionDiv.style.transform = 'translateX(0)';
      });
      
      suggestionDiv.addEventListener('click', () => {
        this.executeCommand(item.id, this.inputElement.value);
      });
      
      this.suggestionsDiv.appendChild(suggestionDiv);
    });
  }
  
  async executeCommand(commandId, query) {
    const cmd = this.options.commands.find(c => c.id === commandId);
    console.log(`Executing command: ${cmd?.label || commandId}`, { query });
    
    // Enable learning mode for this interaction
    this.isLearning = true;
    
    // Update centroid based on user selection
    const queryEmbedding = await this.textToEmbedding(query);
    this.updateCentroid(commandId, queryEmbedding);
    
    this.suggestionsDiv.style.display = 'none';
  }
  
  // === PUBLIC API ===
  enableLearning() {
    this.isLearning = true;
  }
  
  disableLearning() {
    this.isLearning = false;
  }
  
  /**
   * Load pre-trained centroids from data object
   * @param {Object} centroidsData - Object mapping command_id -> centroid array
   */
  loadCentroids(centroidsData) {
    // Handle both Map and plain object formats
    const centroidsMap = centroidsData instanceof Map 
      ? centroidsData 
      : new Map(Object.entries(centroidsData));
    
    centroidsMap.forEach((centroidArray, cmdId) => {
      // Convert array to Float32Array
      const centroid = new Float32Array(centroidArray);
      
      // Ensure correct dimension
      if (centroid.length !== this.options.embeddingSize) {
        console.warn(`Centroid for '${cmdId}' has dimension ${centroid.length}, expected ${this.options.embeddingSize}. Resizing...`);
        const resized = new Float32Array(this.options.embeddingSize);
        const copyLength = Math.min(centroid.length, this.options.embeddingSize);
        resized.set(centroid.subarray(0, copyLength));
        this.centroids.set(cmdId, resized);
      } else {
        this.centroids.set(cmdId, centroid);
      }
      
      // Initialize convergence state
      this.convergenceState.set(cmdId, {
        converged: false,
        similarity: 0,
        iterations: 0
      });
    });
    
    // Update commands list if needed
    const loadedIds = Array.from(this.centroids.keys());
    const existingIds = this.options.commands.map(c => c.id);
    const newCommands = loadedIds.filter(id => !existingIds.includes(id));
    
    if (newCommands.length > 0) {
      newCommands.forEach(id => {
        this.options.commands.push({
          id: id,
          label: id.charAt(0).toUpperCase() + id.slice(1),
          centroid: null
        });
      });
    }
    
    console.log(`Loaded ${this.centroids.size} pre-trained centroids:`, Array.from(this.centroids.keys()));
  }
  
  /**
   * Static method to load centroids from JSON file
   * @param {string} jsonPath - Path to JSON file
   * @returns {Promise<Object>} Centroids data object
   */
  static async loadCentroidsFromJSON(jsonPath) {
    try {
      const response = await fetch(jsonPath);
      if (!response.ok) {
        throw new Error(`Failed to load centroids: ${response.statusText}`);
      }
      const data = await response.json();
      return data.centroids || data;
    } catch (error) {
      console.error('Error loading centroids from JSON:', error);
      throw error;
    }
  }
  
  reset() {
    // Reset all state
    this.momentum.fill(0);
    this.gateValues.fill(0.5);
    this.iteration = 0;
    
    // Reinitialize centroids
    this.options.commands.forEach(cmd => {
      this.centroids.set(cmd.id, this.randomVector());
      this.convergenceState.set(cmd.id, {
        converged: false,
        similarity: 0,
        iterations: 0
      });
    });
  }
  
  getState() {
    return {
      momentum: Array.from(this.momentum),
      gateValues: Array.from(this.gateValues),
      centroids: Object.fromEntries(
        Array.from(this.centroids.entries()).map(([id, vec]) => [id, Array.from(vec)])
      ),
      convergenceState: Object.fromEntries(this.convergenceState),
      iteration: this.iteration
    };
  }
}

// === USAGE EXAMPLE ===
// Make it available globally
if (typeof window !== 'undefined') {
  window.SemanticSearchBar = SemanticSearchBar;
  
  // Example initialization
  window.initSemanticSearch = function(inputId) {
    const input = document.getElementById(inputId);
    if (!input) {
      console.error(`Input element with id "${inputId}" not found`);
      return null;
    }
    
    return new SemanticSearchBar(input, {
      onCommandDetected: (command, confidence, query) => {
        console.log(`Detected command: ${command} (${(confidence * 100).toFixed(1)}%)`, query);
      },
      onConfidenceChange: (confidences) => {
        // Update UI with confidence scores
        console.log('Confidence scores:', confidences);
      }
    });
  };
}

// === EXPORT FOR NODE.JS ===
if (typeof module !== 'undefined' && module.exports) {
  module.exports = SemanticSearchBar;
}
