"""
Train semantic centroids using real embeddings from Sentence Transformers
This creates properly separated semantic representations for commands
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Dict, List

def create_training_examples() -> Dict[str, List[str]]:
    """
    Create diverse training examples for each command
    More examples = better centroid quality
    """
    return {
        'search': [
            'search', 'find', 'look for', 'query', 'locate', 'discover',
            'search for files', 'find documents', 'look up data',
            'query database', 'locate records', 'discover information',
            'search through files', 'find all documents', 'look for specific data',
            'query the system', 'locate user files', 'discover hidden files',
            'search by keyword', 'find by name', 'look up by id',
            'filter results', 'scan directory', 'retrieve data',
            'fetch records', 'list files', 'show documents',
            'search functionality', 'lookup tool', 'find utility',
            'I need to search for something', 'Can you help me find files',
            'Where can I look for documents', 'How do I query the database'
        ],
        
        'navigate': [
            'navigate', 'go to', 'move to', 'travel', 'walk',
            'navigate to home', 'go to settings', 'move to dashboard',
            'travel to folder', 'walk through menu', 'jump to section',
            'open page', 'enter directory', 'visit location',
            'switch to view', 'change screen', 'redirect to',
            'go back', 'move forward', 'jump up', 'go down',
            'navigate menu', 'browse folders', 'explore directories',
            'route to page', 'direct me to', 'take me to',
            'I want to go to settings', 'Can you navigate to home',
            'Please take me to the dashboard', 'How do I get to the folder',
            'Show me the way to', 'Direct me to the page'
        ],
        
        'execute': [
            'execute', 'run', 'perform', 'start', 'begin', 'launch',
            'execute command', 'run process', 'perform action',
            'start task', 'begin operation', 'launch program',
            'execute script', 'run batch', 'perform workflow',
            'start service', 'begin job', 'launch application',
            'trigger action', 'initiate process', 'activate function',
            'invoke command', 'call procedure', 'run function',
            'execute now', 'run immediately', 'start right away',
            'I need to execute a command', 'Can you run this process',
            'Please perform this action', 'How do I start the task',
            'Execute the backup', 'Run the sync', 'Start the import'
        ],
        
        'help': [
            'help', 'assist', 'support', 'guide', 'explain',
            'help me', 'assist please', 'support needed',
            'guide me through', 'explain how', 'teach me',
            'show instructions', 'display help', 'provide guidance',
            'what is this', 'how does this work', 'tell me about',
            'help documentation', 'support docs', 'user guide',
            'show examples', 'provide tutorial', 'explain feature',
            'I need help', 'Can you assist me', 'Please support',
            'How do I use this', 'What does this do', 'Explain the feature',
            'Show me how', 'Give me instructions', 'Provide examples',
            'Help with commands', 'Support for features', 'Guide for beginners'
        ]
    }

def train_centroids_with_real_embeddings(
    model_name: str = 'all-MiniLM-L6-v2',
    output_file: str = 'centroids.json'
):
    """
    Train semantic centroids using Sentence Transformers
    
    Args:
        model_name: HuggingFace model name (default: fast 384-dim model)
                    Alternatives:
                    - 'all-MiniLM-L6-v2' (384 dim, fast, good quality)
                    - 'all-mpnet-base-v2' (768 dim, slower, better quality)
                    - 'all-MiniLM-L12-v2' (384 dim, medium speed, good quality)
        output_file: Output JSON file path
    """
    
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {embedding_dim}")
    
    # Get training examples
    training_examples = create_training_examples()
    
    print("\nGenerating embeddings for training examples...")
    centroids = {}
    
    for command_id, examples in training_examples.items():
        print(f"\nProcessing '{command_id}' with {len(examples)} examples")
        
        # Generate embeddings for all examples
        embeddings = model.encode(examples, convert_to_numpy=True, show_progress_bar=True)
        
        # Create centroid by averaging all example embeddings
        centroid = np.mean(embeddings, axis=0)
        
        # Normalize centroid
        centroid = centroid / np.linalg.norm(centroid)
        
        # Store as list for JSON serialization
        centroids[command_id] = centroid.tolist()
        
        # Calculate statistics
        similarities = [
            np.dot(centroid, emb / np.linalg.norm(emb)) 
            for emb in embeddings
        ]
        avg_sim = np.mean(similarities)
        min_sim = np.min(similarities)
        
        print(f"  Centroid created: avg similarity to examples = {avg_sim:.3f}, min = {min_sim:.3f}")
    
    # Calculate inter-centroid distances (separation)
    print("\n=== Centroid Separation Analysis ===")
    command_ids = list(centroids.keys())
    
    for i, cmd1 in enumerate(command_ids):
        for cmd2 in command_ids[i+1:]:
            vec1 = np.array(centroids[cmd1])
            vec2 = np.array(centroids[cmd2])
            similarity = np.dot(vec1, vec2)
            distance = 1 - similarity
            print(f"  {cmd1} <-> {cmd2}: similarity = {similarity:.3f}, distance = {distance:.3f}")
    
    # Create output structure
    output = {
        "embeddingSize": embedding_dim,
        "model": model_name,
        "commands": list(centroids.keys()),
        "centroids": centroids,
        "metadata": {
            "num_training_examples": {cmd: len(examples) for cmd, examples in training_examples.items()},
            "training_method": "averaging",
            "normalized": True
        }
    }
    
    # Save to JSON
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ SUCCESS! Trained centroids saved to: {output_path}")
    print(f"   Model: {model_name}")
    print(f"   Embedding dimension: {embedding_dim}")
    print(f"   Commands: {list(centroids.keys())}")
    print("\nNext steps:")
    print("1. Update tempfile.js to use the new embedding dimension")
    print("2. Open semantic-search-example.html to test")
    print("3. The centroids should now have much better separation!")
    
    return centroids

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train semantic centroids with real embeddings')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence Transformer model name')
    parser.add_argument('--output', type=str, default='centroids.json',
                        help='Output JSON file path')
    
    args = parser.parse_args()
    
    train_centroids_with_real_embeddings(
        model_name=args.model,
        output_file=args.output
    )

