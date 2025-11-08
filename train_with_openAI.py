"""
Train semantic centroids using OpenAI embeddings
Requires: pip install openai
Set environment variable: OPENAI_API_KEY
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import os

try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI library not installed")
    print("Install with: pip install openai")
    exit(1)

def create_training_examples() -> Dict[str, List[str]]:
    """Same training examples as Sentence Transformers version"""
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
        ],
        'navigate': [
            'navigate', 'go to', 'move to', 'travel', 'walk',
            'navigate to home', 'go to settings', 'move to dashboard',
            'travel to folder', 'walk through menu', 'jump to section',
            'open page', 'enter directory', 'visit location',
            'switch to view', 'change screen', 'redirect to',
            'go back', 'move forward', 'jump up', 'go down',
        ],
        'execute': [
            'execute', 'run', 'perform', 'start', 'begin', 'launch',
            'execute command', 'run process', 'perform action',
            'start task', 'begin operation', 'launch program',
            'execute script', 'run batch', 'perform workflow',
            'start service', 'begin job', 'launch application',
        ],
        'help': [
            'help', 'assist', 'support', 'guide', 'explain',
            'help me', 'assist please', 'support needed',
            'guide me through', 'explain how', 'teach me',
            'show instructions', 'display help', 'provide guidance',
            'what is this', 'how does this work', 'tell me about',
        ]
    }

def train_centroids_with_openai(
    model: str = 'text-embedding-3-small',
    output_file: str = 'centroids.json'
):
    """
    Train semantic centroids using OpenAI embeddings
    
    Args:
        model: OpenAI embedding model
               - 'text-embedding-3-small' (1536 dim, $0.02/1M tokens)
               - 'text-embedding-3-large' (3072 dim, $0.13/1M tokens)
        output_file: Output JSON file path
    """
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    print(f"Using OpenAI model: {model}")
    client = OpenAI(api_key=api_key)
    
    # Get training examples
    training_examples = create_training_examples()
    
    print("\nGenerating embeddings for training examples...")
    centroids = {}
    embedding_dim = None
    
    for command_id, examples in training_examples.items():
        print(f"\nProcessing '{command_id}' with {len(examples)} examples")
        
        # Generate embeddings via OpenAI API
        response = client.embeddings.create(
            input=examples,
            model=model
        )
        
        # Extract embeddings
        embeddings = np.array([item.embedding for item in response.data])
        
        if embedding_dim is None:
            embedding_dim = embeddings.shape[1]
            print(f"Embedding dimension: {embedding_dim}")
        
        # Create centroid by averaging
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        centroids[command_id] = centroid.tolist()
        
        # Statistics
        similarities = [np.dot(centroid, emb / np.linalg.norm(emb)) for emb in embeddings]
        print(f"  Centroid created: avg similarity = {np.mean(similarities):.3f}")
    
    # Analyze separation
    print("\n=== Centroid Separation Analysis ===")
    command_ids = list(centroids.keys())
    for i, cmd1 in enumerate(command_ids):
        for cmd2 in command_ids[i+1:]:
            vec1 = np.array(centroids[cmd1])
            vec2 = np.array(centroids[cmd2])
            similarity = np.dot(vec1, vec2)
            print(f"  {cmd1} <-> {cmd2}: similarity = {similarity:.3f}, distance = {1-similarity:.3f}")
    
    # Save
    output = {
        "embeddingSize": embedding_dim,
        "model": model,
        "commands": list(centroids.keys()),
        "centroids": centroids
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ SUCCESS! Saved to: {output_file}")
    print(f"Remember to update embeddingSize to {embedding_dim} in tempfile.js!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='text-embedding-3-small')
    parser.add_argument('--output', default='centroids.json')
    args = parser.parse_args()
    
    train_centroids_with_openai(args.model, args.output)

