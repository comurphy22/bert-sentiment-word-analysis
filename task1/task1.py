from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import time
import random

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Cache for word embeddings to avoid recomputing
embedding_cache = {}

def get_word_embedding(word, tokenizer, model):
    """Get word embedding with caching for efficiency"""
    if word in embedding_cache:
        return embedding_cache[word]
    
    tokens = tokenizer.tokenize(word)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    with torch.no_grad():
        outputs = model(torch.tensor([token_ids]))
    embeddings = outputs.last_hidden_state.squeeze(0)
    word_embedding = embeddings.mean(dim=0).numpy()
    
    # Cache the result
    embedding_cache[word] = word_embedding
    return word_embedding

# Parse the dataset
groups = {}
current_group = None

with open('word-test.v1.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        if line.startswith(':'):
            current_group = line[2:]
            groups[current_group] = []
        elif current_group is not None:
            words = line.lower().split()
            if words:
                groups[current_group].append(words)

def compute_word_analogy(a, b, c, candidates, tokenizer, model, metric="cosine"):
    """
    Compute word analogy a:b::c:? and return ranked candidates
    """
    # Get embeddings (these will use the cache if available)
    a_emb = get_word_embedding(a, tokenizer, model)
    b_emb = get_word_embedding(b, tokenizer, model)
    c_emb = get_word_embedding(c, tokenizer, model)
    
    # Compute relationship vector (a - b)
    relationship = a_emb - b_emb
    
    # Compute candidate scores
    scores = []
    for candidate in candidates:
        d_emb = get_word_embedding(candidate, tokenizer, model)
        
        if metric == "cosine":
            # Compute cosine similarity (higher is better)
            similarity = cosine_similarity([relationship], [c_emb - d_emb])[0][0]
            scores.append((candidate, similarity))
        elif metric == "l2":
            # Compute L2 distance (lower is better)
            distance = euclidean(relationship, c_emb - d_emb)
            scores.append((candidate, distance))
    
    # Sort by score
    if metric == "cosine":
        sorted_candidates = sorted(scores, key=lambda x: x[1], reverse=True)
    else:  # l2
        sorted_candidates = sorted(scores, key=lambda x: x[1], reverse=False)
        
    return sorted_candidates

def evaluate_analogy_task(group_name, k_values=[1, 2, 5, 10, 20], metrics=["cosine", "l2"], max_examples=30):
    """
    Evaluate the word analogy prediction task for a specific group
    
    Parameters:
    - group_name: The name of the group to evaluate
    - k_values: List of k values for top-k accuracy
    - metrics: List of metrics to use ("cosine", "l2")
    - max_examples: Maximum number of examples to evaluate for efficiency
    
    Returns:
    - Dictionary with results for each metric and k value
    """
    if group_name not in groups:
        print(f"Group '{group_name}' not found.")
        return {}
    
    lines = groups[group_name]
    
    # If there are too many examples, randomly sample max_examples
    if len(lines) > max_examples:
        print(f"Sampling {max_examples} examples from {len(lines)} in group '{group_name}'")
        random.seed(42)  # For reproducibility
        lines = random.sample(lines, max_examples)
    
    # Extract all b and d words as candidates from the entire group (not just the sampled examples)
    # This ensures we have a proper candidate pool
    candidates = []
    for line in groups[group_name]:
        if len(line) >= 4:
            candidates.append(line[1])  # b word
            candidates.append(line[3])  # d word
    
    candidates = list(set(candidates))  # Remove duplicates
    print(f"Group '{group_name}' has {len(lines)} examples and {len(candidates)} unique candidates")
    
    # Precompute embeddings for all candidates to improve efficiency
    print("Precomputing embeddings for all candidates...")
    start_time = time.time()
    for word in candidates:
        if word not in embedding_cache:
            get_word_embedding(word, tokenizer, model)
    print(f"Finished precomputing {len(candidates)} embeddings in {time.time() - start_time:.2f} seconds")
    
    results = {metric: {k: 0 for k in k_values} for metric in metrics}
    total_examples = 0
    
    # Process each example
    start_time = time.time()
    for i, line in enumerate(lines):
        if len(line) < 4:
            continue
            
        a, b, c, d = line[0], line[1], line[2], line[3]
        print(f"Processing analogy {i+1}/{len(lines)}: {a}:{b}::{c}:?")
        total_examples += 1
        
        # Make sure we have embeddings for the current words
        for word in [a, b, c, d]:
            if word not in embedding_cache:
                get_word_embedding(word, tokenizer, model)
        
        # Evaluate with each metric
        for metric in metrics:
            predictions = compute_word_analogy(a, b, c, candidates, tokenizer, model, metric)
            
            # Check accuracy for each k value
            for k in k_values:
                top_k = [word for word, _ in predictions[:k]]
                if d in top_k:
                    results[metric][k] += 1
            
            # Print top predictions for first metric only
            if metric == metrics[0]:
                top_5 = [word for word, _ in predictions[:5]]
                print(f"Top 5 predictions ({metric}): {top_5}")
                print(f"Correct answer: {d} {'✓' if d in top_5 else '✗'}")
    
    processing_time = time.time() - start_time
    print(f"Processed {total_examples} examples in {processing_time:.2f} seconds "
          f"({processing_time/total_examples:.2f} seconds per example)")
    
    # Calculate accuracy percentages
    for metric in metrics:
        for k in k_values:
            if total_examples > 0:
                results[metric][k] = (results[metric][k] / total_examples, f"{results[metric][k]}/{total_examples}")
    
    # Display results table
    print(f"\nResults for group '{group_name}' ({total_examples} examples):")
    print("k\tCosine Similarity\tL2 Distance")
    for k in k_values:
        cosine_acc = f"{results['cosine'][k][0]:.4f} ({results['cosine'][k][1]})" if 'cosine' in metrics else "N/A"
        l2_acc = f"{results['l2'][k][0]:.4f} ({results['l2'][k][1]})" if 'l2' in metrics else "N/A"
        print(f"{k}\t{cosine_acc}\t{l2_acc}")
    
    return results

def run_task1(selected_groups=None, max_examples=30):
    """
    Run Task 1 evaluation on selected groups
    
    Parameters:
    - selected_groups: List of group names to evaluate. If None, uses three specific groups.
    - max_examples: Maximum number of examples to evaluate per group for efficiency
    """
    k_values = [1, 2, 5, 10, 20]
    metrics = ["cosine", "l2"]
    
    # Default to three specific groups that are manageable and representative
    if not selected_groups:
        # Select three groups with reasonable sizes for evaluation
        default_groups = ["capital-common-countries", "family", "gram1-adjective-to-adverb"]
        selected_groups = [g for g in default_groups if g in groups]
        
        # If any of the default groups are missing, add groups from what's available
        if len(selected_groups) < 3:
            available_groups = list(groups.keys())
            for group in available_groups:
                if group not in selected_groups:
                    selected_groups.append(group)
                    if len(selected_groups) >= 3:
                        break
    
    print(f"Running Task 1 evaluation on groups: {selected_groups}")
    print(f"Using k values: {k_values}")
    print(f"Using metrics: {metrics}")
    print(f"Maximum examples per group: {max_examples}")
    print("-" * 50)
    
    all_results = {}
    
    for group in selected_groups:
        print(f"\nEvaluating group: {group}")
        results = evaluate_analogy_task(
            group_name=group,
            k_values=k_values,
            metrics=metrics,
            max_examples=max_examples
        )
        all_results[group] = results
        print("-" * 50)
    
    # Final summary
    print("\nTask 1 Summary:")
    for group, results in all_results.items():
        print(f"\nGroup: {group}")
        print("k\tCosine Similarity\tL2 Distance")
        for k in k_values:
            cosine_acc = f"{results['cosine'][k][0]:.4f}" if 'cosine' in metrics else "N/A"
            l2_acc = f"{results['l2'][k][0]:.4f}" if 'l2' in metrics else "N/A"
            print(f"{k}\t{cosine_acc}\t{l2_acc}")
    
    return all_results

def save_results_to_file(results, filename="task1_results.txt"):
    """Save results to a text file"""
    with open(filename, 'w') as f:
        f.write("TASK 1 - WORD ANALOGY PREDICTION RESULTS\n\n")
        
        for group, group_results in results.items():
            f.write(f"GROUP: {group}\n")
            f.write("k\tCosine Similarity\tL2 Distance\n")
            
            for k in [1, 2, 5, 10, 20]:
                cosine_acc = f"{group_results['cosine'][k][0]:.4f} ({group_results['cosine'][k][1]})"
                l2_acc = f"{group_results['l2'][k][0]:.4f} ({group_results['l2'][k][1]})"
                f.write(f"{k}\t{cosine_acc}\t{l2_acc}\n")
            
            f.write("\n")
    
    print(f"Results saved to {filename}")

# Check if we have any groups before trying to select them
if groups:
    print(f"Found {len(groups)} groups:")
    for i, group_name in enumerate(groups.keys()):
        print(f"{i+1}. {group_name} ({len(groups[group_name])} examples)")
    
    # Run Task 1 with three specific groups and a limit of 30 examples per group
    # This gives a good balance between accuracy and runtime
    results = run_task1(max_examples=30)
    
    # Or run with specific groups if you prefer
    # selected_groups = ["capital-common-countries", "family", "gram1-adjective-to-adverb"]
    # results = run_task1(selected_groups=selected_groups, max_examples=30)
    
    # Save results to file
    save_results_to_file(results)
else:
    print("No groups were parsed from the file. Check if the file format is correct.")
    print("Make sure 'word-test.v1.txt' exists and has the right format with group headers starting with ':'")