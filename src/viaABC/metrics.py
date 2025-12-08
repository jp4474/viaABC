import numpy as np

def l1_distance(x, y):
    """Compute the Manhattan Distance (L1) between two vectors."""
    #return np.linalg.norm(y - x, ord=1)
    return np.sum(np.abs(y - x), axis=-1)

def l2_distance(x, y):
    """Compute the Euclidean Distance (L2) between two vectors."""
    return np.linalg.norm(y - x, ord=2, axis=-1)

def cosine_similarity(x, y):
    """Compute the Cosine Similarity between two vectors."""
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)

    y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)

    # Compute the cosine similarity matrix for each batch
    cosine_sim = np.dot(x_norm, y_norm.T)

    return cosine_sim.item()

def bert_score(x, y):
    """Compute BERTScore (Precision, Recall, F1) between two sets of embeddings."""
    # Normalize the embeddings across the feature dimension (columns)
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    
    # If y is 3D, normalize each batch individually
    if y.ndim == 3:
        y_norm = y / np.linalg.norm(y, axis=2, keepdims=True)
    else:
        y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)

    # Compute the cosine similarity matrix for each batch
    cosine_sim = np.matmul(x_norm, y_norm.transpose(0, 2, 1)) if y.ndim == 3 else np.dot(x_norm, y_norm.T)

    # Precision: average of the maximum cosine similarity for each token in x
    precision = np.mean(np.max(cosine_sim, axis=-1), axis=-1)  # Max along last axis for each x in each batch
    
    # Recall: average of the maximum cosine similarity for each token in y
    recall = np.mean(np.max(cosine_sim, axis=-2), axis=-1)  # Max along second-to-last axis for each y in each batch

    # F1 score: harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

def bert_score_batch(x, y):
    scores = []
    for a, b in zip(x, y):
        precision, recall, f1 = bert_score(a, b)
        scores.append(f1)
    scores = np.array(scores)
    return scores.mean()

def pairwise_cosine(x, y):
    assert x.ndim == 3 and y.ndim == 3, "Input arrays must be 3D"

    x_norm = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y_norm = y / np.linalg.norm(y, axis=-1, keepdims=True)

    cos_sim = np.mean(np.sum(x_norm * y_norm, axis=-1), axis=-1)[0]

    return cos_sim

def maxSim(x, y):
    """
    x: [num_query_tokens, dim] - query embeddings (numpy array)
    y: [num_doc_tokens, dim] - document embeddings (numpy array)
    
    Returns:
        scalar maxSim score (float)
    """
    # remove 1st dimension
    if x.ndim == 3:
        x = x.squeeze(0)  # [Q, D]
    if y.ndim == 3:
        y = y.squeeze(0)

    # L2 normalize embeddings
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)  # [Q, D]
    y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)  # [D, D]

    # Compute cosine similarity matrix [Q, D]
    sim_matrix = np.dot(x_norm, y_norm.T)

    # Take max over document tokens for each query token, then sum
    max_sim = np.max(sim_matrix, axis=1).sum()
    
    return float(max_sim)