import numpy as np

def l1_distance(x, y):
    """Compute the Manhattan Distance (L1) between two vectors."""
    return np.linalg.norm(x - y, ord=1)

def l2_distance(x, y):
    """Compute the Euclidean Distance (L2) between two vectors."""
    return np.linalg.norm(x - y, ord=2)

def cosine_similarity(x, y):
    """Compute the Cosine Similarity between two vectors."""
    x_norm = x / np.linalg.norm(x, axis=0, keepdims=True)

    y_norm = y / np.linalg.norm(y, axis=0, keepdims=True)

    # Compute the cosine similarity matrix for each batch
    cosine_sim = np.dot(x_norm, y_norm.T)

    return cosine_sim


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