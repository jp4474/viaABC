import numpy as np

def l1_distance(x, y):
    """Compute the Manhattan Distance (L1) between two vectors."""
    return np.linalg.norm(x - y, ord=1)

def l2_distance(x, y):
    """Compute the Euclidean Distance (L2) between two vectors."""
    return np.linalg.norm(x - y, ord=2)

def cosine_similarity(x, y):
    """Compute the Cosine Similarity between two vectors."""
    x_norm = x / np.linalg.norm(x)
    y_norm = y / np.linalg.norm(y)
    return np.dot(x_norm, y_norm)

def bert_score(x, y):
    """Compute BERTScore (Precision, Recall, F1) between two sets of embeddings."""
    # Normalize the embeddings across the feature dimension (columns)
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)

    # Compute the cosine similarity matrix
    cosine_sim = np.dot(x_norm, y_norm.T)

    # Precision: average of the maximum cosine similarity for each token in x
    precision = np.mean(np.max(cosine_sim, axis=1))

    # Recall: average of the maximum cosine similarity for each token in y
    recall = np.mean(np.max(cosine_sim, axis=0))

    # F1 score: harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1
