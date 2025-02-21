import numpy as np

def l1_distance(x, y):
    return np.linalg.norm(x-y, ord=1)

def l2_distance(x, y):
    return np.linalg.norm(x-y, ord=2)

def cosine_similarity(x, y):
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    return np.dot(x, y)

def bert_score(x, y):
    # Normalize the embeddings across the feature dimension (columns)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)

    # Compute the cosine similarity matrix
    cosine_sim = np.dot(x, y.T)

    # Precision: average of the maximum cosine similarity for each token in x
    precision = np.mean(np.max(cosine_sim, axis=1))

    # Recall: average of the maximum cosine similarity for each token in y
    recall = np.mean(np.max(cosine_sim, axis=0))

    # F1 score: harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1