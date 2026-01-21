from src.models import get_embedding
from src.utils import normalize_vector

import numpy as np
import faiss

# Function to build an HNSW index
def build_faiss_hnsw_index(dimension, ef_construction=200, M=32):
    """
    build_faiss_hnsw_index: Add a description here.

    Args:
        # List the arguments with types and descriptions.

    Returns:
        # Specify what the function returns.
    """
    """
    Builds a FAISS HNSW index for cosine similarity.

    Parameters:
        dimension (int): Dimensionality of the embeddings.
        ef_construction (int): Trade-off parameter between index construction speed and accuracy.
        M (int): Number of neighbors in the graph.

    Returns:
        index (faiss.IndexHNSWFlat): Initialized FAISS HNSW index.
    """
    index = faiss.IndexHNSWFlat(dimension, M)  # HNSW index
    index.hnsw.efConstruction = ef_construction  # Construction accuracy
    index.metric_type = faiss.METRIC_INNER_PRODUCT  # Cosine similarity via normalized vectors
    return index

# Function to populate the FAISS index
def populate_faiss_index(index: faiss.Index, documents: dict, batch_size: int=20):
    """
    populate_faiss_index: Add a description here.

    Args:
        # List the arguments with types and descriptions.

    Returns:
        # Specify what the function returns.
    """
    """
    Populates the FAISS HNSW index with normalized embeddings from the dataset.

    Parameters:
        index (faiss.Index): FAISS index to populate.
        documents (pd.Series): documents like List[list[str]]
        batch_size (int): Number of questions to process at a time.
    """
    buffer = []
    i = 0

    for _, embedding in documents.items():
        embedding = normalize_vector(embedding)
        buffer.append(embedding)
        i += 1

        # Add embeddings to the index in batches
        if len(buffer) >= batch_size:
            index.add(np.array(buffer, dtype=np.float32))
            buffer = []

    # Add remaining embeddings
    if buffer:
        index.add(np.array(buffer, dtype=np.float32))

# Function to perform a search query
def search_faiss_index(embeddings_storage, query, k=5):
    """
    search_faiss_index: Add a description here.

    Args:
        # List the arguments with types and descriptions.

    Returns:
        # Specify what the function returns.
    """
    """
    Searches the FAISS index for the closest matches to a query.

    Parameters:
        embeddings_storage (faiss.Index): FAISS index to search.
        query (str): Query string to search.
        k (int): Number of closest matches to retrieve.

    Returns:
        indices (np.ndarray): Indices of the top-k results.
        distances (np.ndarray): Distances of the top-k results.
    """
    # Preprocess and normalize the query embedding
    query_embedding = get_embedding(query)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    query_embedding = normalize_vector(query_embedding)
    # Search the embeddings_storage
    top_k_distances, top_k_indices = embeddings_storage.search(np.array([query_embedding], dtype=np.float32), k)

    # Match return format with that used in numpy storage search
    # Note that list manipulations will give an overhead
    top_k_indices_list = top_k_indices[0].tolist()
    top_k_distances_list = top_k_distances[0].tolist()

    return top_k_indices_list, top_k_distances_list

