# нормализовка векторов
def normalize_vector(embedding: np.ndarray) -> np.ndarray:
    """
    Normalizes a given vector to have unit length.

    Args:
        embedding (np.ndarray): A NumPy array representing the vector to normalize.

    Returns:
        np.ndarray: A normalized vector with unit length.
    """

    norm = np.linalg.norm(embedding)
    if abs(norm) >= 1e-9: #защита от взрыва и погрешности
      embedding /= norm

    return embedding


# Убрать стоп слова и знаки припенания

# Почистить текст
# Обновлено 28.12.2024 в 14:06
