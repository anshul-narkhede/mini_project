"""
NLP model loading and FAISS vector index construction.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import MODEL_NAME


def load_model() -> SentenceTransformer:
    """Load the sentence-transformer model."""
    return SentenceTransformer(MODEL_NAME)


def build_faiss_index(model: SentenceTransformer, texts: list[str]) -> faiss.IndexFlatL2:
    """
    Encode a list of texts into 384-dim vectors and store them
    in a FAISS IndexFlatL2 for exact nearest-neighbor search.
    """
    embeddings = model.encode(texts, convert_to_tensor=False)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index
