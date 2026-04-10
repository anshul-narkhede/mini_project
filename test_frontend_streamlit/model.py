import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'all-MiniLM-L6-v2'


@st.cache_resource
def load_model():
    """Load the sentence-transformer NLP model."""
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def build_index(_model, text_data):
    """Build a FAISS index from the given text data using the model."""
    embeddings = _model.encode(text_data, convert_to_tensor=False)
    embeddings = np.array(embeddings).astype('float32')

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
