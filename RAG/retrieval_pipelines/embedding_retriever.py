import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Tuple, Any # Added List, Tuple, Any for type hinting

# Import the base class
from .base_retriever import BaseRetriever

DEFAULT_EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-7B-instruct"
DEFAULT_EMBEDDING_CACHE = str(Path(__file__).resolve().parents[2] / "hf_cache")

# Inherit from BaseRetriever
class EmbeddingRetriever(BaseRetriever):
    """
    Retrieves relevant text chunks based on semantic similarity using sentence embeddings.
    Inherits from BaseRetriever.
    """
    def __init__(
        self,
        model_name=DEFAULT_EMBEDDING_MODEL,
        max_length=8192,
        cache_folder=DEFAULT_EMBEDDING_CACHE,
    ):
        """
        Initializes the EmbeddingRetriever.

        Args:
            model_name (str): The name of the Hugging Face model to use for embeddings.
            max_length (int): The maximum sequence length for the tokenizer.
        """
        super().__init__() # Call base class initializer
        self.max_length = max_length

        # Auto-detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize SentenceTransformer (handles model + tokenizer + pooling internally)
        model_kwargs = {
            "torch_dtype": torch.float16,
            "attn_implementation": "eager",
        } if device == "cuda" else {
            "torch_dtype": torch.float32,
            "attn_implementation": "eager",
        }

        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
            cache_folder=cache_folder,
            model_kwargs=model_kwargs,
        )
        # CRITICAL: Expose tokenizer for external consumers (e.g., token_counter, create_databases)
        self.tokenizer = self.model.tokenizer
        # Set max sequence length so SentenceTransformer handles truncation
        self.model.max_seq_length = self.max_length

    # Implement the abstract method from BaseRetriever
    def vectorize_text(self, text_chunk: str) -> List[List[float]]:
        """
        Vectorizes a single text chunk into an embedding.

        Delegates to SentenceTransformer which handles tokenization, model forward pass,
        and pooling internally.

        Args:
            text_chunk (str): The text content to vectorize.

        Returns:
            List[List[float]]: The embedding vector, wrapped in a list.
        """
        embedding_np = self.model.encode(
            text_chunk,
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # encode() returns a numpy array of shape (embed_dim,) for a single string.
        # Wrap in a list to match the existing List[List[float]] return type contract.
        return [embedding_np.tolist()]

    # Implement the abstract method from BaseRetriever
    def retrieve_relevant_chunks(
        self,
        query_representation: List[float], # Expecting a single query embedding vector
        document_representations: List[List[float]], # Expecting a list of document embedding vectors
        document_chunks_text: List[str],
        top_k: int = 3
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieves the most relevant document chunks based on cosine similarity
        between the query embedding and document embeddings.

        Note: This method calculates similarity in memory. For large datasets,
        using a vector database query (like ChromaDB's collection.query) is
        typically more efficient and is the approach used in rag_tester.py.
        This implementation fulfills the BaseRetriever contract but might not be
        directly used by rag_tester.py for embedding retrieval.

        Args:
            query_representation (List[float]): The embedding vector of the query.
            document_representations (List[List[float]]): A list of embedding vectors
                                                         for the documents.
            document_chunks_text (List[str]): The original text of the document chunks.
            top_k (int): The maximum number of relevant chunks to retrieve.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing:
                - A list of the most relevant document chunk texts.
                - A list of corresponding cosine similarity scores.
        """
        if not document_representations or not query_representation:
            return [], []

        # Ensure tensors are on the same device if using GPU
        # device = self.model.device # Or self.device if defined in __init__
        # query_embedding_tensor = torch.tensor(query_representation, device=device).unsqueeze(0) # Add batch dim
        # document_embeddings_tensor = torch.tensor(document_representations, device=device)

        # Simpler CPU version:
        query_embedding_tensor = torch.tensor(query_representation).unsqueeze(0) # Shape: (1, embed_dim)
        document_embeddings_tensor = torch.tensor(document_representations) # Shape: (num_docs, embed_dim)

        # Calculate cosine similarities (efficiently)
        # Normalize tensors just in case they aren't already (F.normalize was used in vectorize_text)
        query_embedding_tensor = F.normalize(query_embedding_tensor, p=2, dim=1)
        document_embeddings_tensor = F.normalize(document_embeddings_tensor, p=2, dim=1)

        # Cosine similarity = dot product of normalized vectors
        similarities = torch.matmul(document_embeddings_tensor, query_embedding_tensor.T).squeeze() # Shape: (num_docs,)

        # Get top_k results
        # Ensure top_k is not larger than the number of documents
        actual_top_k = min(top_k, len(document_chunks_text))
        if actual_top_k <= 0:
            return [], []

        # .topk returns values and indices
        top_k_similarities, top_k_indices = torch.topk(similarities, k=actual_top_k)

        # Convert indices and similarities to lists
        top_k_indices_list = top_k_indices.tolist()
        top_k_similarities_list = top_k_similarities.tolist()

        # Retrieve the corresponding text chunks
        relevant_chunks = [document_chunks_text[i] for i in top_k_indices_list]

        return relevant_chunks, top_k_similarities_list
