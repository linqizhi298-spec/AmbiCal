import chromadb
from chromadb.utils import embedding_functions
import hashlib
# Import specific retrievers
from retrieval_pipelines.embedding_retriever import EmbeddingRetriever, DEFAULT_EMBEDDING_MODEL
from retrieval_pipelines.ambical import (
    AmbiCalConfig,
    AmbiCalProfile,
    AmbiCalRetriever,
)
# Import BaseRetriever for type hinting
from retrieval_pipelines.base_retriever import BaseRetriever
from evaluation.evaluation_metrics import evaluate_rag_pipeline
from analysis.analysis_tools import (
    analyze_evaluation_results,
    load_dataset,
    analyze_dataset_across_types,
)
import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional # Import Optional

# --- Configuration ---
persist_directory = "chroma_db_gte_qwen2_7b"
config_file_path = "config.json"  # Path to config file

# --- Helper Functions ---
def generate_chunk_id(text_chunk: str) -> str:
    """Generates a unique ID for a text chunk."""
    return hashlib.sha256(text_chunk.encode("utf-8")).hexdigest()

# Update return type hint to BaseRetriever
# Update signature to accept chroma_client and collection_name
def initialize_retriever(
    retrieval_strategy_str: str,
    chroma_client: Optional[chromadb.ClientAPI] = None, # Add chroma_client
    collection_name: Optional[str] = None, # Add collection_name
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    embedding_max_length: int = 8192,
    ambical_config: Optional[AmbiCalConfig] = None,
    ambical_profile: Optional[AmbiCalProfile] = None,
    ambical_translator: Any = None,
    ambical_reranker: Any = None,
) -> Any:
    """
    Initializes the retriever based on the specified strategy.

    Args:
        retrieval_strategy_str (str): The retrieval strategy ('embedding' or
            'ambical').
        chroma_client (Optional[chromadb.ClientAPI]): The ChromaDB client instance
            (required for ambical).
        collection_name (Optional[str]): The ChromaDB collection name (required for
            ambical).
        ambical_config: Fully resolved AmbiCal method configuration.
        ambical_profile: Existing read-only calibration profile, when required.
        ambical_translator: Deterministic translator used by AmbiCal.

    Returns:
        BaseRetriever: An instance of the specified retriever.

    Raises:
        ValueError: If the strategy is unknown or required arguments are missing.
    """
    print(f"Initializing retriever for strategy: '{retrieval_strategy_str}'...") # Added logging
    if retrieval_strategy_str == "embedding":
        # EmbeddingRetriever doesn't strictly need client/collection at init
        # It's used externally in rag_tester for querying ChromaDB
        return EmbeddingRetriever(
            model_name=embedding_model_name,
            max_length=embedding_max_length,
        )
    elif retrieval_strategy_str == "ambical":
        if chroma_client is None or not collection_name:
            raise ValueError(
                "AmbiCalRetriever requires chroma_client and collection_name "
                "during initialization."
            )
        if ambical_config is None:
            raise ValueError(
                "AmbiCalRetriever requires a resolved ambical_config."
            )
        collection = chroma_client.get_collection(name=collection_name)
        embedding_retriever = EmbeddingRetriever(
            model_name=embedding_model_name,
            max_length=embedding_max_length,
        )
        return AmbiCalRetriever(
            collection=collection,
            embedding_retriever=embedding_retriever,
            config=ambical_config,
            translator=ambical_translator,
            profile=ambical_profile,
            reranker=ambical_reranker,
        )
    # elif retrieval_strategy_str == "some_other_future_strategy":
    #     raise NotImplementedError("This other strategy is not yet implemented.")
    else:
        raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy_str}")

# Update retriever parameter type hint to BaseRetriever
def embed_and_add_chunks_to_db(
    document_chunks_text: List[str],
    collection: chromadb.Collection,
    retriever: BaseRetriever # Use BaseRetriever type hint
) -> None:
    # Note: This function is primarily for *populating* the DB and assumes an
    # EmbeddingRetriever is used during DB creation.
    """Embeds text chunks and adds them to ChromaDB, avoiding duplicates WITHIN this specific collection."""
    # This function remains largely the same, but operates on the specific collection passed to it.
    added_count = 0
    skipped_count = 0
    error_count = 0
    total_chunks = len(document_chunks_text)
    collection_name = collection.name # Get name for logging
    print(f"Starting embedding process for {total_chunks} chunks into collection '{collection_name}'...")

    # Determine if the retriever is embedding-capable for adding to Chroma
    is_embedding_retriever = isinstance(retriever, EmbeddingRetriever)

    if not is_embedding_retriever:
        print(f"Warning: The provided retriever ({type(retriever).__name__}) may not be suitable for generating embeddings required by ChromaDB 'add'. Proceeding by adding documents only.")
        # Fallback: Add documents only if no embedding capability found
        for i, chunk_text in enumerate(document_chunks_text):
            chunk_id = generate_chunk_id(chunk_text)
            try:
                existing = collection.get(ids=[chunk_id], include=[])
                if existing["ids"]:
                    skipped_count += 1
                else:
                    collection.add(documents=[chunk_text], ids=[chunk_id])
                    added_count += 1
                if (i + 1) % 50 == 0 or (i + 1) == total_chunks:
                    print(f"  Collection '{collection_name}': Processed {i + 1}/{total_chunks} chunks (Added Docs Only: {added_count}, Skipped: {skipped_count}, Errors: {error_count})")
            except Exception as e:
                print(f"!!! ERROR processing chunk {i} (doc only) for collection '{collection_name}' (ID: {chunk_id[:10]}...): {e}")
                error_count += 1
        print(f"Document-only adding finished for collection '{collection_name}'. Added: {added_count}, Skipped: {skipped_count}, Errors: {error_count}")
        return # Exit function after document-only add

    # Proceed with embedding if retriever is suitable
    embedder_to_use = retriever if is_embedding_retriever else internal_embedding_retriever

    for i, chunk_text in enumerate(document_chunks_text):
        chunk_id = generate_chunk_id(chunk_text)
        try:
            # Check if chunk ID already exists efficiently
            existing = collection.get(
                ids=[chunk_id], include=[]
            )  # Don't include embeddings/docs
            if existing["ids"]:
                # print(f"Chunk {i} (ID {chunk_id[:10]}...) exists. Skipping.")
                skipped_count += 1
            else:
                # Use the appropriate embedder's vectorize_text method
                # Assuming it returns List[List[float]]
                chunk_embedding = embedder_to_use.vectorize_text(chunk_text)

                if isinstance(chunk_embedding, list) and isinstance(chunk_embedding[0], list) and isinstance(chunk_embedding[0][0], float):
                     # Standard embedding format
                     collection.add(
                         embeddings=chunk_embedding, # Should be List[List[float]]
                         documents=[chunk_text],
                         ids=[chunk_id]
                     )
                     added_count += 1
                else:
                     # Handle unexpected format from embedder
                     print(f"Warning: Chunk {i} embedding format unexpected ({type(chunk_embedding)}). Adding document only.")
                     collection.add(
                         documents=[chunk_text],
                         ids=[chunk_id]
                         # Potentially add metadata if vectorize_text produced keywords, etc.
                         # metadatas=[{"keywords": chunk_representation}] # Example
                     )
                     added_count += 1 # Or handle differently

            if (i + 1) % 50 == 0 or (i + 1) == total_chunks:
                 print(f"  Collection '{collection_name}': Processed {i + 1}/{total_chunks} chunks (Added w/ Embeddings: {added_count}, Skipped: {skipped_count}, Errors: {error_count})")
        except Exception as e:
            print(f"!!! ERROR processing chunk {i} for collection '{collection_name}' (ID: {chunk_id[:10]}...): {e}")
            error_count += 1
            # Optionally continue or break depending on severity
            # continue
    print(
        f"Embedding finished for collection '{collection_name}'. Added: {added_count}, Skipped (already exist): {skipped_count}, Errors: {error_count}"
    )


# --- Main Execution Block ---
# This code only runs when rag_pipeline.py is executed directly
if __name__ == "__main__":

    if os.path.exists(config_file_path):
        with open(config_file_path, 'r', encoding='utf-8-sig') as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"Configuration file not found at: {config_file_path}")

    rag_params = config.get("rag_parameters", {})
    language_configs = config.get("language_configs", [])
    dataset_paths = config.get("question_dataset_paths", {})
    if not language_configs:
        raise ValueError("No 'language_configs' found in config.json. Please define language-specific settings.")
    if not dataset_paths:
        raise ValueError("No 'question_dataset_paths' found in config.json.")

    embedding_model_name = config.get("embedding_model_name", DEFAULT_EMBEDDING_MODEL)
    embedding_max_length = int(config.get("embedding_max_length", 8192))
    temp_embed_retriever = EmbeddingRetriever(
        model_name=embedding_model_name,
        max_length=embedding_max_length,
    )
    tokenizer: Any = temp_embed_retriever.tokenizer
    gte_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model_name
    )
    persist_directory = config.get("chroma_persist_dir", "chroma_db_gte_qwen2_7b")
    chroma_client = chromadb.PersistentClient(path=persist_directory)

    print("\n--- Running RAG Pipeline Setup (Embedding & Evaluation) for Configured Languages ---")

    # --- Get common RAG parameters ---
    # Allow overriding chunk/overlap from config if needed, otherwise use defaults
    # These parameters are now primarily used for naming collections and splitting text.
    chunk_sizes_to_process = rag_params.get("chunk_sizes_to_test", [2000]) # Default if not in config
    overlap_sizes_to_process = rag_params.get("overlap_sizes_to_test", [50]) # Default if not in config

    # --- Loop through chunk/overlap combinations defined in config ---
    # This script now creates DBs for all combinations specified in the config's test parameters
    for chunk_size in chunk_sizes_to_process:
        for overlap_size in overlap_sizes_to_process:
            print(f"\n{'='*10} Processing for Chunk Size: {chunk_size}, Overlap: {overlap_size} {'='*10}")

            # --- Loop through each language configuration ---
            for lang_config in language_configs:
                language = lang_config.get("language")
                manual_path = lang_config.get("manual_path")
                base_collection_name = lang_config.get("collection_base_name")

                if not all([language, manual_path, base_collection_name]):
                    print(f"Warning: Skipping invalid language config entry: {lang_config}")
                    continue

                print(f"\n===== Processing Language: {language.upper()} =====")
                print(f"Manual Path: {manual_path}")

                # --- Determine Dynamic Collection Name ---
                dynamic_collection_name = f"{base_collection_name}_cs{chunk_size}_os{overlap_size}"
                print(f"Target collection name: '{dynamic_collection_name}'")

                # --- Check if Collection Exists ---
                collection_exists = False
                collection = None
                try:
                    # Use the specific embedding function when getting/creating embedding collections
                    collection = chroma_client.get_collection(
                        name=dynamic_collection_name,
                        embedding_function=gte_embedding_function # Needed if collection stores embeddings
                    )
                    print(f"Collection '{dynamic_collection_name}' already exists.")
                    collection_exists = True
                except Exception as e:
                    # Use specific exception check if possible, e.g., chromadb.errors.CollectionNotFoundError
                    # For simplicity, catching general Exception here.
                    # print(f"Collection '{dynamic_collection_name}' does not exist yet (Error: {e}). Will proceed with creation.")
                    print(f"Collection '{dynamic_collection_name}' does not exist yet. Will proceed with creation.")
                    collection_exists = False

                # --- Conditional Embedding/Processing ---
                if not collection_exists:
                    print(f"Creating new collection: '{dynamic_collection_name}'")
            # Create the collection explicitly
                    collection = chroma_client.create_collection(
                        name=dynamic_collection_name,
                        embedding_function=gte_embedding_function # Specify EF at creation
                        # Add metadata if needed: metadata={"hnsw:space": "cosine"} # Example
                    )

                    # Ensure tokenizer is available if needed for splitting
                    if not tokenizer:
                         print("Warning: Tokenizer not found, required for token splitting based on HuggingFace model.")
                         # Decide if this is fatal or if an alternative splitter can be used
                         # Let's try to proceed but log the warning.


                    print(f"Loading text from: {manual_path}")
                    try:
                        with open(manual_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    except FileNotFoundError:
                        print(f"!!! ERROR: Manual file not found at '{manual_path}'. Skipping processing for {language} / CS={chunk_size} / OS={overlap_size}.")
                        continue # Skip to the next language

                    # Define the splitter using current parameters
                    # Ensure tokenizer is available for this specific splitter
                    if tokenizer:
                        token_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                            tokenizer=tokenizer,
                            chunk_size=chunk_size,
                            chunk_overlap=overlap_size
                        )
                        print(f"Splitting text using token limits: chunk_size={chunk_size}, overlap={overlap_size}")
                        document_chunks_text = token_splitter.split_text(text)
                        print(f"Generated {len(document_chunks_text)} token-based chunks for {language}.")
                    else:
                        # Fallback or error if tokenizer needed but not found
                        print(f"Warning: Tokenizer not available. Using basic character splitter.")
                        # Example fallback:
                        char_splitter = RecursiveCharacterTextSplitter(
                             chunk_size=chunk_size * 4, # Heuristic: multiply by avg token length
                             chunk_overlap=overlap_size * 4, # Heuristic
                             length_function=len,
                             is_separator_regex=False,
                        )
                        document_chunks_text = char_splitter.split_text(text)
                        print(f"Generated {len(document_chunks_text)} character-based chunks for {language}.")


                    # --- Embedding/Processing and Adding Chunks ---
                    # For populating the database, we should use an EmbeddingRetriever
                    # to ensure embeddings are generated correctly.
                    # The global 'retriever' is commented out, so let's instantiate one here.
                    db_populating_retriever = EmbeddingRetriever() # Use EmbeddingRetriever for DB population
                    embed_and_add_chunks_to_db(document_chunks_text, collection, db_populating_retriever)
                else:
                    print(f"Skipping processing for existing collection '{dynamic_collection_name}'.")

                # --- Evaluation (Optional, run against the current collection) ---
                # This evaluation likely assumes an embedding retriever/collection setup
                print(f"\n--- Running RAG Pipeline Evaluation (Retrieval Metrics on '{dynamic_collection_name}') ---")

                if collection is None:
                     # Try to get the collection again if it existed but wasn't assigned
                     try:
                         collection = chroma_client.get_collection(name=dynamic_collection_name, embedding_function=gte_embedding_function)
                     except Exception:
                          print(f"Error: Collection object '{dynamic_collection_name}' not available for evaluation. Skipping evaluation.")
                          continue # Skip evaluation

                # Use the English question datasets for evaluation
                all_evaluation_results_for_lang_combo = {}
                for dataset_name, dataset_path in dataset_paths.items():
                    dataset = load_dataset(dataset_path)
                    if dataset:
                        print(f"\n--- Evaluating Retrieval on {dataset_name} dataset using collection '{dynamic_collection_name}' ---")
                        # For evaluation here, let's use an EmbeddingRetriever as well,
                        # since evaluate_rag_pipeline likely expects embedding logic.
                        eval_retriever = EmbeddingRetriever()
                        evaluation_results = evaluate_rag_pipeline(dataset, eval_retriever, collection, rag_params)
                        all_evaluation_results_for_lang_combo[dataset_name] = evaluation_results
                        analysis_label = f"{dataset_name} ({language} manual, CS={chunk_size}, OS={overlap_size})"
                        analyze_evaluation_results(evaluation_results, analysis_label)

                print(f"\n--- Overall Retrieval Evaluation Analysis for Collection '{dynamic_collection_name}' ---")
                for dataset_name, results in all_evaluation_results_for_lang_combo.items():
                     # Check if 'source_hit_rate' exists before accessing
                     hit_rate = results.get('source_hit_rate', 'N/A')
                     if isinstance(hit_rate, float):
                         print(f"  Dataset: {dataset_name}, Source Hit Rate: {hit_rate:.2f}%")
                     else:
                         print(f"  Dataset: {dataset_name}, Source Hit Rate: {hit_rate}")


    # Analyze English datasets once after all loops
    print("\n--- Analysis of English Question Datasets Used ---")
    analyze_dataset_across_types(dataset_paths)

    print("\n--- RAG Pipeline Script Finished ---")
