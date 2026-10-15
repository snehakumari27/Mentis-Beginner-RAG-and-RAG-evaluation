"""
vector_store.py

Chunking + vector storage module for the RAG pipeline.

Takes LangChain `Document` objects (e.g. produced by `ingest.py`) and:
    1. Splits them into overlapping chunks via `RecursiveCharacterTextSplitter`.
    2. Embeds each chunk locally using a HuggingFace sentence-transformer
       model (CPU-forced, no external API calls / no API key required).
    3. Persists everything to a local on-disk Chroma vector store.

Exposes two main entry points:
    - add_documents_to_db(docs): chunk + embed + persist a list of Documents.
    - get_retriever(): load the persisted store and return a retriever.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------- #
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "rag_documents"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

EMBEDDING_MODEL_NAME = "nomic-embed-text"

# --------------------------------------------------------------------- #
# Shared singletons
# --------------------------------------------------------------------- #
_embedding_function: Optional[OllamaEmbeddings] = None

def _get_embedding_function() -> OllamaEmbeddings:
    """Lazily initialize and cache the Ollama embedding model."""
    global _embedding_function
    if _embedding_function is None:
        logger.info("Loading embedding model '%s' via Ollama...", EMBEDDING_MODEL_NAME)
        _embedding_function = OllamaEmbeddings(
            model=EMBEDDING_MODEL_NAME
        )
    return _embedding_function


def _get_text_splitter() -> SemanticChunker:
    """Build the semantic chunker using the Ollama embedding model."""
    # SemanticChunker requires the embedding model to determine breakpoints
    return SemanticChunker(_get_embedding_function())

def _get_vector_store() -> Chroma:
    """
    Load (or create, if it doesn't exist yet) the on-disk Chroma collection
    using the shared embedding function.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embedding_function(),
        persist_directory=PERSIST_DIRECTORY,
    )


# --------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------- #
def add_documents_to_db(docs: List[Document]) -> int:
    """
    Chunk the given Documents and persist their embeddings to the local
    Chroma vector store.

    Args:
        docs: A list of LangChain `Document` objects (e.g. from
            `DocumentIngestor` in ingest.py). Each Document's existing
            metadata (source, title, page/slide, etc.) is preserved on
            every chunk derived from it.

    Returns:
        The number of chunks that were added to the store.
    """
    if not docs:
        logger.warning("add_documents_to_db called with an empty document list; nothing to do.")
        return 0

    splitter = _get_text_splitter()
    chunks = splitter.split_documents(docs)

    if not chunks:
        logger.warning("Text splitting produced zero chunks from %d input document(s).", len(docs))
        return 0

    vector_store = _get_vector_store()
    vector_store.add_documents(chunks)
    
    # NOTE: vector_store.persist() was removed here because langchain-chroma 
    # now saves to disk automatically when you add documents!

    logger.info(
        "Added %d chunk(s) from %d source document(s) to Chroma at '%s'.",
        len(chunks),
        len(docs),
        PERSIST_DIRECTORY,
    )
    return len(chunks)


def get_retriever(search_kwargs: Optional[dict] = None):
    vector_store = _get_vector_store()
    # Upgrade to MMR for better context diversity
    kwargs = search_kwargs or {
        "k": 5, 
        "fetch_k": 20
    }
    return vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs=kwargs
    )


if __name__ == "__main__":
    # Minimal smoke-test / usage example.
    #
    # from ingest import DocumentIngestor
    # ingestor = DocumentIngestor()
    # docs = ingestor.ingest_pdf("example.pdf")
    # add_documents_to_db(docs)
    #
    # retriever = get_retriever()
    # results = retriever.invoke("What is this document about?")
    # for r in results:
    #     print(r.metadata, "->", r.page_content[:80])
    pass