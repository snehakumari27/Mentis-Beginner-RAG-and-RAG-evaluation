"""
rag_pipeline.py

Wires the local Chroma retriever (from vector_store.py) to a local Ollama
LLM using LangChain Expression Language (LCEL), forming a full
retrieve -> augment -> generate pipeline.

Usage:
    python rag_pipeline.py "What does the document say about X?"
    python rag_pipeline.py "What does the document say about X?" --model phi3
"""

from __future__ import annotations

import argparse
import logging
from typing import List

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from vector_store import get_retriever

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------- #
DEFAULT_MODEL = "qwen3:8b"

PROMPT_TEMPLATE = """You are a strict, highly accurate AI study assistant. 
You must answer the question using ONLY the information provided in the Context below. 
If the exact answer cannot be deduced from the Context, you must reply exactly: "I do not know based on the provided context."
Under no circumstances should you use outside knowledge or hallucinate information.

Context:
{context}

Question:
{question}

Answer:"""


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def format_docs(docs: List[Document]) -> str:
    """Concatenate retrieved chunks into a single context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(model_name: str = DEFAULT_MODEL):
    """
    Construct the full LCEL RAG chain: retriever -> prompt -> Ollama LLM -> string output.

    Args:
        model_name: Name of the local Ollama model to use (e.g. "llama3", "phi3").
            The model must already be pulled locally via `ollama pull <model_name>`.

    Returns:
        A tuple of (retriever, chain), where `chain` is a Runnable that accepts
        a raw query string and returns the generated answer as a string. The
        retriever is returned separately so callers can also fetch source
        documents/metadata for the same query.
    """
    retriever = get_retriever()

    llm = ChatOllama(model=model_name, temperature=0.2)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, chain


def answer_query(query: str, model_name: str = DEFAULT_MODEL) -> dict:
    """
    Run a single query end-to-end: retrieve relevant chunks, generate an
    answer grounded in those chunks, and return both.

    Args:
        query: The natural language question to ask.
        model_name: Name of the local Ollama model to use.

    Returns:
        A dict with keys:
            "answer": the generated answer string.
            "sources": list of metadata dicts, one per retrieved chunk.
    """
    retriever, chain = build_rag_chain(model_name=model_name)

    # Retrieve once and reuse the same chunks both for generation and for
    # reporting sources, so the sources always match what the LLM actually saw.
    retrieved_docs = retriever.invoke(query)
    context = format_docs(retrieved_docs)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOllama(model=model_name)
    generation_chain = prompt | llm | StrOutputParser()

    answer = generation_chain.invoke({"context": context, "question": query})

    sources = [doc.metadata for doc in retrieved_docs]
    context_texts = [doc.page_content for doc in retrieved_docs]

    return {"answer": answer, "sources": sources, "context_texts": context_texts}


# --------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the local RAG pipeline (Chroma + Ollama)."
    )
    parser.add_argument("query", type=str, help="The question to ask.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Local Ollama model to use (default: {DEFAULT_MODEL}). E.g. 'phi3'.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logger.info("Running query against Ollama model '%s'...", args.model)
    result = answer_query(args.query, model_name=args.model)

    print("\n=== ANSWER ===")
    print(result["answer"].strip())

    print("\n=== SOURCES ===")
    if not result["sources"]:
        print("No source chunks were retrieved.")
    else:
        for i, meta in enumerate(result["sources"], start=1):
            print(f"[{i}] {meta}")


if __name__ == "__main__":
    main()