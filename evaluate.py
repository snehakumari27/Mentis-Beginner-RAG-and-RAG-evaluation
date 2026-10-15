"""
evaluate.py

Evaluates the local RAG pipeline using native LangSmith evaluation.
(Replaces ragas to avoid dependency conflicts)

- Judge LLM: Google Gemini (`ChatGoogleGenerativeAI`), via GOOGLE_API_KEY.
- Tracing & Eval: LangSmith (via LANGCHAIN_API_KEY).

Usage:
    python evaluate.py
"""

from __future__ import annotations

import logging
import os
import time

from dotenv import load_dotenv
from langsmith import Client, evaluate
from langsmith.schemas import Run, Example
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, wait_exponential, stop_after_attempt

from rag_pipeline import answer_query, DEFAULT_MODEL

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Check Env Vars
# --------------------------------------------------------------------- #
def _check_env_vars() -> None:
    if not os.getenv("LANGCHAIN_API_KEY"):
        logger.warning("LANGCHAIN_API_KEY is missing. Evaluation results will not be uploaded to LangSmith.")
    
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is not set. Add it to your .env file to use Gemini as the judge.")


# --------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------- #
EVAL_DATASET =  [
    {
        "inputs": {"question": "What are the three stages of training a Large Language Model like ChatGPT, and what is the primary goal of the first stage?"},
        "outputs": {"ground_truth": "The three stages of training are pretraining, instruction tuning (or Supervised Fine-Tuning), and Reinforcement Learning from Human Feedback (RLHF). The primary goal of the first stage, pretraining, is self-supervised learning on massive web-scale corpora to perform causal language modeling (predicting the next token) to build general linguistic capability."}
    },
    {
        "inputs": {"question": "Why do modern LLMs use subword tokenization methods like Byte Pair Encoding (BPE) instead of purely word-based or character-based tokenizers?"},
        "outputs": {"ground_truth": "Word-based tokenizers suffer from an excessively large vocabulary and cannot handle Out-Of-Vocabulary (OOV) words, while character-based tokenizers lead to extremely long sequence lengths and lack semantic information. Subword tokenization (such as BPE) starts with characters and iteratively merges the most frequent adjacent pairs, balancing sequence length and vocabulary size."}
    },
    {
        "inputs": {"question": "What is the limitation of One-Hot Encoding for word representation in Natural Language Processing, and how do Word Embeddings resolve it?"},
        "outputs": {"ground_truth": "One-Hot Encoding produces sparse, high-dimensional vectors that are orthogonal to each other, meaning they capture no semantic or contextual relationship between words. Word Embeddings resolve this by mapping tokens into a dense, low-dimensional continuous vector space where geometric closeness represents semantic similarity."}
    },
    {
        "inputs": {"question": "How does Positional Encoding help the Transformer architecture process sequential text data?"},
        "outputs": {"ground_truth": "Because self-attention processes all tokens in parallel, the Transformer architecture is permutation-invariant and has no built-in sense of word order. Positional Encoding solves this by adding unique sine and cosine wave values of varying frequencies directly to the word embeddings, injecting sequence order information into the inputs."}
    },
    {
        "inputs": {"question": "Explain the roles of the Query, Key, and Value vectors in the Transformer's self-attention mechanism."},
        "outputs": {"ground_truth": "In self-attention, the Query vector represents what a specific token is searching for in the text, the Key vector represents the identity and characteristics that the token has to offer, and the Value vector contains the actual semantic content of that token. The attention score is calculated by matching Queries with Keys to weight and sum the corresponding Values."}
    },
    {
        "inputs": {"question": "Why are raw attention scores divided by the scaling factor sqrt(d_k) in the scaled dot-product attention formula?"},
        "outputs": {"ground_truth": "For large head dimensions (d_k), the dot products of Queries and Keys can grow extremely large in magnitude. This pushes the softmax function into regions with very small gradients (vanishing gradients), which halts model training; dividing by the scaling factor sqrt(d_k) stabilizes the variance of these dot products."}
    },
    {
        "inputs": {"question": "What is the Chinchilla scaling law, and what does it tell us about optimizing Large Language Model pretraining?"},
        "outputs": {"ground_truth": "The Chinchilla scaling law is an empirical guideline establishing that for optimal downstream performance within a fixed compute budget, a model's parameter size (N) and the size of its training dataset in tokens (D) should be scaled in equal, proportional measures, rather than simply scaling parameters on static data."}
    },
    {
        "inputs": {"question": "How does the Temperature parameter (T) influence the token generation of a Large Language Model during inference?"},
        "outputs": {"ground_truth": "Temperature scales the raw logits (z / T) before softmax. A low temperature (T -> 0) concentrates probability on the highest-scoring token, making generation deterministic and logical but repetitive; a high temperature (T > 1.0) flattens the probability distribution, allowing lower-scoring tokens to be selected, which increases creativity but also raises the rate of hallucinations."}
    },
    {
        "inputs": {"question": "How does Retrieval-Augmented Generation (RAG) address the limitations of static Large Language Models?"},
        "outputs": {"ground_truth": "RAG overcomes model knowledge cutoffs, hallucinations, and lack of access to private data by chunking and embedding external documents into a vector database. At inference, it retrieves the top-K relevant chunks based on query similarity and injects them as grounding context into the prompt, enabling the LLM to write highly accurate, verified answers."}
    },
    {
        "inputs": {"question": "What is the performance bottleneck in autoregressive decoding during LLM inference, and how does KV Caching mitigate it?"},
        "outputs": {"ground_truth": "The bottleneck is that generating each new token requires multiplying the current Query with the Key and Value matrices of all previous tokens. Redundantly recomputing these past Key and Value matrices at every step leads to quadratic complexity (O(N^2)) and high latency; KV Caching mitigates this by saving past Key and Value matrices in GPU memory, meaning the model only computes projections for the single new token at each step (O(N) complexity)."}
    },
    {
        "inputs": {"question": "What is the difference between Multi-Head Attention (MHA), Multi-Query Attention (MQA), and Grouped-Query Attention (GQA)?"},
        "outputs": {"ground_truth": "In MHA, each attention head has its own Key, Value, and Query matrices, which results in a heavy memory footprint for KV caching. MQA dramatically reduces memory usage by having all Query heads share a single Key and Value head but sacrifices some performance. GQA acts as a compromise, dividing Query heads into groups where each group shares a single Key and Value head, recovering most of MHA's performance with most of MQA's speed and memory efficiency."}
    }
]

# --------------------------------------------------------------------- #
# Target Function
# --------------------------------------------------------------------- #
def target_pipeline(inputs: dict) -> dict:
    """This function connects LangSmith's dataset loop to your local Ollama pipeline."""
    question = inputs["question"]
    result = answer_query(question, model_name=DEFAULT_MODEL)
    return {
        "answer": result["answer"],
        "context_texts": result["context_texts"]
    }


# --------------------------------------------------------------------- #
# Rate-Limited API Helper
# --------------------------------------------------------------------- #
@retry(wait=wait_exponential(multiplier=2, min=4, max=30), stop=stop_after_attempt(5))
def safe_gemini_invoke(llm, prompt: str) -> str:
    """
    Wraps the Gemini API call with a mandatory sleep to prevent RPM limits, 
    and exponential backoff to handle 429 Too Many Requests errors.
    """
    time.sleep(5)  # Pace to max 12 requests per minute
    raw_content = llm.invoke(prompt).content
    if isinstance(raw_content, list):
        return "".join(str(item) for item in raw_content)
    return str(raw_content)


# --------------------------------------------------------------------- #
# Custom Gemini Evaluators
# --------------------------------------------------------------------- #
def correctness_evaluator(run: Run, example: Example) -> dict:
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    
    question = example.inputs["question"]
    expected = example.outputs["ground_truth"]
    prediction = run.outputs["answer"]
    
    prompt = f"""You are an expert grader. Compare the predicted answer against the expected ground truth answer.

Question: {question}
Expected Answer: {expected}
Predicted Answer: {prediction}

Does the predicted answer fundamentally contain the correct information presented in the expected answer? 
Respond with only a single word: YES or NO.
"""
    
    score = 0.0
    try:
        response_text = safe_gemini_invoke(llm, prompt)
        response = response_text.strip().upper()
        score = 1.0 if "YES" in response else 0.0
    except Exception as e:
        logger.error(f"Error during correctness evaluation: {e}")
        
    return {"key": "correctness", "score": score}


def faithfulness_evaluator(run: Run, example: Example) -> dict:
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    
    prediction = run.outputs["answer"]
    contexts = run.outputs.get("context_texts", [])
    context_str = "\n\n".join(contexts)
    
    prompt = f"""You are an expert grader evaluating AI hallucination.
    
Retrieved Context: 
{context_str}

Generated Answer: 
{prediction}

Is the generated answer entirely supported by the retrieved context? It should not contain outside knowledge.
Respond with only a single word: YES or NO.
"""

    score = 0.0
    try:
        response_text = safe_gemini_invoke(llm, prompt)
        response = response_text.strip().upper()
        score = 1.0 if "YES" in response else 0.0
    except Exception as e:
        logger.error(f"Error during faithfulness evaluation: {e}")
        
    return {"key": "faithfulness", "score": score}


# --------------------------------------------------------------------- #
# Main Execution
# --------------------------------------------------------------------- #
def main() -> None:
    _check_env_vars()
    
    logger.info("Starting native LangSmith evaluation...")
    logger.info(f"Targeting local model: {DEFAULT_MODEL}")
    logger.info("Using Gemini (gemini-3.6-flash) as the Judge with Rate Limiting enabled.")

    client = Client()
    
    dataset_name = "Mentis_Eval"

    dataset_exists = False
    try:
        client.read_dataset(dataset_name=dataset_name)
        dataset_exists = True
        logger.info(f"Found existing dataset '{dataset_name}' in LangSmith.")
    except Exception:
        pass

    if not dataset_exists:
        logger.info(f"Creating new dataset '{dataset_name}' in LangSmith...")
        dataset = client.create_dataset(dataset_name=dataset_name, description="Dataset for evaluating local RAG")
        client.create_examples(
            inputs=[item["inputs"] for item in EVAL_DATASET],
            outputs=[item["outputs"] for item in EVAL_DATASET],
            dataset_id=dataset.id,
        )

    experiment_results = evaluate(
        target_pipeline,
        data=dataset_name,
        evaluators=[correctness_evaluator, faithfulness_evaluator],
        experiment_prefix="Mentis_Evaluation",
        max_concurrency=1, 
    )
    
    print("\n=== EVALUATION COMPLETE ===")
    print("Check your LangSmith dashboard to see the detailed traces and scores!")


if __name__ == "__main__":
    main()