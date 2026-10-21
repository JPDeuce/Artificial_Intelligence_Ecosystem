"""Local RAG application: FAISS retrieval + cross-encoder rerank + Ollama (Qwen 3) generation.

Retrieval pipeline:
    document -> chunks -> Sentence Transformers embeddings -> FAISS IndexFlatL2
    question -> FAISS top_k candidates -> cross-encoder rerank -> top_m chunks
    top_m chunks + question -> Ollama chat API -> grounded answer
"""

import logging
import os
import sys
import warnings

import faiss
import numpy as np
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import logging as transformers_logging

# -----------------------------
# Parameters
# -----------------------------
chunk_size = 500
chunk_overlap = 50
embedding_model_name = "sentence-transformers/all-distilroberta-v1"
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_k = 20
top_m = 8
llm_model = "qwen3:4b"
ollama_url = "http://localhost:11434/api/chat"
ollama_tags_url = "http://localhost:11434/api/tags"
# Qwen 3 reasons before it answers, and the reasoning is billed against the
# token budget, so the budget has to be large enough to finish the reasoning
# and still leave room for the answer.
num_predict = 8000
retry_num_predict = 12000
temperature = 0.1

# Source document, resolved relative to this file so the script runs from any directory.
document_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Selected_Document.txt"
)
request_timeout = 600

# -----------------------------
# Quiet the noisy libraries
# -----------------------------
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the context "
    "provided. Use the specific facts, names and numbers from the context. If the "
    "context contains information that answers the question, answer it even if the "
    "context does not use the exact wording of the question. Do not use outside "
    "knowledge. Only if the context has no relevant information at all, say exactly: "
    "'I don't know based on the provided context.'"
)


# -----------------------------
# Loading and splitting the document
# -----------------------------
def load_document(path=document_path):
    """Read the source document and return its text."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Source document not found: {path}. Run text_extractor_web.py first."
        )

    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    if not text.strip():
        raise ValueError(f"Source document is empty: {path}")

    return text


def split_document(text):
    """Split the document text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = [chunk for chunk in splitter.split_text(text) if chunk.strip()]

    if not chunks:
        raise ValueError("No text chunks were created from the source document.")

    return chunks


# -----------------------------
# Building the FAISS index
# -----------------------------
def build_index(chunks):
    """Embed the chunks and store them in a FAISS IndexFlatL2 index."""
    print(f"Loading embedding model '{embedding_model_name}'...")
    embedder = SentenceTransformer(embedding_model_name)

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embedder.encode(
        chunks, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"FAISS index ready: {index.ntotal} chunks, dimension {dimension}.")
    return embedder, index


# -----------------------------
# Retrieval and reranking
# -----------------------------
def retrieve_chunks(embedder, index, chunks, question, k=top_k):
    """Return the k chunks closest to the question in embedding space."""
    question_vector = np.asarray(
        embedder.encode([question], convert_to_numpy=True), dtype="float32"
    )

    k = min(k, len(chunks))
    distances, indices = index.search(question_vector, k)

    candidates = []
    for position, chunk_index in enumerate(indices[0]):
        if chunk_index == -1:
            continue
        candidates.append((chunks[chunk_index], float(distances[0][position])))
    return candidates


def _dedupe_preserve_order(items):
    """Remove duplicate chunks while keeping their original order."""
    seen = set()
    unique = []
    for item in items:
        key = " ".join(item.split())
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def rerank_chunks(reranker, question, candidates, m=top_m):
    """Score (question, chunk) pairs with the cross-encoder and keep the top m."""
    if not candidates:
        return []

    texts = [chunk for chunk, _ in candidates]
    scores = reranker.predict([(question, text) for text in texts])

    ranked = sorted(
        zip(texts, scores), key=lambda pair: float(pair[1]), reverse=True
    )
    best = [text for text, _ in ranked[: min(m, len(ranked))]]
    return _dedupe_preserve_order(best)


# -----------------------------
# Ollama helpers
# -----------------------------
def model_supports_thinking(model=llm_model):
    """Ask Ollama whether the model can separate reasoning from the answer.

    Qwen 3 reasons before it answers and Ollama only splits the two apart when
    the model is asked for thinking. Models without thinking support (llama,
    gemma, ...) reject the flag, so it is only sent when it is supported.
    """
    try:
        response = requests.get(ollama_tags_url, timeout=30)
        response.raise_for_status()
        models = response.json().get("models", [])
    except (requests.exceptions.RequestException, ValueError):
        return False

    for entry in models:
        name = entry.get("name", "")
        if name == model or name == f"{model}:latest" or name.startswith(f"{model}:"):
            return "thinking" in (entry.get("capabilities") or [])
    return False


def _strip_reasoning(text):
    """Drop a reasoning preamble if the model left one in the answer content."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        text = text.rsplit("<think>", 1)[1]
    return text.strip()


def _post_to_ollama(payload):
    """POST a chat payload to Ollama and return the decoded JSON response."""
    try:
        response = requests.post(ollama_url, json=payload, timeout=request_timeout)
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(
            f"Could not reach Ollama at {ollama_url}. Is 'ollama serve' running?"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise TimeoutError(
            f"Ollama did not respond within {request_timeout} seconds."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Request to Ollama failed: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Ollama returned a response that was not valid JSON."
        ) from exc


def generate_answer(question, context_chunks, think=None):
    """Send the question and context to Ollama and return the generated answer."""
    if not context_chunks:
        return "I don't know based on the provided context."

    if think is None:
        think = model_supports_thinking()

    context = "\n\n".join(context_chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
        },
    ]

    def request(budget):
        return _post_to_ollama(
            {
                "model": llm_model,
                "messages": messages,
                "stream": False,
                "think": think,
                "options": {"num_predict": budget, "temperature": temperature},
            }
        )

    data = request(num_predict)
    message = data.get("message") or {}
    content = (message.get("content") or "").strip()

    # The model spent its whole budget reasoning and never got to the answer.
    if not content:
        print("  (ran out of tokens while reasoning, retrying with a larger budget)")
        data = request(retry_num_predict)
        message = data.get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            raise RuntimeError(
                "Ollama returned an empty response. The model used its whole "
                "token budget reasoning and never produced an answer."
            )

    return _strip_reasoning(content)


def answer_question(embedder, index, reranker, chunks, question, think=None):
    """Full pipeline: retrieve, rerank, generate."""
    candidates = retrieve_chunks(embedder, index, chunks, question, k=top_k)
    context_chunks = rerank_chunks(reranker, question, candidates, m=top_m)
    return generate_answer(question, context_chunks, think=think)


# -----------------------------
# Interactive loop
# -----------------------------
def main():
    try:
        text = load_document()
        chunks = split_document(text)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    embedder, index = build_index(chunks)

    print(f"Loading reranker '{reranker_model_name}'...")
    reranker = CrossEncoder(reranker_model_name)

    think = model_supports_thinking()
    print(f"\nRAG app ready. LLM: {llm_model} via {ollama_url}")
    print(f"Thinking mode: {'on' if think else 'off'}")
    print("Enter 'exit' or 'quit' to end.\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        try:
            answer = answer_question(
                embedder, index, reranker, chunks, question, think=think
            )
        except (ConnectionError, TimeoutError, RuntimeError) as exc:
            print(f"Error: {exc}")
            continue

        print(f"Answer: {answer}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
