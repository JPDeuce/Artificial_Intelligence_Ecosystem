"""Local RAG application: FAISS retrieval + cross-encoder rerank + Ollama (Qwen 3) generation.

Everything runs on this machine: no API keys, no .env file, no cloud services. The only
external dependency is a local Ollama server, which the app talks to over plain HTTP on
localhost.

Retrieval pipeline:
    document -> chunks -> Sentence Transformers embeddings -> FAISS IndexFlatL2
    question -> FAISS top_k candidates -> cross-encoder rerank -> top_m chunks
    top_m chunks + question -> Ollama chat API -> grounded answer

The stages are deliberately separate functions so each one can be tested or replaced on its
own. Section 4 of the reflection report explains why the chunk size, the reranking stage
and the prompt each matter, and Section 5 lists the bugs that were found and fixed while
building this.

Run it with:
    source .venv/bin/activate
    python rag_app.py
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
# Everything the instructor (or a future experiment) might want to change lives here.
# Changing chunk_size, chunk_overlap, top_k or top_m means restarting the app: the chunks,
# the embeddings and the FAISS index are all built once at startup.

# Size of each chunk, in characters. Chunks are the unit of retrieval, so this is the
# single most consequential parameter in the project. 500 characters was chosen by
# experiment (see Section 3 of the reflection report): 200 characters fragmented
# sentences, 1000 characters smears competing facts together and made the model answer
# incorrectly even when the right chunk was retrieved.
chunk_size = 500

# How much of each chunk is repeated at the start of the next one, in characters. The
# overlap stops a sentence that straddles a chunk boundary from being cut in half, which
# would leave two fragments that embed badly. 50 characters is roughly 10% of the chunk
# size; the experiment showed a 20% overlap mostly duplicated text in the prompt.
chunk_overlap = 50

# Bi-encoder used to embed both the document chunks and the incoming question. It is fast
# enough to embed the whole corpus at startup and produces 768-dimensional dense vectors.
embedding_model_name = "sentence-transformers/all-distilroberta-v1"

# Cross-encoder used to rerank the FAISS shortlist. It reads the question and a chunk
# together, so it is far more accurate than the bi-encoder but costs one forward pass per
# pair - which is why it only runs over the shortlist (top_k), not the whole corpus.
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many candidates FAISS returns, and how many of those survive reranking and are
# handed to the LLM. top_k is generous because the reranker is good at discarding the
# noise; top_m is small because the LLM's accuracy drops when the context gets large.
top_k = 20
top_m = 8

# The local LLM, served by Ollama. Any Ollama model can be substituted here; the app
# detects whether the new model supports "thinking" mode and adapts automatically.
llm_model = "qwen3:4b"

# Ollama's local HTTP endpoints. /api/chat is used for generation and /api/tags is used
# to read the model's advertised capabilities.
ollama_url = "http://localhost:11434/api/chat"
ollama_tags_url = "http://localhost:11434/api/tags"

# Qwen 3 reasons before it answers, and that reasoning is billed against the token
# budget. A small budget (the course example uses 500) is consumed entirely by the
# reasoning, which returns a truncated preamble and NO answer at all. 8000 tokens leaves
# room for the reasoning and the answer; retry_num_predict is the larger budget used for
# the one retry when the first attempt still comes back empty.
num_predict = 8000
retry_num_predict = 12000

# Low temperature: the task is extraction from a given context, not creative writing, so
# the model should prefer the wording in the context over its own priors.
temperature = 0.1

# Source document, resolved relative to this file so the script runs from any directory
# (and so a fresh clone works without editing anything).
document_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Selected_Document.txt"
)

# Generous timeout: a question that forces the model to reconcile two conflicting
# numbers can take a couple of minutes to reason and answer.
request_timeout = 600

# -----------------------------
# Quiet the noisy libraries
# -----------------------------
# Sentence Transformers and Transformers print model-loading and progress information that
# would otherwise be interleaved with the app's own output.
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# -----------------------------
# Prompts
# -----------------------------
# The system prompt is the last line of defence against hallucination: it is what makes
# the model answer from the retrieved context and refuse when the context is silent.
#
# Each clause exists because of a specific observed failure:
#   "using ONLY the context provided"      - stops the model answering from its own
#                                            training knowledge.
#   "Use the specific facts, names and      - pushes the model toward the exact figures
#    numbers from the context"                 in the source instead of vague summaries.
#   "even if the context does not use the    - an earlier, stricter version of this
#    exact wording of the question"            prompt made the model look for the literal
#                                            words of the question and refuse to answer
#                                            "Who is the main villian?" even though
#                                            five chunks described Ganondorf as the
#                                            antagonist. The instruction has to describe
#                                            SEMANTIC relevance, not string matching.
#   "Do not use outside knowledge"          - reinforces the grounding constraint.
#   "say exactly: 'I don't know based on    - gives the model one quotable refusal
#    the provided context.'"                   instead of letting it invent an answer.
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the context "
    "provided. Use the specific facts, names and numbers from the context. If the "
    "context contains information that answers the question, answer it even if the "
    "context does not use the exact wording of the question. Do not use outside "
    "knowledge. Only if the context has no relevant information at all, say exactly: "
    "'I don't know based on the provided context.'"
)

# The user prompt carries the retrieved evidence and the question. The layout matters:
# the context comes first and the "Answer:" cue comes last, so the model continues
# straight into the answer instead of restating the question. The context is joined
# with blank lines so the model can see where one chunk ends and the next begins.
USER_PROMPT_TEMPLATE = "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"


# -----------------------------
# Loading and splitting the document
# -----------------------------
def load_document(path=document_path):
    """Read the source document and return its text.

    Raises FileNotFoundError if Selected_Document.txt is missing (it is produced by
    text_extractor_web.py) and ValueError if it is empty, because an empty document
    would silently produce an index with nothing in it.
    """
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
    """Split the document text into overlapping chunks.

    RecursiveCharacterTextSplitter is used with the separator order
    ["\\n\\n", "\\n", " ", ""]: it tries to break on a paragraph break first, then a line
    break, then a space, and only falls back to a hard character cut for a paragraph
    that is longer than chunk_size. That keeps chunks aligned with the article's own
    structure instead of slicing mid-sentence wherever it happens to fit.
    """
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
    """Embed the chunks and store them in a FAISS IndexFlatL2 index.

    IndexFlatL2 is an exact, brute-force index: it stores the raw vectors and compares
    the query against every one of them. For a corpus this small (a few hundred chunks)
    that is the right choice - there is no quantization, no graph traversal and no
    missed nearest neighbours. Approximate indexes such as IVFFlat or HNSW only start to
    pay off at hundreds of thousands of vectors.

    Note: the course example normalises its vectors and uses IndexFlatIP (inner product).
    This app does not normalise, so the L2 distance carries a mild penalty for chunks
    with unusually large norms. With Sentence Transformer embeddings the norms are very
    similar, so it made no measurable difference, but normalising here would make the
    ranking exactly equivalent to cosine similarity.
    """
    print(f"Loading embedding model '{embedding_model_name}'...")
    embedder = SentenceTransformer(embedding_model_name)

    # The corpus is embedded once, here. Every question afterwards only needs the query
    # embedded, which is why a 110-chunk corpus starts instantly and answers in seconds.
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embedder.encode(
        chunks, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    # The index width is fixed by the encoder: 768 dimensions for all-distilroberta-v1.
    # A query vector of any other length would be rejected at search time.
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"FAISS index ready: {index.ntotal} chunks, dimension {dimension}.")
    return embedder, index


# -----------------------------
# Retrieval and reranking
# -----------------------------
def retrieve_chunks(embedder, index, chunks, question, k=top_k):
    """Return the k chunks closest to the question in embedding space.

    The question is embedded with the same encoder that embedded the corpus, so both
    live in the same 768-dimensional space. FAISS returns the k smallest squared
    Euclidean distances, which is the vector-space equivalent of "most semantically
    similar".

    The shortlist is deliberately larger than the number of chunks that eventually reach
    the model (k = top_k = 20 versus top_m = 8), because the reranker in the next stage
    is much better at telling similar passages apart than the bi-encoder is.
    """
    question_vector = np.asarray(
        embedder.encode([question], convert_to_numpy=True), dtype="float32"
    )

    # Clamp the request to the size of the corpus, so asking for 20 candidates from a
    # 4-chunk document returns 4 rather than failing.
    k = min(k, len(chunks))
    distances, indices = index.search(question_vector, k)

    candidates = []
    for position, chunk_index in enumerate(indices[0]):
        # FAISS pads with -1 when fewer than k results are available.
        if chunk_index == -1:
            continue
        candidates.append((chunks[chunk_index], float(distances[0][position])))
    return candidates


def _dedupe_preserve_order(items):
    """Remove duplicate chunks while keeping their original order.

    Overlapping chunks can produce near-identical passages. Feeding the same text to the
    model twice wastes context without adding evidence, so exact duplicates (compared
    with whitespace normalised) are dropped after reranking.
    """
    seen = set()
    unique = []
    for item in items:
        key = " ".join(item.split())
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def rerank_chunks(reranker, question, candidates, m=top_m):
    """Score (question, chunk) pairs with the cross-encoder and keep the top m.

    The bi-encoder scored every chunk independently, so it could not compare two chunks
    with each other. The cross-encoder reads the question and a chunk together, which
    lets it separate passages that are lexically almost identical but answer different
    parts of the question - for example "7.6 million copies worldwide" versus
    "7 million cartridges".
    """
    if not candidates:
        return []

    texts = [chunk for chunk, _ in candidates]
    scores = reranker.predict([(question, text) for text in texts])

    # Sort by descending relevance score.
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

    Qwen 3 reasons before it answers and Ollama only splits the two apart when the
    model is asked for thinking. Models without thinking support (llama, gemma, ...)
    reject the flag outright, so it is only sent when the manifest says the model can
    handle it. That keeps the app working when the model constant is changed.
    """
    try:
        response = requests.get(ollama_tags_url, timeout=30)
        response.raise_for_status()
        models = response.json().get("models", [])
    except (requests.exceptions.RequestException, ValueError):
        # If the capability list cannot be read, assume the conservative default.
        return False

    for entry in models:
        name = entry.get("name", "")
        # Ollama reports names like "qwen3:4b", "qwen3:4b-instruct" or "qwen3:4b:latest".
        if name == model or name == f"{model}:latest" or name.startswith(f"{model}:"):
            return "thinking" in (entry.get("capabilities") or [])
    return False


def _strip_reasoning(text):
    """Drop a reasoning preamble if the model left one in the answer content.

    With thinking mode on, Ollama puts the reasoning in message["thinking"] and the
    answer in message["content"], so this is normally a no-op. It is a safety net for
    the case where the model closes a <think> block itself and then restarts its
    analysis as plain text, which would otherwise leak a wall of reasoning into the
    answer.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        text = text.rsplit("<think>", 1)[1]
    return text.strip()


def _post_to_ollama(payload):
    """POST a chat payload to Ollama and return the decoded JSON response.

    The exception types are chosen so the caller can tell the user what actually went
    wrong: a stopped Ollama server, a slow model and a broken request are three
    different problems with three different fixes.
    """
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

    # A 404 here means the model name is not pulled yet ("ollama pull <model>").
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
    """Send the question and context to Ollama and return the generated answer.

    The context and the question are the only things the model sees, so the answer can
    only be grounded in what was retrieved.
    """
    # Nothing was retrieved, so there is nothing to ground an answer in. Refusing here
    # is the same behaviour the system prompt asks for when the context is irrelevant.
    if not context_chunks:
        return "I don't know based on the provided context."

    # Detect thinking support once per app run rather than once per question.
    if think is None:
        think = model_supports_thinking()

    context = "\n\n".join(context_chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                context=context, question=question
            ),
        },
    ]

    def request(budget):
        """Send one chat request with the given token budget."""
        return _post_to_ollama(
            {
                "model": llm_model,
                "messages": messages,
                "stream": False,
                # Ask Ollama to separate the reasoning from the answer when the model
                # supports it. This is what keeps the reasoning out of the answer.
                "think": think,
                "options": {"num_predict": budget, "temperature": temperature},
            }
        )

    data = request(num_predict)
    message = data.get("message") or {}
    content = (message.get("content") or "").strip()

    # An empty answer means the model spent its whole budget reasoning and never got to
    # the answer. One retry with a bigger budget recovers most of those cases.
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
    """Load the document, build the index once, then answer questions until exit."""
    # Fail early and clearly if the document is missing or empty - the interactive loop
    # below would otherwise loop forever over an empty corpus.
    try:
        text = load_document()
        chunks = split_document(text)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    # Both models are loaded once and reused for every question. The embedding model
    # indexes the corpus; the cross-encoder is only needed for reranking.
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
            # Ctrl+D or Ctrl+C should end the session cleanly rather than traceback.
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
            # A failed request should not kill the session: report it and keep asking.
            print(f"Error: {exc}")
            continue

        print(f"Answer: {answer}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
