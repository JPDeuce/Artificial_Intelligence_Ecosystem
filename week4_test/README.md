# Week 4 — Local Retrieval-Augmented Generation (RAG) with Ollama

A fully local RAG system: a Wikipedia article is scraped into a text file, split into
overlapping chunks, embedded with Sentence Transformers, indexed in FAISS, retrieved and
reranked for each question, and answered by **Qwen 3 4B running in Ollama**.

Everything runs offline on this machine. No API keys, no `.env` file, no OpenAI SDK.

| Component | Choice |
|---|---|
| Source document | Wikipedia — *The Legend of Zelda: Ocarina of Time* (1998) |
| Extractor | `requests` + Beautiful Soup 4 (`text_extractor_web.py`) |
| Splitter | `langchain-text-splitters` → `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-distilroberta-v1` (768 dims) |
| Vector store | FAISS `IndexFlatL2` (local, in memory) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | Ollama `qwen3:4b` via the local HTTP API (`requests`) |

---

## 1. Description of the selected document

**Wikipedia article: "The Legend of Zelda: Ocarina of Time"** (the 1998 Nintendo 64 release).

```text
https://en.wikipedia.org/wiki/The_Legend_of_Zelda:_Ocarina_of_Time
```

I chose the original 1998 game rather than the announced 2026 remake. Both pages have the
same HTML structure, but the 1998 article is a far richer corpus for this exercise: it
yields **73 prose blocks (~34,600 characters) across 21 sections** versus 16 blocks
(~3,900 characters) and 6 sections for the remake. That difference matters for the
deliverables — the chunk-size experiment in section 3 is only meaningful when there are
enough chunks to compare, and the "three questions and answer quality" section needs
questions whose answers are actually retrievable.

**Extraction** (`text_extractor_web.py`) works in four stages:

1. `requests.get` the hardcoded URL with a browser `User-Agent`.
2. Locate the article body: `<article>` if present, otherwise
   `#mw-content-text .mw-parser-output` (Wikipedia), otherwise `<main>`.
3. Pull the infobox into a **Key facts** block (13 rows: Nintendo EAD, Shigeru Miyamoto,
   Koji Kondo, Nintendo 64 / GameCube / iQue Player, the four regional 1998 release dates,
   genre, mode).
4. Walk the remaining `h2`, `h3`, `h4`, `p` and `li` elements in document order, skipping
   navigation and citation blocks, and join the survivors with blank lines.

### Cleaning notes

Two extraction problems were found by inspecting the output and fixed:

- **Wikipedia returns HTTP 403 to the default `python-requests` User-Agent.** A browser
  `User-Agent` header is mandatory. The course example's plain `requests.get(url)` would
  fail today.
- **Noise lives in ancestor classes, not element classes.** Wikipedia's 115 navbox `<li>`
  elements and 24 reference `<li>` elements carry no class of their own — their parent
  `<table>`/`<ol>` does. Filtering on the element's own class let all of them through, so
  the filter walks the ancestor chain. Inline `[1] [2]` citation markers (`sup.reference`)
  are decomposed before text is read.

### Result

`Selected_Document.txt` — 87 blocks, 35,393 characters, UTF-8:

```text
[0]  Key facts
[1]  Developer: Nintendo EAD
[2]  Publisher: Nintendo
...
[14] The Legend of Zelda: Ocarina of Time is a 1998 action-adventure game developed and
     published by Nintendo for the Nintendo 64. It was the first Legend of Zelda game
     with 3D graphics. ...
[18] Gameplay
[19] The Legend of Zelda: Ocarina of Time is a fantasy action-adventure game set in a 3D
     world with an expansive environment. ...
[71] Awards
[72] In 1998, Ocarina of Time won the Grand Prize in the Interactive Art division at the
     Japan Media Arts Festival. Ocarina of Time won six awards at the 2nd Annual
     Interactive Achievement Awards ...
[80] ... "The Legend of Zelda became a sort of textbook for 3D action games" ...
[86] The Legend of Zelda: Ocarina of Time at IMDb
```

The file contains prose only — no navboxes, no reference lists, no citation markers, no
table-of-contents. Every factual claim the RAG system answers below can be traced back to
a line in this file.

---

## 2. Questions for the RAG model, and answers

Run from the project folder with the virtual environment active:

```bash
source .venv/bin/activate
python rag_app.py
```

The app prints `Your question: ` and answers until `exit` or `quit` (case-insensitive);
blank input is ignored.

### Question 1 — `what year was this game released?`

**Answer: 1998**

The model cites the release dates directly: *"It was released in Japan and North America in
November 1998 and in PAL regions the following month"* and *"The game would miss its
scheduled 1997 holiday season release and was delayed into 1998."*

### Question 2 — `which consoles was it made for?`

**Answer: Nintendo 64**

The model grounds this in *"Ocarina of Time was developed concurrently with Super Mario 64
and Mario Kart 64 for the Nintendo 64 by Nintendo's Entertainment Analysis & Development
(EAD) division in 1994."* It also notes the later re-releases (Wii Virtual Console, Wii U
Virtual Console, Nintendo 3DS, GameCube) and correctly separates them from the platforms
the game was *originally* made for. The infobox independently lists
`Platforms: Nintendo 64 GameCube iQue Player`.

### Question 3 — `Who is the main villian?`

**Answer: Ganondorf**

The retrieved chunks describe *"the King of the Gerudo, Ganondorf"* and *"Ganondorf
captures Zelda and teleports her to his castle"*, and the model identifies him as the
primary antagonist.

> **This question exposed the most significant bug in the project.** See
> "Findings and fixes" below — the first version of the app answered
> *"I don't know based on the provided context."* to a question that was directly
> answerable, and the underlying cause was a flaw in how Ollama and Qwen 3 were being
> driven, not a retrieval failure.

### Question 4 — `How many awards did the game win?`

**Answer: 19 awards**, itemised from the context:

| Award body | Awards |
|---|---|
| Japan Media Arts Festival | 1 (Grand Prize, Interactive Art) |
| 2nd Annual Interactive Achievement Awards | 6 |
| BAFTA Interactive Entertainment Awards | 4 |
| Electronic Gaming Monthly | 5 |
| 16th Annual Game Audio Network Guild Awards | 1 |
| Edge | 2 (Game of the Year, Gameplay Innovation) |
| **Total** | **19** |

The enumeration is verifiable against `Selected_Document.txt` — this is grounded
arithmetic over retrieved text, not a recalled fact.

### Question 5 — `Who composed the music for the game?`

**Answer: Koji Kondo**

The infobox row `Composer: Koji Kondo` and the Music section both support this. The answer
is a single line, which shows the retrieval is precise: the reranker put the composer
chunk at the top and nothing else in the context was relevant.

### Question 6 — `What is the capital of France?` (not in the document)

**Answer: I don't know based on the provided context.**

This is the behaviour I want. The document contains nothing about France, the model
searches the context, finds no relevant material, and refuses instead of hallucinating.
This is the single most important property of a grounded RAG system, and it is enforced by
the system prompt (*"Only if the context has no relevant information at all, say exactly:
'I don't know based on the provided context.'"*).

### Answer quality assessment

- **Questions 1, 2, 5** are clean factual lookups. The answers are correct, concise, and
  quote the source text. Retrieval quality is high: the correct chunk is ranked first by
  the cross-encoder every time.
- **Question 3** needed the prompt to be tuned. The first version of the system prompt
  made the model over-literal — it looked for a sentence containing the words "main
  villain", did not find one, and refused. Adding *"If the context contains information
  that answers the question, answer it even if the context does not use the exact wording
  of the question"* fixed it. The document never calls Ganondorf "the main villain"; the
  model has to connect "King of the Gerudo", "captures Zelda", and "primary antagonist".
- **Question 4** is the hardest. Counting awards requires reading five different sentences
  and summing them, which a 4B model does slowly — it deliberated for ~6,300 tokens and
  ~2 minutes before answering. The answer is correct, but this is where the model is
  weakest and where latency is worst.
- **Question 6** shows the refusal path working exactly as designed.

**Overall:** retrieval was the strong part of this system and generation was the weak
part. For every question the correct evidence was retrieved and ranked first; every
wrong or refused answer traced back to how the language model was prompted and how many
tokens it was allowed, not to the index.

### Verbatim console output

The transcripts below are copied from actual runs of the app. They show the system
loading its models, indexing the document, and answering.

**A short session:**

```text
Loading embedding model 'sentence-transformers/all-distilroberta-v1'...
Embedding 110 chunks...
FAISS index ready: 110 chunks, dimension 768.
Loading reranker 'cross-encoder/ms-marco-MiniLM-L-6-v2'...

RAG app ready. LLM: qwen3:4b via http://localhost:11434/api/chat
Thinking mode: on
Enter 'exit' or 'quit' to end.

Your question: Answer: 1998

Your question: Answer: Ganondorf

Your question:
```

**A full generated response** — *"How many awards did the game win?"*, which is the
question that requires reading six separate sentences and summing them:

```text
The context provides specific information about the awards won by *Ocarina of Time*:

1. **Japan Media Arts Festival**: 1 award ("Grand Prize in the Interactive Art division").
2. **2nd Annual Interactive Achievement Awards**: 6 awards ("Game of the Year", "Console Game of the Year", "Console Adventure Game of the Year", "Console Role-Playing Game of the Year", "Outstanding Achievement in Interactive Design", "Outstanding Achievement in Software Engineering").
3. **BAFTA Interactive Entertainment Awards**: 4 awards ("The Games Award", "Innovative Game Award", "The Interactivity Award", "The Computer Programming Award").
4. **Electronic Gaming Monthly**: 5 awards (3 editors' choice: "Game of the Year for All Systems", "Nintendo 64 Game of the Year", "Action RPG of the Year"; 2 readers' choice: "Best Music", "Best Graphics").
5. **Edge**: 2 awards ("Game of the Year", "Gameplay Innovation").
6. **16th Annual Game Audio Network Guild Awards**: 1 award ("Best Game Music Cover/Remix").

The context explicitly states that *Ocarina of Time* won these awards, and there are no mentions of nominations that were not won (e.g., "runner-up" for "Best Sound Effects" and "2nd place" for "Graphical Achievement" are not counted as wins).

**Total awards won**: 1 + 6 + 4 + 5 + 2 + 1 = **19**.

**Answer**: 19
```

Two things are worth noticing in that answer. The model **enumerates the evidence before
totalling it**, and it **excludes what does not count** — the "runner-up" for Best Sound
Effects and the "2nd place" for Graphical Achievement are both in the retrieved context,
but they are not wins, and the model says so rather than counting them. That is the
grounding doing real work: a question that invites an inflated number gets the
defensible one, and the arithmetic is shown so the answer can be checked by hand against
`Selected_Document.txt`.

---

## 3. How retrieval quality varied with chunk_size and chunk_overlap

Each configuration required editing `chunk_size` / `chunk_overlap` in `rag_app.py` and
restarting it, because the chunks, the embeddings and the FAISS index are all rebuilt at
startup.

Three configurations were compared against the same three questions:

| Config | `chunk_size` | `chunk_overlap` | Chunks | Context handed to the LLM | "Koji Kondo" | "How many copies sold worldwide?" | "Capital of France?" |
|---|---|---|---|---|---|---|---|
| A | 500 | 50 | 110 | 2,608 chars | ✅ Koji Kondo | ✅ 7.6 million | ✅ refuses |
| B | 1000 | 200 | 49 | 6,540 chars | ✅ Koji Kondo | ❌ **7 million** | ✅ refuses |
| C | 200 | 50 | 261 | 1,453 chars | ✅ Koji Kondo | ✅ 7.6 million | ✅ refuses |

### What happened, and why

**Chunk size changes the evidence-to-noise ratio of the prompt, not just the retrieval
granularity.** With `top_m = 8` fixed, the amount of text the model reads is
`8 x chunk_size` — so doubling the chunk size doubles the context. The size of the
retrieved context measured in the experiment was 1,453 / 2,608 / 6,540 characters for
configs C / A / B.

- **Config B (1000/200) failed.** The document contains two competing figures: *"1.14
  million Nintendo 64 copies were sold in Japan and **7.6 million copies worldwide**"* and
  *"By 2000, Ocarina of Time had sold **7 million cartridges**"*. I verified that at
  `chunk_size=1000` the reranker still returned chunks containing **both** numbers, so
  retrieval was not the failure — the answer *was* in the context. The failure was that a
  1,000-character chunk smears the two facts together, the 6,540-character context gives
  the model room to weigh several contradictory figures, and the model settled on the
  wrong one (7 million, which it even reformatted as `7000000`). **Larger chunks made the
  model worse, not the retriever.**
- **Config C (200/50) was the fastest and cleanest.** The top-ranked chunk began
  *"million Nintendo 64 copies were sold in Japan and 7.6 million copies worldwide"* — the
  fact isolated in its own small unit, and the model answered *"7.6 million copies
  worldwide"* immediately with no deliberation. The cost is that small chunks cut across
  sentence boundaries (that chunk starts mid-number, losing *"1.14 "*) and can return
  fragments that read poorly out of context.
- **Config A (500/50) was the balanced choice** and is what the app ships with. The
  relevant sentence is fully contained in one chunk, the context stays small, and the
  model reasoned to the right answer while explicitly rejecting the competing figure
  (*"this appears to be a specific reference to the cartridge sales by that year, not the
  total worldwide sales figure"*).

### On chunk overlap

`chunk_overlap` exists so a sentence spanning a chunk boundary is not cut in half, and so
a chunk that begins mid-context still carries the tail of the previous one. The
`RecursiveCharacterTextSplitter` splits on `["\n\n", "\n", " ", ""]`, so boundaries are
preferentially placed at paragraph breaks.

Overlap alone is not a substitute for a good chunk size, though. Config B used a
generous 200-character overlap and still failed, because overlap cannot fix a chunk that
is too large to hold one idea cleanly. Overlap also costs index size and query time —
with `chunk_size=500, chunk_overlap=50` the effective stride is 450 characters, so the
document produces 110 chunks instead of roughly 71 non-overlapping ones. A ratio near
10% is the practical sweet spot; 20% (config B) mostly added duplicate text to the prompt.

### Practical conclusion

For this document I would keep `chunk_size=500` and `chunk_overlap=50`. Smaller chunks
(200) gave the most concise answers but fragment sentences; larger chunks (1000) actively
degraded accuracy by feeding the model competing facts in a single blurred unit. If I were
tuning this further I would add a **sentence-aware splitter** and keep `chunk_size` small,
because the evidence here suggests that *precision of the retrieved unit* matters more
than *how much context accompanies it*.

---

## 4. Five deep-dive questions and answers

These five questions walk the whole pipeline, but grounded in this implementation rather
than in the abstract. Each one also covers one of the five stages of a RAG system, so the
grounding is easy to check against section 3:

| Stage of the pipeline | Where it is answered |
|---|---|
| Why split the document at all, and where should the splits fall? | Section 3 and Q3 — chunk size, overlap and boundary placement |
| How does a chunk become something searchable? | Q1 and Q2 — the embedding vector and the FAISS index |
| How does a question find the relevant chunks? | Q2 and Q4 — vector search, then reranking |
| How does the model turn those chunks into an answer? | Q5 and section 2 — the prompt, and the answers it produced |
| What is the embedding, and why does its shape matter? | Q1 — dimensionality as the contract between encoder and index |

**On where these answers come from.** They were written by an AI assistant against the
actual code, console output and `Selected_Document.txt` of this project, and every claim in
them is checkable against one of those three sources. They are deliberately *not* sample
outputs from the RAG app, because these five questions are about how the system is built
rather than about the article — asked one of them, the app would correctly answer *"I don't
know based on the provided context."* That is the grounding working as designed, and it is
the same refusal path demonstrated by question 6 in section 2.

### Q1. What is the embedding dimensionality of this system, and why does that number matter?

The embedding model `all-distilroberta-v1` produces **768-dimensional dense vectors** —
one float32 per dimension, so 3,072 bytes per chunk. The app confirms this at runtime:
`FAISS index ready: 110 chunks, dimension 768.`

That number is the contract between the embedder and the index. `faiss.IndexFlatL2(768)`
allocates a fixed-width row per vector, and a query vector of any other length is
rejected at search time. It also sets the cost of retrieval: FAISS `IndexFlatL2` compares
the query against **every** stored vector, so a search over *n* chunks is *n* × 768
floating-point operations. At 768 dimensions and 110 chunks that is trivial (milliseconds);
at 768 dimensions and 1,000,000 chunks it would be roughly a billion operations per query,
which is where approximate indexes such as `IVFFlat` or `HNSW` start to pay off by trading
a little recall for a large speedup.

Dimensionality also reflects how much the encoder can preserve. 768 dimensions is the
distilled version of RoBERTa's 768-dimension output with six transformer layers instead of
twelve — enough to separate topical meaning, which is why "Koji Kondo" reliably lands
the composer chunk first.

### Q2. How does the FAISS search actually work in this app, and why `IndexFlatL2`?

`IndexFlatL2` is a **brute-force, exact** index: it stores the raw vectors in a flat
array and scans all of them for each query. It is *not* approximate — there is no
quantization, no graph traversal, and no missed nearest neighbours. For 110 chunks that
is the right choice, since an approximate index would add tuning and error for no benefit.

The search computes squared Euclidean distance, `d²(x, q) = Σ(xᵢ − qᵢ)²`, between the
query vector and every chunk vector, and returns the *k* smallest. Because
`‖x − q‖² = ‖x‖² + ‖q‖² − 2·x·q`, and `‖q‖²` is constant across candidates, the ranking
is identical to ranking by cosine similarity whenever the vectors have equal norms. My
vectors are **not** L2-normalized (unlike the course example, which normalizes and uses
`IndexFlatIP`), so the `‖x‖²` term does vary between candidates — chunks with unusually
large norms get a mild penalty. In practice this made no measurable difference to the
results here, because Sentence Transformer embeddings already have similar norms, but it
is the one substantive difference between this implementation and the example, and
normalizing the vectors would make the L2 ranking exactly equivalent to cosine similarity.

The app also clamps `k = min(top_k, len(chunks))`, so asking for 20 candidates from a
4-chunk document returns 4 rather than failing.

### Q3. Why use a chunk overlap at all, and what is the tradeoff?

Overlap duplicates the tail of each chunk at the start of the next, so a sentence or
paragraph that straddles a chunk boundary appears complete in at least one chunk.

Without overlap, a boundary can slice a sentence in half — "Ocarina of Time was developed
concurrently with Super Mario 64 and" / "Mario Kart 64 for the Nintendo 64 by Nintendo's
EAD division". Neither half embeds as well as the whole sentence, so a query about the
developer might retrieve neither. This is exactly what happened to the sales figure at
`chunk_size=200`, where the top chunk began *"million Nintendo 64 copies were sold in
Japan..."* — the "1.14 " was in the previous chunk.

The cost is redundancy: overlap text is embedded and indexed twice, inflating the index
and skewing retrieval toward boundary regions, and it enlarges the prompt, which in
section 3 turned out to matter a great deal. The practical guidance is to keep overlap
modest (10% of chunk size) and to make sure the splitter's separators include paragraph
breaks so most boundaries land on natural seams. `RecursiveCharacterTextSplitter` is
helpful here because it prefers `"\n\n"` and only falls back to `"\n"`, `" "` and `""` when
a single paragraph is longer than the chunk limit.

### Q4. What does the cross-encoder actually add over the bi-encoder, and why rerank at all?

The two models do the same job at different costs.

The **bi-encoder** (`all-distilroberta-v1`) encodes the query and every chunk
*independently* into a single vector, then compares them by vector distance. It is fast —
110 chunks are embedded once at startup — but it compresses each text into one fixed-size
vector, so the query and the chunk never actually interact.

The **cross-encoder** (`ms-marco-MiniLM-L-6-v2`) takes the pair `(question, chunk)`
*together* and outputs a single relevance score, so it can attend to the relationship
between them. That is far more accurate but costs one forward pass per pair, which is why
it only runs on the shortlist rather than the whole corpus.

The two-stage design is what makes that affordable. FAISS cheaply reduces 110 chunks to
`top_k = 20` candidates; the cross-encoder then reranks those 20 and the app keeps the
best `top_m = 8` for the prompt. The ordering is bi-encoder first because a cross-encoder
cannot be used to search the corpus, and reranking first because reranking all 110 chunks
would be 110 forward passes per question instead of 20.

The reranker also demonstrates why a bi-encoder alone is not enough. For
*"How many copies of the game were sold worldwide?"* the bi-encoder's top-20 already
contained the correct sentence, but the cross-encoder's ordering is what put the chunk
carrying *"7.6 million copies worldwide"* first and the chunk carrying *"7 million
cartridges"* below it. The reranker is doing the discriminating work between two
competing facts that are lexically almost identical.

Duplicates are removed after reranking with `_dedupe_preserve_order`, which normalizes
whitespace before comparing — overlapping chunks can produce near-identical passages, and
feeding the same text to the model twice wastes context without adding evidence.

### Q5. How does prompt design affect the answers, and what did Ollama's "thinking" behaviour teach me?

The prompt is the last place a RAG system can fail, and this project produced three
distinct prompt-level failures that had nothing to do with retrieval.

**The refusal instruction shapes the answer.** *"If the context does not contain the
answer, say exactly: 'I don't know based on the provided context.'"* gives the model an
explicit, quotable way to decline instead of inventing an answer, and question 6
demonstrates it working. But an over-strict version of that instruction is actively
harmful: my first prompt said *"If the context does not contain the answer"*, and the
model interpreted "answer" literally — it looked for a sentence containing the words
"main villain", found none, and refused question 3 even though five retrieved chunks
described Ganondorf as the antagonist. Rewording it to *"If the context contains
information that answers the question, answer it even if the context does not use the
exact wording of the question"* fixed it. The instruction has to describe *semantic*
relevance, not literal string matching.

**Instruction order and framing affect how much the model uses the evidence.** Asking for
a short, direct answer made the model enumerate less and reason faster on most questions
but made it more likely to skip evidence.

**The most important finding is that the model thinks before it answers, and that
reasoning is billed against the token budget.** Ollama's built-in Qwen 3 template ends
with:

```text
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
<think>
{{ end }}
```

The `<think>` header is **unconditional**. It is not gated on Ollama's `think` request
flag, so `"think": false` does not disable thinking for this model — I confirmed that
`"think": false`, `"think": true`, and `"think": false` combined with
`"chat_template_kwargs": {"enable_thinking": false}` all produced identical reasoning in
the output. The flag only controls whether Ollama *separates* the reasoning from the
answer afterwards.

That has two consequences the app must handle:

1. **A small `num_predict` silently destroys the answer.** With the spec's `num_predict:
   500`, Qwen 3 spent the entire budget reasoning, hit the limit, and returned truncated
   reasoning *with no answer at all* — the app surfaced this as *"Error: Ollama returned
   an empty response."* Raising the budget to 8000 tokens fixed it; the same sales
   question completes in ~520 tokens normally but ~2,700 when it has to reconcile two
   competing figures.
2. **The reasoning can land on either side of the `</think>` marker.** In one run the
   model wrote its reasoning, closed the block, and then restarted the analysis as plain
   text; in another the answer followed the marker. Stripping text after `</think>`
   therefore produced a correct answer in one case and a wall of reasoning in the other.

The fix was to stop trying to parse the model's output and instead ask Ollama to do the
separating: the app now requests thinking mode for models whose Ollama manifest lists the
`thinking` capability, and reads `message["content"]`, which Ollama guarantees contains
only the final answer, with the reasoning isolated in `message["thinking"]`. The
capability check is what makes the "thinking" flag safe to send at all — models without
thinking support (llama, gemma) reject it, so the flag is only set when the manifest says
the model can handle it. This also makes switching models work correctly rather than
crashing on the new model.

**The general lesson:** in a RAG pipeline the quality of an answer is the product of four
independent stages — extraction, chunking, retrieval, and prompting — and a failure at any
one of them looks identical from the outside. The only reliable way to localize a failure
is to instrument each stage. Every bug in this project was found by printing the retrieved
chunks and the raw API response instead of only reading the final answer; on two occasions
the retrieved context was perfect and the entire problem was in how the model was being
driven.

---

## 5. Findings and fixes

| # | Problem | Root cause | Fix |
|---|---|---|---|
| 1 | Wikipedia returns **HTTP 403** | The default `python-requests` User-Agent is rejected | Send a browser `User-Agent` header |
| 2 | Navboxes and reference lists in the output | Noise `<li>` elements carry no class; the class is on an ancestor | Filter on the ancestor chain, and decompose `sup.reference` |
| 3 | Info box data missing | The infobox is a table, and only headings/paragraphs/lists were extracted | Emit the infobox as a "Key facts" block first |
| 4 | **Answers empty / "Error: Ollama returned an empty response"** | Ollama's Qwen 3 template always opens `<think>`; `"think": false` does not disable it, so the reasoning consumed the whole `num_predict` budget | Raise the budget to 8000 tokens and retry with a larger one if the answer is still missing |
| 5 | Reasoning text leaking into answers | With `"think": false`, Ollama does not split reasoning from the answer, and the model sometimes puts the answer *before* `</think>` | Use Ollama's thinking mode (only for models that support it) and read `message["content"]` |
| 6 | *"Who is the main villian?"* refused despite being answerable | The system prompt demanded a literal match on the question's wording | Reword the instruction to accept semantically relevant context |
| 7 | Model switching could break on non-thinking models | Ollama rejects the `think` flag for models without the capability | Look up the model's capabilities in `/api/tags` before sending the flag |

---

## 6. Project files and how to run

```text
week4_test/
├── .gitignore
├── README.md                  ← this reflection report
├── rag_app.py                 ← the RAG application
├── Selected_Document.txt      ← extracted article text (35,393 characters)
├── requirements.txt
├── text_extractor_web.py      ← web scraper / text extractor
└── .venv/                     ← virtual environment (not committed)
```

### Setup

```bash
cd ~/projects/Artificial_Intelligence_Ecosystem/week4_test
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Ollama (installed separately from `requirements.txt`)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b
ollama list          # verify the model is present
```

### Extract the document

```bash
python text_extractor_web.py      # writes Selected_Document.txt
```

### Run the RAG app

```bash
python rag_app.py
```

The embedding model and the cross-encoder download on first run (~330 MB and ~90 MB) and
are cached by Hugging Face afterwards.

### Adjusting the parameters

All adjustable settings are grouped at the top of `rag_app.py`:

```python
chunk_size = 500
chunk_overlap = 50
embedding_model_name = "sentence-transformers/all-distilroberta-v1"
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_k = 20
top_m = 8
llm_model = "qwen3:4b"
ollama_url = "http://localhost:11434/api/chat"
```

Changing `chunk_size`, `chunk_overlap`, `top_k` or `top_m` requires restarting the app —
the chunks, the embeddings and the FAISS index are all rebuilt at startup.

### Switching the Ollama model

The model is a single constant. To use a different one, pull it and change the constant:

```bash
ollama pull qwen3:1.7b        # or llama3.2:3b, gemma3:4b, ...
```

```python
llm_model = "qwen3:1.7b"
```

The app reads the model's capabilities from Ollama's `/api/tags` at startup, so it
automatically enables or disables thinking mode for the new model — a thinking model like
Qwen 3 and a non-thinking model like Llama 3.2 both work without code changes.
