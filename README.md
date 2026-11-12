# Documentation claim auditor

An offline auditor comparing documentation promises with Python call patterns.
Source is parsed without execution. No network access or telemetry is used.

```sh
embedding-claim-audit audit . --json
python3 -m embedding_claim_audit audit . --fail-on medium --ignore vendor/
embedding-claim-audit --version
```

The root README and Markdown files below a docs component are checked.
Fenced blocks are skipped; inline code counts. Tests supply separate evidence.
Exit status is 0 below the selected threshold, 2 at or above it, and 1 for
usage or I/O errors. The default threshold is high. Ignore prefixes are
relative paths and may be repeated. JSON includes verdict and findings;
verbose output includes evidence paths.

```text
documented-embedding-claim-false HIGH
 document with only test evidence: MEDIUM
documented-no-embedding-claim-false HIGH
```

## Differentiation

The following tools cover adjacent operations:

```text
semantic-search-claim-audit: retrieval alone; an encode step without a
similarity comparison or index query is not evidence there, and is exactly
what embedding-claim-audit checks.
retrieval-augmented-generation-claim-audit: retrieve-then-generate,
no standalone generation step.
document-question-answering-claim-audit: pipeline instance call with no
generation step.
text-classification-claim-audit: labels, no vectors.
tokenization-claim-audit: splitting text, no vectors.
```

## Honest limitations

The matching hazards are reproduced below:

```text
- The bare words "embedding" and "embeddings" are deliberately
  EMBEDDING phrases: in Python project docs they almost always
  name vector generation from text. A doc that mentions embedding a
  video, an embedded device, or "embedding" as a generic verb is a
  false positive; the same accepted shape as other families' short
  phrases.
- "word embedding" and "word embeddings" are deliberately phrases:
  they can also describe Word2Vec / gensim static word vectors,
  which this tool does NOT detect as evidence. The tool flags the
  claim; only the code decides, and a Word2Vec-only codebase fails
  the check by design — say so in the README.
- "vector representation", "vector representations", "dense
  representation", and "dense representations" are deliberately
  phrases: they can describe any dense vector, including image or
  audio vectors. The tool flags the claim; only the code decides.
- "text to vector", "text to vectors", "converts text to vectors",
  and "converts text into vectors" are deliberately phrases: they
  can describe a TF-IDF / hashing / one-hot pipeline that turns
  text into numbers without a learned model. Such code is NOT
  evidence; the tool flags the claim and the code decides.
- A generic `embed(` call also names unrelated APIs (for example
  message embeds in chat libraries): the tool counts the call, and
  the doc claim plus the code's call together decide.
- A generic `encode_text(` / `encode_texts(` call can name a codec
  or a serialiser rather than a vector model: same rule — the tool
  counts the call, the claim plus the code decide.
- A torch `nn.Embedding` layer construction alone is NOT counted:
  declaring a lookup table is not generating vectors from text.
  The layer only matters if the code calls one of the listed
  routes on it, which a bare layer never does.
- A hand-rolled vector generator that produces vectors with
  differently-named methods (for example
  `model.encode_documents(...)` is covered by the generic list, but
  `model.vectorize(...)` is not) is NOT detected unless it uses a
  listed route or a listed generic name. The tool covers the
  sentence-transformers / OpenAI / Cohere / transformers /
  LangChain / LlamaIndex routes plus the listed generic names only.
- As with every family: static pattern matching, not semantic
  judgement; dead code counts; alias chains the tracer cannot follow
  are missed; exact phrases only, not semantics; denial handling
  depends on the denial phrase list; each pair must sit in a single
  module.
```

## Development

```sh
python3 -m unittest discover
python3 -m build
```
