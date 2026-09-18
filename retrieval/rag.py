"""
Research Paper AI - Conversational RAG CLI

Pipeline:

User Question
      ↓
Conversation-aware query understanding
      ↓
Focused retrieval query
      ↓
Semantic retrieval
      ↓
Paper-aware reranking
      ↓
Diversity filtering
      ↓
Context construction
      ↓
LLM answer
      ↓
Conversation history

The system is generic and is NOT tied to PPO/DQN.
"""

import json
import os
import re
from pathlib import Path


from retrieval.llm_provider import generate_with_fallback


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDINGS_FILE = os.getenv(
    "EMBEDDINGS_FILE",
    "data/processed/research_embeddings.json",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

TOP_K = int(
    os.getenv("TOP_K", "6")
)

RETRIEVAL_K = int(
    os.getenv("RETRIEVAL_K", "20")
)

MIN_SIMILARITY = float(
    os.getenv("MIN_SIMILARITY", "0.12")
)

MAX_HISTORY_TURNS = int(
    os.getenv("MAX_HISTORY_TURNS", "6")
)


# ============================================================
# STARTUP
# ============================================================

print("=" * 72)
print("LOADING RESEARCH PAPER AI")
print("=" * 72)


# ------------------------------------------------------------
# LAZY EMBEDDING MODEL
# ------------------------------------------------------------
#
# IMPORTANT FOR DEPLOYMENT:
#
# Do NOT load SentenceTransformer during module import.
#
# Render/Uvicorn needs the application to start and bind to
# $PORT quickly. Loading the Hugging Face model during import
# can delay startup and cause Render's port scanner to timeout.
#
# The model will be loaded only when retrieve_chunks() is called.
# ------------------------------------------------------------

# ============================================================
# LIGHTWEIGHT DEPLOYMENT MODE
# ============================================================
# Render Free has 512 MB RAM, so retrieval uses lightweight
# lexical scoring instead of loading a local transformer model.
# The LLM is still used for conversational query rewriting
# and grounded answer generation.

def get_embedding_model():
    return None

# ============================================================
# LOAD RESEARCH-PAPER EMBEDDINGS
# ============================================================

print(
    "\nLoading research-paper embeddings..."
)

embeddings_path = Path(
    EMBEDDINGS_FILE
)

if not embeddings_path.exists():

    raise FileNotFoundError(
        f"Embeddings file not found: "
        f"{embeddings_path}"
    )


with embeddings_path.open(
    "r",
    encoding="utf-8",
) as file:

    documents = json.load(
        file
    )


print(
    f"Total chunks loaded: "
    f"{len(documents)}"
)


if not documents:

    raise ValueError(
        "The embeddings file is empty."
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    text: str,
) -> str:
    """
    Normalize text for lightweight keyword matching.
    """

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s-]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# QUERY KEYWORDS
# ============================================================

def extract_keywords(
    question: str,
):
    """
    Extract useful keywords.

    Semantic similarity remains the primary retrieval signal.
    """

    stop_words = {
        "what",
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "or",
        "why",
        "how",
        "does",
        "do",
        "did",
        "was",
        "were",
        "with",
        "from",
        "about",
        "this",
        "that",
        "it",
        "its",
        "they",
        "them",
        "their",
        "be",
        "can",
        "could",
        "would",
        "should",
    }

    words = normalize_text(
        question
    ).split()

    return {
        word
        for word in words
        if word not in stop_words
        and len(word) > 1
    }


# ============================================================
# KEYWORD OVERLAP
# ============================================================

def keyword_overlap(
    question_keywords,
    text,
):
    """
    Calculate lightweight keyword overlap.
    """

    if not question_keywords:

        return 0.0

    normalized = normalize_text(
        text
    )

    matched = 0

    for keyword in question_keywords:

        if keyword in normalized:

            matched += 1

    return matched / len(
        question_keywords
    )


# ============================================================
# CONVERSATION-AWARE QUERY REWRITING
# ============================================================

def build_retrieval_query(
    question,
    history,
):
    """
    Convert a conversational follow-up question into a
    self-contained retrieval query.

    We do NOT concatenate the entire conversation.

    Example:

        User: What is PPO?
        User: Why is it useful?

    Retrieval query becomes:

        Why is PPO useful?

    This keeps semantic retrieval focused while still
    allowing conversational follow-ups.
    """

    if not history:

        return question.strip()


    recent_history = history[
        -MAX_HISTORY_TURNS:
    ]


    history_text = "\n".join(
        f"{item['role'].upper()}: "
        f"{item['content']}"
        for item in recent_history
    )


    rewrite_system_prompt = """
You are a retrieval-query rewriting assistant for a
research-paper question-answering system.

Your job is to rewrite the user's latest question into
ONE concise, self-contained search query.

Rules:

1. Preserve the exact technical meaning of the user's question.

2. Resolve conversational references such as:
   - it
   - this
   - that
   - the method
   - the algorithm
   - the paper
   - they
   - them

3. Use the conversation history only to understand those references.

4. DO NOT include previous questions unless they are necessary
   to resolve the meaning of the current question.

5. Do not answer the question.

6. Do not explain anything.

7. Return ONLY the rewritten retrieval query.

Examples:

Conversation:
USER: What is PPO?
USER: Why is it useful?

Latest question:
Why is it useful?

Output:
Why is PPO useful?

---

Conversation:
USER: What is PPO?
USER: How does it differ from TRPO?

Latest question:
Why was it proposed?

Output:
Why was PPO proposed compared with TRPO?

---

Conversation:
USER: What is DQN?
USER: What is experience replay?

Latest question:
Why is it important?

Output:
Why is experience replay important in DQN?
"""


    messages = [
        {
            "role": "system",
            "content": (
                rewrite_system_prompt.strip()
            ),
        },
        {
            "role": "user",
            "content": (
                "CONVERSATION HISTORY\n"
                "--------------------\n"
                f"{history_text}\n\n"
                "LATEST USER QUESTION\n"
                "--------------------\n"
                f"{question}\n\n"
                "Return only the rewritten retrieval query."
            ),
        },
    ]


    try:

        result = generate_with_fallback(
            messages
        )

        rewritten = result.text.strip()

        # Remove accidental quotation marks.
        rewritten = rewritten.strip(
            "\"'"
        )

        # Prevent excessively long retrieval queries.
        if (
            rewritten
            and len(rewritten) <= 500
        ):

            return rewritten

    except Exception:

        pass


    # Safe fallback:
    # If rewriting fails, use the original question.
    return question.strip()


# ============================================================
# RETRIEVAL
# ============================================================

def _make_result(doc, score, overlap):
    return {
        "chunk_id": doc.get("chunk_id", "unknown"),
        "paper_id": doc.get("paper_id", "unknown"),
        "paper_name": doc.get(
            "paper_name",
            doc.get("source", "unknown"),
        ),
        "page": doc.get("page", "?"),
        "section": doc.get("section", ""),
        "text": doc.get("text", ""),
        "similarity": float(score),
        "keyword_overlap": float(overlap),
        "score": float(score),
    }


def _rank_documents(question, candidate_documents, top_k=TOP_K):
    """
    Lightweight BM25-style lexical retrieval.

    No PyTorch, Hugging Face model, or transformer is loaded.
    This keeps the application compatible with Render Free.
    """
    question_keywords = extract_keywords(question)

    if not question_keywords:
        question_keywords = set(
            normalize_text(question).split()
        )

    candidates = []

    for doc in candidate_documents:
        text = doc.get("text", "")
        normalized = normalize_text(text)
        words = normalized.split()

        if not words:
            continue

        word_set = set(words)

        matched = question_keywords.intersection(word_set)

        if not matched:
            # Also support partial technical terms.
            matched = {
                keyword
                for keyword in question_keywords
                if any(
                    keyword in word
                    for word in word_set
                )
            }

        overlap = (
            len(matched) / len(question_keywords)
            if question_keywords
            else 0.0
        )

        # Frequency bonus: repeated relevant terms matter.
        frequency_score = 0.0

        for keyword in matched:
            count = words.count(keyword)
            frequency_score += min(count, 5) / 5.0

        if matched:
            frequency_score /= len(matched)

        # Section/title bonus.
        section_text = normalize_text(
            str(doc.get("section", ""))
        )

        section_bonus = 0.0

        for keyword in question_keywords:
            if keyword in section_text:
                section_bonus += 1.0

        if question_keywords:
            section_bonus /= len(question_keywords)

        score = (
            0.65 * overlap
            + 0.20 * frequency_score
            + 0.15 * section_bonus
        )

        candidates.append(
            _make_result(
                doc,
                score,
                overlap,
            )
        )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    # Diversity: maximum two chunks from the same page.
    selected = []
    seen_pages = {}

    for item in candidates:
        page_key = (
            item["paper_name"],
            item["page"],
        )

        count = seen_pages.get(page_key, 0)

        if count >= 2:
            continue

        selected.append(item)
        seen_pages[page_key] = count + 1

        if len(selected) >= top_k:
            break

    return selected


def _keyword_retrieve(question, candidate_docs, top_k=TOP_K):
    """Lightweight retrieval that does not load a transformer model."""

    question_keywords = extract_keywords(question)
    candidates = []

    for doc in candidate_docs:
        text = doc.get("text", "")
        overlap = keyword_overlap(question_keywords, text)

        # Also reward exact phrase matches.
        normalized_question = normalize_text(question)
        normalized_text = normalize_text(text)

        phrase_bonus = 0.0
        if normalized_question and normalized_question in normalized_text:
            phrase_bonus = 0.5

        score = min(1.0, overlap + phrase_bonus)

        candidates.append({
            "chunk_id": doc.get("chunk_id", "unknown"),
            "paper_id": doc.get("paper_id", "unknown"),
            "paper_name": doc.get(
                "paper_name",
                doc.get("source", "unknown")
            ),
            "page": doc.get("page", "?"),
            "section": doc.get("section", ""),
            "text": text,
            "similarity": score,
            "keyword_overlap": overlap,
            "score": score,
        })

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    selected = []
    seen_pages = {}

    for item in candidates:
        page_key = (
            item["paper_name"],
            item["page"],
        )

        count = seen_pages.get(page_key, 0)

        if count >= 2:
            continue

        selected.append(item)
        seen_pages[page_key] = count + 1

        if len(selected) >= top_k:
            break

    return selected


def retrieve_chunks_for_paper(
    question,
    paper_id,
    top_k=TOP_K,
):
    """Retrieve relevant chunks from one paper without embeddings."""

    paper_docs = [
        doc for doc in documents
        if doc.get("paper_id") == paper_id
    ]

    return _keyword_retrieve(
        question,
        paper_docs,
        top_k,
    )


def retrieve_chunks(
    question,
    top_k=TOP_K,
):
    """
    Lightweight retrieval for low-memory deployment.

    Uses keyword overlap instead of SentenceTransformer embeddings.
    """

    return _keyword_retrieve(
        question,
        documents,
        top_k,
    )


# ============================================================
# CONTEXT
# ============================================================

def build_context(
    results,
):
    """
    Convert retrieved chunks into LLM context.
    """

    parts = []


    for index, result in enumerate(
        results,
        start=1,
    ):

        parts.append(
            f"""
SOURCE {index}

Paper: {result['paper_name']}
Paper ID: {result['paper_id']}
Page: {result['page']}
Section: {result['section']}

{result['text']}
"""
        )


    return "\n".join(
        parts
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Research Paper AI, an academic research assistant.

Your job is to answer questions about the research papers
available in the retrieved evidence.

IMPORTANT RULES:

1. Use the retrieved research-paper evidence as the primary
   source of factual information.

2. Do not invent facts, experiments, equations, citations,
   paper details, or page numbers.

3. If the retrieved evidence does not contain enough information
   to answer the question confidently, say:

   "I couldn't find enough information in the retrieved
   research papers to answer this confidently."

4. Cite evidence naturally using the paper name and page number.

   Example:
   "According to 2509.08221v1, page 9, PPO uses..."

5. When several papers are relevant, distinguish them clearly.

6. Explain technical concepts clearly and naturally.

7. General knowledge may be used to clarify terminology,
   but do not present unsupported claims as if they came
   from the retrieved papers.

8. Conversation history may be used to understand references
   such as "it", "this", "that", "the method", "the algorithm",
   or "the previous paper".

9. Answer the CURRENT question directly.

10. Do not repeat the entire previous conversation.

11. Do not mention internal retrieval scores.

12. Do not mention provider failures, API keys, embeddings,
    vector databases, or implementation details unless
    explicitly asked.

13. Do not claim that a paper proves something unless the
    retrieved evidence actually supports that claim.

14. Prefer concise but useful answers.

15. If the user asks for a comparison, use a table when useful.

16. If the user asks a follow-up question, maintain the topic
    from the conversation while answering only the new question.
"""


# ============================================================
# MESSAGE CONSTRUCTION
# ============================================================

def build_messages(
    question,
    context,
    history,
):
    """
    Construct the final answer-generation messages.
    """

    history_text = ""


    if history:

        history_text = (
            "\n\nCONVERSATION HISTORY\n"
            "--------------------\n"
            + "\n".join(
                f"{item['role'].upper()}: "
                f"{item['content']}"
                for item in history[
                    -MAX_HISTORY_TURNS:
                ]
            )
        )


    user_prompt = f"""
RETRIEVED RESEARCH-PAPER EVIDENCE
---------------------------------

{context}

---------------------------------

{history_text}

CURRENT USER QUESTION
---------------------

{question}

Answer the CURRENT user question using the retrieved
research-paper evidence.
"""


    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.strip(),
        },
        {
            "role": "user",
            "content": user_prompt.strip(),
        },
    ]


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(
    results,
):
    """
    Display retrieved sources for debugging/transparency.
    """

    print(
        "\n"
        + "-" * 72
    )


    print(
        "RETRIEVED SOURCES"
    )


    print(
        "-" * 72
    )


    for index, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"{index}. "
            f"{result['paper_name']} "
            f"| Page {result['page']} "
            f"| Similarity "
            f"{result['similarity']:.4f} "
            f"| Score "
            f"{result['score']:.4f}"
        )


# ============================================================
# PROVIDER DISPLAY
# ============================================================

def display_provider(
    result,
):
    """
    Display which LLM generated the answer.
    """

    print(
        f"\n[ANSWER GENERATED BY] "
        f"{result.provider} "
        f"→ {result.model}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 72
    )


    print(
        "RESEARCH PAPER AI"
    )


    print(
        "=" * 72
    )


    print(
        "\nAsk questions about your research papers."
    )


    print(
        "Commands:"
    )


    print(
        "  clear   - clear conversation history"
    )


    print(
        "  history - show conversation history"
    )


    print(
        "  exit    - stop"
    )


    print()


    history = []


    while True:

        try:

            question = input(
                "You: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print(
                "\n\nExiting..."
            )

            break


        if not question:

            continue


        # ====================================================
        # COMMANDS
        # ====================================================

        if question.lower() == "exit":

            print(
                "\nExiting..."
            )

            break


        if question.lower() == "clear":

            history.clear()

            print(
                "\nConversation cleared.\n"
            )

            continue


        if question.lower() == "history":

            if not history:

                print(
                    "\nNo conversation history yet.\n"
                )

            else:

                print(
                    "\n"
                    + "-" * 72
                )


                print(
                    "CONVERSATION HISTORY"
                )


                print(
                    "-" * 72
                )


                for item in history:

                    print(
                        f"{item['role'].upper()}: "
                        f"{item['content']}\n"
                    )

            continue


        # ====================================================
        # CASUAL CONVERSATION
        # ====================================================

        casual_words = {
            "hi",
            "hey",
            "hello",
            "thanks",
            "thank you",
            "good morning",
            "good evening",
            "good night",
        }


        if question.lower() in casual_words:

            print(
                "\nCasual conversation..."
            )


            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a friendly research "
                        "assistant. Respond naturally "
                        "and briefly."
                    ),
                },
                {
                    "role": "user",
                    "content": question,
                },
            ]


            try:

                result = generate_with_fallback(
                    messages
                )


                display_provider(
                    result
                )


                print(
                    "\nAI:"
                )


                print(
                    result.text
                )


            except Exception as exc:

                print(
                    "\n[LLM] All providers failed."
                )


                print(
                    f"Technical reason: {exc}"
                )


            print(
                "\n"
                + "=" * 72
            )


            continue


        # ====================================================
        # RESEARCH QUESTION
        # ====================================================

        print(
            "\nUnderstanding question..."
        )


        # ----------------------------------------------------
        # Create focused retrieval query.
        # ----------------------------------------------------

        retrieval_query = build_retrieval_query(
            question=question,
            history=history,
        )


        print(
            f"[RETRIEVAL QUERY] "
            f"{retrieval_query}"
        )


        # ----------------------------------------------------
        # Retrieve using ONLY the focused query.
        # ----------------------------------------------------

        results = retrieve_chunks(
            retrieval_query
        )


        display_sources(
            results
        )


        if not results:

            print(
                "\nNo relevant research "
                "evidence found."
            )

            continue


        # ====================================================
        # WEAK RETRIEVAL WARNING
        # ====================================================

        if (
            results[0]["similarity"]
            < MIN_SIMILARITY
        ):

            context = (
                "IMPORTANT: Retrieval confidence "
                "is low. The retrieved chunks may "
                "not directly answer the question.\n\n"
                + build_context(
                    results
                )
            )

        else:

            context = build_context(
                results
            )


        # ====================================================
        # GENERATE ANSWER
        # ====================================================

        print(
            "\nGenerating answer..."
        )


        messages = build_messages(
            question=question,
            context=context,
            history=history,
        )


        try:

            result = generate_with_fallback(
                messages
            )


            display_provider(
                result
            )


            print(
                "\nAI:"
            )


            print(
                result.text
            )


            # ------------------------------------------------
            # Store the actual conversation.
            #
            # We intentionally do NOT store the retrieval
            # query as conversation history.
            # ------------------------------------------------

            history.append(
                {
                    "role": "user",
                    "content": question,
                }
            )


            history.append(
                {
                    "role": "assistant",
                    "content": result.text,
                }
            )


        except Exception as exc:

            print(
                "\n[LLM] All providers failed."
            )


            print(
                f"Technical reason: {exc}"
            )


        print(
            "\n"
            + "=" * 72
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()