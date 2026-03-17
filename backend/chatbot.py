"""
╔══════════════════════════════════════════════════════════════════════╗
║  chatbot.py  –  RAG Chatbot Backend for Lokit AI                    ║
║                                                                      ║
║  Exposes a single public function:                                   ║
║      ask_bot(user_input: str) -> str                                 ║
║                                                                      ║
║  The Streamlit frontend imports and calls this directly:             ║
║      from backend.chatbot import ask_bot                             ║
║      answer = ask_bot("What are Lokit's skills?")                   ║
║                                                                      ║
║  Prerequisites:                                                      ║
║      Run  python backend/embed_data.py  before starting the app.    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
from pathlib import Path

# ── LangChain imports ─────────────────────────────────────────────────
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq 

 # swap for any LangChain LLM

# ──────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION  (mirrors embed_data.py)
# ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
VECTOR_DB_PATH  = PROJECT_ROOT / "vector_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# How many top-k chunks the retriever returns per query
RETRIEVER_K = 5


# ──────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATE
# Instructs the LLM to act as Lokit's AI representative and answer
# using ONLY the retrieved context (grounded RAG behaviour).
# ──────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """You are Lokit Patidar, a passionate AI and Machine Learning engineer based in India.
Answer questions about yourself in a helpful, professional, and friendly manner using "I" and "my".

Use ONLY the information provided in the context below to answer the question.
If the answer is not present in the context, politely say:
"I'm not sure about that specific detail, but feel free to ask me anything else!"

Do NOT make up or invent any facts. Be concise yet informative.
When listing items (skills, projects, etc.) use clean bullet points.

─────────────────────────────────────
Your Knowledge Base:
{context}
─────────────────────────────────────

Question: {question}

Answer:"""


# ──────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETONS
# These are loaded once when the module is first imported and reused
# for every subsequent call to ask_bot() — avoids repeated disk I/O
# and model loading on every user message.
# ──────────────────────────────────────────────────────────────────────

_vector_store  = None   # FAISS vector store
_retriever     = None   # LangChain retriever wrapping FAISS
_llm           = None   # LangChain LLM (Groq by default)
_prompt        = None   # PromptTemplate instance


# ──────────────────────────────────────────────────────────────────────
# INITIALISATION HELPERS
# ──────────────────────────────────────────────────────────────────────

def _check_vector_db_exists():
    """
    Verify that the FAISS index files exist on disk.
    Raises a clear RuntimeError if embed_data.py has not been run.
    """
    index_file = VECTOR_DB_PATH / "index.faiss"
    pkl_file   = VECTOR_DB_PATH / "index.pkl"

    if not VECTOR_DB_PATH.exists() or not index_file.exists() or not pkl_file.exists():
        raise RuntimeError(
            "\n"
            "╔══════════════════════════════════════════════════════════╗\n"
            "║  ❌  FAISS vector database not found!                   ║\n"
            "║                                                          ║\n"
            "║  Please run the embedding pipeline first:               ║\n"
            "║      python backend/embed_data.py                       ║\n"
            "║                                                          ║\n"
            "║  This only needs to be done once (or after you update   ║\n"
            "║  any files in the knowledge_base/ folder).              ║\n"
            "╚══════════════════════════════════════════════════════════╝"
        )


def _load_vector_store():
    """
    Load the FAISS index from disk using the same embedding model
    that was used to build it in embed_data.py.

    Returns:
        FAISS: Loaded vector store, ready for similarity search.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # allow_dangerous_deserialization=True is required by LangChain
    # when loading a FAISS index that uses pickle serialisation.
    # This is safe here because we created the index ourselves.
    vector_store = FAISS.load_local(
        folder_path=str(VECTOR_DB_PATH),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store


def _create_retriever(vector_store):
    """
    Wrap the FAISS vector store as a LangChain retriever.

    search_type="similarity"  – standard cosine similarity search.
    k=RETRIEVER_K             – return the top-k most relevant chunks.

    The retriever's .invoke(query) method returns a list of Document
    objects whose page_content fields form the RAG context.

    Args:
        vector_store (FAISS): Loaded FAISS vector store.

    Returns:
        VectorStoreRetriever: LangChain retriever.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )
    return retriever


def _create_llm():
    """
    Initialise the LLM.

    Default: Groq API with Gemma-7b-It (fast, free tier available).
    To swap the LLM, replace this function's body with any
    LangChain-compatible chat model, for example:

        # OpenAI
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

        # Ollama (local, no API key needed)
        from langchain_ollama import ChatOllama
        return ChatOllama(model="llama3", temperature=0.2)

        # Together AI
        from langchain_together import ChatTogether
        return ChatTogether(model="mistralai/Mistral-7B-Instruct-v0.2")

    Environment variable required for Groq:
        GROQ_API_KEY=your_key_here
        Set in a .env file or export in your shell.

    Returns:
        BaseChatModel: Instantiated LangChain LLM.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise EnvironmentError(
            "\n"
            "╔══════════════════════════════════════════════════════════╗\n"
            "║  ❌  GROQ_API_KEY environment variable is not set!      ║\n"
            "║                                                          ║\n"
            "║  Get a free key at: https://console.groq.com            ║\n"
            "║                                                          ║\n"
            "║  Then set it:                                            ║\n"
            "║    • Add  GROQ_API_KEY=your_key  to a .env file, OR     ║\n"
            "║    • Export it in your shell before running the app.    ║\n"
            "╚══════════════════════════════════════════════════════════╝"
        )

    llm = ChatGroq(
        model="gemma2-9b-it",         # fast, high-quality, free tier
        temperature=0.3,              # low temp → factual, grounded answers
        max_tokens=1024,              # max response length
        groq_api_key=groq_api_key,
    )
    return llm


def _initialise():
    """
    Lazy initialisation: loads all heavy components on the first call
    to ask_bot() and caches them in module-level globals.

    Subsequent calls to ask_bot() skip this function entirely
    (all globals are already populated).
    """
    global _vector_store, _retriever, _llm, _prompt

    # ── Validate FAISS index exists ──────────────────────
    _check_vector_db_exists()

    # ── Load FAISS vector store from disk ────────────────
    _vector_store = _load_vector_store()

    # ── Create retriever ─────────────────────────────────
    _retriever = _create_retriever(_vector_store)

    # ── Initialise LLM ───────────────────────────────────
    _llm = _create_llm()

    # ── Compile the prompt template ──────────────────────
    _prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE,
    )


# ──────────────────────────────────────────────────────────────────────
# HELPER: FORMAT RETRIEVED DOCUMENTS INTO A SINGLE CONTEXT STRING
# ──────────────────────────────────────────────────────────────────────

def _format_context(docs: list[Document]) -> str:
    """
    Merge the page_content of retrieved Document chunks into a single
    context string, separated by a blank line.

    Each chunk is prefixed with its source filename so the LLM has
    light provenance context (it won't cite these, but it helps with
    multi-document coherence).

    Args:
        docs (list[Document]): Documents returned by the retriever.

    Returns:
        str: Combined context text to inject into the prompt.
    """
    parts = []
    for doc in docs:
        # Extract just the filename for a clean label
        source = Path(doc.metadata.get("source", "unknown")).name
        parts.append(f"[Source: {source}]\n{doc.page_content.strip()}")

    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────
# PUBLIC API  –  the only function the frontend needs to call
# ──────────────────────────────────────────────────────────────────────

def ask_bot(user_input: str) -> str:
    """
    Main RAG pipeline entry point.

    Full flow:
        1. Lazy-initialise all components (only on first call).
        2. Embed the user's question and run FAISS similarity search
           to retrieve the top-k most relevant knowledge chunks.
        3. Concatenate the chunks into a single context string.
        4. Fill the prompt template with (context, question).
        5. Send the filled prompt to the LLM and receive a response.
        6. Return the plain-text answer.

    Args:
        user_input (str): The user's question or message.

    Returns:
        str: The chatbot's plain-text answer (safe to display in Streamlit).

    Usage in Streamlit:
        from backend.chatbot import ask_bot
        answer = ask_bot("What are Lokit's skills?")
        st.write(answer)
    """
    # ── Guard: empty input ───────────────────────────────
    if not user_input or not user_input.strip():
        return "Please ask me something about Lokit — I'm here to help! 😊"

    # ── Lazy initialisation (runs only once per process) ─
    global _vector_store, _retriever, _llm, _prompt
    if _retriever is None:
        _initialise()

    try:
        # ── STEP 1: Retrieve relevant chunks ────────────
        # The retriever embeds user_input and finds the
        # RETRIEVER_K most semantically similar chunks in FAISS.
        relevant_docs = _retriever.invoke(user_input.strip())

        # ── STEP 2: Build context string ─────────────────
        if not relevant_docs:
            # Edge case: no relevant documents found at all
            return (
                "I couldn't find relevant information in my knowledge base "
                "for that question. You can reach out to Lokit directly for more details!"
            )

        context = _format_context(relevant_docs)

        # ── STEP 3: Fill the prompt template ─────────────
        filled_prompt = _prompt.format(
            context=context,
            question=user_input.strip(),
        )

        # ── STEP 4: Call the LLM ──────────────────────────
        # Wrap in a HumanMessage for chat-model compatibility
        from langchain_core.messages import HumanMessage

        response = _llm.invoke([HumanMessage(content=filled_prompt)])

        # ── STEP 5: Extract and return the answer ─────────
        # response.content is a plain string for all LangChain chat models
        answer = response.content.strip()
        return answer if answer else "I received an empty response. Please try rephrasing your question."

    except Exception as e:
        # Return a user-friendly error rather than crashing the UI
        return (
            f"⚠️ An error occurred while processing your question.\n\n"
            f"**Error details:** {str(e)}\n\n"
            "Please check your API key, ensure the FAISS index exists, "
            "and try again."
        )


# ──────────────────────────────────────────────────────────────────────
# OPTIONAL: DIRECT TERMINAL TEST
# Run  python backend/chatbot.py  to test the backend without Streamlit
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Lokit AI – Terminal Test Mode")
    print("  Type 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        user_q = input("You: ").strip()
        if user_q.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_q:
            continue

        print("\nLokit AI: ", end="", flush=True)
        answer = ask_bot(user_q)
        print(answer)
        print()
