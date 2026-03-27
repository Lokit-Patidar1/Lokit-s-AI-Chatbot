import os
import sys
from pathlib import Path

# Optional .env loading (local dev convenience)
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq 

# PATH CONFIGURATION

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
VECTOR_DB_PATH  = PROJECT_ROOT / "vector_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVER_K = 5

# Load .env at project root if available
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

# PROMPT TEMPLATE

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



# MODULE-LEVEL SINGLETONS

_vector_store  = None   
_retriever     = None   
_llm           = None   
_prompt        = None   

# INITIALISATION HELPERS

def _check_vector_db_exists():
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
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = FAISS.load_local(
        folder_path=str(VECTOR_DB_PATH),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store


def _create_retriever(vector_store):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )
    return retriever


def _create_llm():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        try:
            import streamlit as st
            groq_api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            groq_api_key = None
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
        model="llama-3.1-8b-instant", # replacement for deprecated gemma2-9b-it
        temperature=0.3,              # low temp → factual, grounded answers
        max_tokens=1024,              # max response length
        groq_api_key=groq_api_key,
    )
    return llm


def _initialise():
    global _vector_store, _retriever, _llm, _prompt
    _check_vector_db_exists()
    _vector_store = _load_vector_store()
    _retriever = _create_retriever(_vector_store)
    _llm = _create_llm()
    _prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE,
    )

# HELPER: FORMAT RETRIEVED DOCUMENTS INTO A SINGLE CONTEXT STRING
def _format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        # Extract just the filename for a clean label
        source = Path(doc.metadata.get("source", "unknown")).name
        parts.append(f"[Source: {source}]\n{doc.page_content.strip()}")

    return "\n\n".join(parts)

# PUBLIC API  –  the only function the frontend needs to call

def ask_bot(user_input: str) -> str:
    if not user_input or not user_input.strip():
        return "Please ask me something about Lokit — I'm here to help! 😊"
    global _vector_store, _retriever, _llm, _prompt
    if _retriever is None:
        _initialise()

    try:
        relevant_docs = _retriever.invoke(user_input.strip())
        if not relevant_docs:
            return (
                "I couldn't find relevant information in my knowledge base "
                "for that question. You can reach out to Lokit directly for more details!"
            )

        context = _format_context(relevant_docs)
        filled_prompt = _prompt.format(
            context=context,
            question=user_input.strip(),
        )
        from langchain_core.messages import HumanMessage

        response = _llm.invoke([HumanMessage(content=filled_prompt)])
        answer = response.content.strip()
        return answer if answer else "I received an empty response. Please try rephrasing your question."

    except Exception as e:
        return (
            f"⚠️ An error occurred while processing your question.\n\n"
            f"**Error details:** {str(e)}\n\n"
            "Please check your API key, ensure the FAISS index exists, "
            "and try again."
        )

# OPTIONAL: DIRECT TERMINAL TEST

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

