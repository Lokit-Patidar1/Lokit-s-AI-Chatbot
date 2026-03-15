"""
╔══════════════════════════════════════════════════════════════════════╗
║  embed_data.py  –  Knowledge Base Embedding & FAISS Index Builder   ║
║                                                                      ║
║  Run this script ONCE (and whenever you update the knowledge base)   ║
║  to:                                                                 ║
║    1. Load all .txt files from  knowledge_base/                      ║
║    2. Split them into overlapping chunks                             ║
║    3. Generate embeddings with sentence-transformers/all-MiniLM-L6-v2║
║    4. Build a FAISS vector index and save it to  vector_db/          ║
║                                                                      ║
║  Usage:                                                              ║
║    python backend/embed_data.py                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
from pathlib import Path

# ── LangChain imports ─────────────────────────────────────────────────
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ──────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# Resolve paths relative to this file so the script works regardless
# of the working directory it is invoked from.
# ──────────────────────────────────────────────────────────────────────

# Root of the project  (one level up from backend/)
PROJECT_ROOT    = Path(__file__).resolve().parent.parent

# Folder containing raw .txt knowledge files
KNOWLEDGE_BASE  = PROJECT_ROOT / "knowledge_base"

# Folder where the FAISS index will be saved
VECTOR_DB_PATH  = PROJECT_ROOT / "vector_db"

# Embedding model – lightweight, high-quality, runs on CPU
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text-splitting parameters
CHUNK_SIZE      = 500   # characters per chunk
CHUNK_OVERLAP   = 80    # overlap to preserve context across chunks


# ──────────────────────────────────────────────────────────────────────
# STEP 1 – LOAD DOCUMENTS
# ──────────────────────────────────────────────────────────────────────

def load_documents():
    """
    Load all .txt files from the knowledge_base/ directory.

    Uses LangChain's DirectoryLoader with TextLoader for plain text.
    Each file becomes one or more LangChain Document objects that
    carry both page_content (the text) and metadata (source path).

    Returns:
        list[Document]: Raw LangChain documents, one per file.

    Raises:
        SystemExit: If the knowledge_base folder is missing or empty.
    """
    if not KNOWLEDGE_BASE.exists():
        print(f"[ERROR] Knowledge base folder not found: {KNOWLEDGE_BASE}")
        print("        Please create the 'knowledge_base/' directory and add .txt files.")
        sys.exit(1)

    txt_files = list(KNOWLEDGE_BASE.glob("*.txt"))
    if not txt_files:
        print(f"[ERROR] No .txt files found in: {KNOWLEDGE_BASE}")
        print("        Add at least one .txt file to the knowledge_base/ folder.")
        sys.exit(1)

    print(f"[INFO] Found {len(txt_files)} file(s) in {KNOWLEDGE_BASE}:")
    for f in txt_files:
        print(f"         • {f.name}")

    # DirectoryLoader automatically iterates over matching files
    loader = DirectoryLoader(
        str(KNOWLEDGE_BASE),
        glob="*.txt",                      # only text files
        loader_cls=TextLoader,             # use simple text loader
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,                # progress bar in terminal
    )

    documents = loader.load()
    print(f"[INFO] Loaded {len(documents)} document(s) total.\n")
    return documents


# ──────────────────────────────────────────────────────────────────────
# STEP 2 – SPLIT DOCUMENTS INTO CHUNKS
# ──────────────────────────────────────────────────────────────────────

def split_documents(documents):
    """
    Split raw documents into smaller, overlapping text chunks.

    Why chunking?
        FAISS retrieves individual chunks, not whole files.
        Smaller chunks = more precise similarity matching.
        Overlap ensures that no important sentence is cut in half
        and lost between two adjacent chunks.

    Args:
        documents (list[Document]): Raw documents from load_documents().

    Returns:
        list[Document]: Chunked documents ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Preferred split points (tries paragraph → sentence → word → char)
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    print(f"[INFO] Split {len(documents)} document(s) into {len(chunks)} chunk(s).")
    print(f"       Chunk size: {CHUNK_SIZE} chars  |  Overlap: {CHUNK_OVERLAP} chars\n")
    return chunks


# ──────────────────────────────────────────────────────────────────────
# STEP 3 – CREATE EMBEDDING MODEL
# ──────────────────────────────────────────────────────────────────────

def create_embedding_model():
    """
    Initialise the HuggingFace sentence-transformer embedding model.

    Model: sentence-transformers/all-MiniLM-L6-v2
        • 384-dimensional dense vectors
        • ~22 MB download on first run (cached locally afterward)
        • Fast CPU inference, excellent semantic similarity quality
        • Widely used for RAG applications

    Returns:
        HuggingFaceEmbeddings: Ready-to-use embedding model.
    """
    print(f"[INFO] Loading embedding model: {EMBEDDING_MODEL}")
    print("       (First run may take ~30s to download the model weights…)\n")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},    # change to "cuda" if GPU is available
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity ready
    )

    print("[INFO] Embedding model loaded successfully.\n")
    return embeddings


# ──────────────────────────────────────────────────────────────────────
# STEP 4 – BUILD AND SAVE THE FAISS INDEX
# ──────────────────────────────────────────────────────────────────────

def build_and_save_faiss(chunks, embeddings):
    """
    Build a FAISS vector store from the document chunks and persist it.

    What happens here:
        1. Each chunk's text is passed through the embedding model
           to produce a 384-dim float vector.
        2. All vectors are inserted into a FAISS IndexFlatL2 index
           (exact nearest-neighbour, no approximation).
        3. The index + docstore are serialised to disk so they can be
           loaded quickly by chatbot.py without re-embedding.

    Args:
        chunks    (list[Document]): Chunked documents.
        embeddings (HuggingFaceEmbeddings): Embedding model.

    Returns:
        FAISS: The in-memory FAISS vector store (also saved to disk).
    """
    print("[INFO] Building FAISS vector store…")
    print(f"       Embedding {len(chunks)} chunks — this may take a moment…\n")

    # Build the vector store (embeds all chunks in one pass)
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    # Ensure the output directory exists
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

    # Save the index to disk (creates index.faiss + index.pkl)
    vector_store.save_local(str(VECTOR_DB_PATH))

    print(f"[SUCCESS] FAISS index saved to: {VECTOR_DB_PATH}")
    print(f"          Files created: index.faiss, index.pkl\n")

    return vector_store


# ──────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────

def main():
    """
    End-to-end pipeline:
        load → split → embed → save FAISS index
    Run this once before starting the chatbot.
    """
    print("=" * 60)
    print("  Lokit AI – Knowledge Base Embedding Pipeline")
    print("=" * 60 + "\n")

    # 1. Load raw text documents
    documents = load_documents()

    # 2. Split into smaller overlapping chunks
    chunks = split_documents(documents)

    # 3. Load the embedding model
    embeddings = create_embedding_model()

    # 4. Embed chunks and save FAISS index
    build_and_save_faiss(chunks, embeddings)

    print("=" * 60)
    print("  ✅  Embedding pipeline complete!")
    print("  You can now run the Streamlit app:")
    print("      streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
