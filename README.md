# Lokit AI – Personal AI Assistant

A sophisticated AI-powered personal assistant and portfolio website built with RAG (Retrieval-Augmented Generation), LangChain, and Streamlit. This application serves as an interactive chatbot that answers questions about Lokit Patidar's background, skills, experience, and projects using contextual information from a knowledge base.

##  Features

- **Interactive Chat Interface**: Modern, responsive chat UI with dark theme and smooth animations
- **RAG-Powered Responses**: Uses retrieval-augmented generation for accurate, context-aware answers
- **Vector Search**: FAISS-powered semantic search through embedded knowledge base
- **Resume Download**: Direct download functionality for Lokit's resume
- **Mobile Responsive**: Optimized for both desktop and mobile devices
- **Real-time Streaming**: Live response streaming for better user experience

##  Tech Stack

### Frontend
- **Streamlit**: Web framework for the chat interface
- **Custom CSS**: Dark premium theme with teal accent colors
- **Google Fonts**: DM Sans and Space Mono for typography

### Backend
- **LangChain**: Framework for building LLM applications
- **FAISS**: Vector database for efficient similarity search
- **Sentence Transformers**: For generating text embeddings
- **Groq API**: Fast LLM inference (can be swapped for other providers)
- **Python**: Core programming language

### Key Dependencies
- `streamlit` - Web interface
- `langchain` - LLM orchestration
- `langchain-community` - Community integrations
- `faiss-cpu` - Vector search
- `sentence-transformers` - Text embeddings
- `openai` - Alternative LLM provider
- `python-dotenv` - Environment variable management
- `tiktoken` - Token counting

## 📋 Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)
- OpenAI API key (or Groq API key for faster responses)
- Knowledge base documents (PDFs, text files about Lokit)

##  Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "My Portfolio"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirement.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   # OR
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Prepare the knowledge base:**
   - Place your documents (PDFs, text files) in the `knowledge base/` folder
   - Run the embedding script:
   ```bash
   python backend/embed_data.py
   ```

##  Usage

1. **Start the application:**
   ```bash
   streamlit run frontend/app.py
   ```

2. **Open your browser:**
   Navigate to `http://localhost:8501`

3. **Interact with the chatbot:**
   - Ask questions about Lokit's background, skills, experience, or projects
   - Download the resume using the sidebar button
   - Explore different conversation topics

##  Project Structure

```
My Portfolio/
├── frontend/
│   └── app.py                 # Streamlit frontend application
├── backend/
│   ├── chatbot.py             # RAG chatbot backend logic
│   ├── embed_data.py          # Knowledge base embedding script
│   └── lokit_ai_backend.zip   # Archived backend files
├── knowledge base/            # Directory for knowledge base documents
├── vector_db/                 # Generated vector database (created after running embed_data.py)
├── requirement.txt            # Python dependencies
└── README.md                  # This file
```

## 🔧 Configuration

### Embedding Model
The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`. You can change this in `backend/chatbot.py` and `backend/embed_data.py`.

### LLM Provider
Currently configured to use Groq API. To switch to OpenAI:
1. Update the import in `backend/chatbot.py`:
   ```python
   from langchain_openai import ChatOpenAI
   ```
2. Update the LLM initialization:
   ```python
   llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
   ```

### Vector Database
- **Retriever K**: Number of top chunks retrieved (default: 5)
- **Chunk Size**: Text chunk size for embedding (default: 1000 characters)
- **Overlap**: Chunk overlap for better context (default: 200 characters)

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Lokit Patidar**
- LinkedIn: [https://www.linkedin.com/in/lokit-patidar/]
- Portfolio: []
- Email: [lokitpatidar17@gmail.com]

---

*Built with using cutting-edge AI technologies*</content>
<parameter name="filePath">d:\Luck\My Portfolio\README.md
=======
# Lokit-s-AI-Chatbot
>>>>>>> 9b0981d (Initial commit)

