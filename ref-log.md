# Reference Log (ref-log.md)

## Assignment 1: RAG-based Document Chat Application

---

## Architecture & RAG Pipeline

### RAG Pipeline Flow

The application implements a complete Retrieval-Augmented Generation (RAG) pipeline with the following steps:

**Step 1: Document Ingestion** (`load_document()`)
- User uploads .txt or .pdf files via Streamlit file uploader
- Files are saved to temporary storage
- LangChain's `TextLoader` (for .txt) or `PyPDFLoader` (for .pdf) loads document content
- Each document is converted to LangChain `Document` objects with metadata (filename)
- Temporary files are cleaned up after loading

**Step 2: Document Chunking** (`chunk_documents()`)
- Uses `RecursiveCharacterTextSplitter` to intelligently split documents
- Parameters: `chunk_size=1000` characters, `chunk_overlap=200` characters
- Splits on natural boundaries: `["\n\n", "\n", " ", ""]` (paragraphs - sentences - words - characters)
- Overlap ensures context continuity across chunk boundaries
- Each chunk becomes a separate `Document` object

**Step 3: Embedding** (within `create_vectorstore()`)
- Initializes `OpenAIEmbeddings` with model `text-embedding-3-large`
- Each document chunk is converted to a 3072-dimensional vector
- Embeddings capture semantic meaning of the text content

**Step 4: Vector Storage** (`create_vectorstore()`)
- Embeddings are stored in ChromaDB vector database
- ChromaDB provides persistent storage and efficient similarity search
- Each vector is associated with its original document chunk and metadata

**Step 5: Retrieval** (within `generate_rag_response()`)
- User submits a question via chat interface
- Question is embedded using the same embedding model
- ChromaDB's `similarity_search()` performs cosine similarity search
- Retrieves top-5 (`TOP_K_CHUNKS=5`) most relevant document chunks
- Similarity search finds chunks with highest semantic relevance to the question

**Step 6: Context Formation** (within `generate_rag_response()`)
- Retrieved chunks are formatted into a single context string
- Context includes all 5 retrieved chunks with their content
- This provides the LLM with relevant background information

**Step 7: Prompt Engineering**
- Prompt template structures the input for the LLM:
  - Role definition: "helpful assistant for question-answering tasks"
  - Instruction: Use retrieved context to answer, say "I don't know" if answer isn't in context
  - User's question
  - Retrieved context chunks
- `PromptTemplate.from_template()` creates the formatted prompt

**Step 8: Generation** (within `generate_rag_response()`)
- LLM (`ChatOpenAI` with `model="gpt-4o"`, `temperature=0.2`) generates response
- Response is grounded in the retrieved document chunks
- LLM synthesizes the retrieved information to answer the user's question
- Answer is returned to the user interface along with source citations

**Step 9: User Interface Display**
- Response is displayed in the chat interface
- Source documents are shown in an expandable section
- User can view which chunks were used to generate the response
- Conversation history is maintained in session state

### Key Design Decisions

1. **Chunk Size (1000)**: Balance between preserving context and enabling precise retrieval
2. **Chunk Overlap (200)**: Prevents information loss at boundaries, maintains continuity
3. **Top-K (5)**: Provides sufficient context without overwhelming the LLM or token budget
4. **Embedding Model**: `text-embedding-3-large` for high-quality semantic representations
5. **LLM Temperature (0.2)**: Lower temperature for more focused, deterministic responses
6. **RecursiveCharacterTextSplitter**: Intelligent splitting on natural text boundaries

---

## External Resources & Documentation

### Official Documentation

1. **LangChain Documentation**
   - URL: https://python.langchain.com/docs/
   - Usage: Primary reference for implementing RAG pipeline components
   - Specific sections used:
     - Document loaders (TextLoader, PyPDFLoader)
     - Text splitters (RecursiveCharacterTextSplitter)
     - Vector stores (Chroma)
     - Embeddings (OpenAIEmbeddings)
     - LLM integration (ChatOpenAI)

2. **Streamlit Documentation**
   - URL: https://docs.streamlit.io/
   - Usage: Building the web interface
   - Specific features used:
     - `st.file_uploader()` for multi-file uploads
     - `st.chat_message()` and `st.chat_input()` for chat interface
     - `st.sidebar()` for document upload section
     - `st.spinner()` for loading indicators
     - Session state management

3. **ChromaDB Documentation**
   - URL: https://docs.trychroma.com/
   - Usage: Understanding vector database setup and operations
   - Specific features used:
     - Vector storage and retrieval
     - Similarity search

4. **OpenAI API Documentation**
   - URL: https://platform.openai.com/docs/
   - Usage: Understanding embedding models and API usage
   - Models used:
     - `gpt-4o` for text generation
     - `text-embedding-3-large` for document embeddings

---


### Templates

1. **Template: chat_with_pdf.py**
   - Source: Assignment 1 branch of course repository
   - Usage: Starting point for basic Streamlit chat interface
   - Modifications: Complete rewrite to implement RAG pipeline

2. **Example: langgraph_chroma_retreiver.ipynb**
   - Source: Assignment 1 branch of course repository
   - Usage: Reference implementation for:
     - Document loading with LangChain
     - Text chunking strategies
     - ChromaDB vector store creation
     - Retrieval and generation workflow

3. **Configuration Files**
   - `requirements.txt`: Base dependencies provided by instructor
   - `.devcontainer`: Codespace configuration (used as-is)

---

## Code Examples & Tutorials

1. **LangChain RAG Tutorial**
   - Source: LangChain official tutorials
   - URL: https://python.langchain.com/docs/tutorials/rag/
   - Usage: Understanding RAG pipeline architecture
   - Concepts applied:
     - Document chunking best practices
     - Retrieval strategies
     - Prompt engineering for RAG

2. **Streamlit Chat Interface Examples**
   - Source: Streamlit documentation and examples
   - URL: https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps
   - Usage: Implementing chat interface with message history
   - Features adapted:
     - Chat message display
     - Session state for conversation history
     - Streaming responses

---

## Libraries & Dependencies

### Python Packages Used

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | >=1.36 | Web interface |
| langchain | 0.3.27 | RAG framework |
| langchain-openai | 0.3.35 | OpenAI integrations |
| langchain-community | 0.3.31 | Document loaders |
| langchain-chroma | >=0.1.0 | ChromaDB integration |
| langchain-text-splitters | >=0.3.0 | Text chunking |
| chromadb | >=0.4.0 | Vector database |
| openai | ~=1.14 | OpenAI API client |
| pypdf | >=4 | PDF processing |

---

## Development Environment

### Tools Used

1. **GitHub Codespaces**
   - Cloud-based development environment
   - Pre-configured with Python 3.11 and dependencies

2. **Visual Studio Code**
   - Code editor (via Codespaces browser interface)
   - Extensions: Python, Jupyter
   - Cline

3. **Git**
   - Version control
   - Working on forked repository

---

## Testing & Validation

### Test Documents

1. **provided sample files**:
   - `data/RAG_source.txt` - Used for initial testing
   - `data/combined_transcript.txt` - Used for multi-document testing

2. **Test cases verified**:
   - Single .txt file upload
   - Single .pdf file upload
   - Multiple files of mixed types
   - Large document chunking
   - Question-answering accuracy

---

### Issues Encountered

1. **Challenge**: Temporary file handling for uploaded files
   - **Solution**: Used `tempfile.NamedTemporaryFile` with proper cleanup
   - **Source**: Python documentation

2. **Challenge**: Managing state across Streamlit reruns
   - **Solution**: Used `st.session_state` for vectorstore and messages
   - **Source**: Streamlit documentation

3. **Challenge**: Displaying source citations cleanly
   - **Solution**: Used expandable sections with `st.expander()`
   - **Source**: Streamlit examples


## GenAI usage
For syntax usages, formatting and UI/CSS

---

**Last Updated**: 27th October, 2025
**Name**: Shubham Chandrodaya 
**Alias**: sc3455 
**Course**: INFO 5940 - Assignment 1



