# INFO 5940 - Assignment 1: RAG-based Document Chat Application

## Quick Start Testing

### Run the Application

**Method 1: Set API key in command**
```bash
export OPENAI_API_KEY="your_actual_API_KEY" 
export OPENAI_BASE_URL="https://api.ai.it.cornell.edu" 
streamlit run chat_with_pdf.py
```

**Method 2: Set API key in devcontainer.json (for local testing)**
1. Edit `.devcontainer/devcontainer.json` and add your API key to `remoteEnv` section
2. Rebuild container
3. Run: `streamlit run chat_with_pdf.py`

###  Test
1. Upload `data/RAG_source.txt` using the sidebar
2. Click "Process Documents"
3. Ask: "What is Zelomax?"
4. Verify response with source citations appear

### Test Scenarios

**Test 1: Single .txt File** 
- Upload `data/RAG_source.txt` - Click "Process Documents" - Ask "What is Zelomax?"
- Expected: Response with source citations

**Test 2: Single .pdf File** 
- Upload a .pdf file - Click "Process Documents" - Ask questions about PDF content
- Expected: PDF text extracted and queryable

**Test 3: Multiple Documents** 
- Upload both `data/RAG_source.txt` and `data/combined_transcript.txt` - Process
- Expected: Can query across all uploaded documents, sources show different files

**Test 4: Document Chunking**
- Upload large document - Check sidebar shows "Total chunks: X"
- Expected: Documents split into multiple chunks for efficient retrieval

**Test 5: RAG Retrieval**
- Upload document -  Ask specific question -  Click "View Sources"
- Expected: Retrieved chunks are relevant, response is accurate and grounded

**Test 6: Conversational Interface**
- Upload documents - Ask multiple questions in sequence
- Expected: Chat history maintained, clear UI feedback, source citations per response

## Changes from Provided Template

### Modified Files

1. **requirements.txt**
   - Added `chromadb>=0.4.0` for vector storage
   - Added `langchain-chroma>=0.1.0` for ChromaDB integration
   - Added `langchain-text-splitters>=0.3.0` for document chunking

2. **chat_with_pdf.py**
   - Complete rewrite implementing full RAG pipeline
   - Added multi-file upload support
   - Added PDF processing capability
   - Implemented document chunking with RecursiveCharacterTextSplitter
   - Integrated ChromaDB vector store
   - Added retrieval and generation pipeline
   - Enhanced UI with source citations and better feedback

### No Changes to `.devcontainer`

## Overview
This application is a Retrieval-Augmented Generation (RAG) system that allows users to upload documents (.txt and .pdf files) and interact with their content through a conversational interface. The application uses LangChain for the RAG pipeline, ChromaDB for vector storage, and Streamlit for the user interface.

## Features
- **Multiple File Upload**: Upload multiple .txt and .pdf documents simultaneously
- **Document Chunking**: Efficiently processes large documents by breaking them into manageable chunks
- **Vector Storage**: Uses ChromaDB with OpenAI embeddings for fast similarity search
- **RAG Pipeline**: Retrieves relevant document chunks and generates accurate, context-based responses
- **Conversational Interface**: Chat-style UI with message history and source citations
- **Source Tracking**: View which document chunks were used to generate each response



## Detailed Setup Instructions

### Prerequisites
- GitHub Codespaces (recommended) or local Python 3.11+ environment
- OpenAI API key


### Running in GitHub Codespaces
1. Fork this repository - Open Codespace
2. Wait for environment setup (auto-installs dependencies)
3. Set API key using Method 1 or 2 from Quick Start section above
4. Access app via popup or refresh browser

### Running Locally
1. Clone repository and `cd` into it
2. Install dependencies: `pip install -r requirements.txt`
3. Set API key and run using Quick Start commands above

## Usage Guide

### Uploading Documents

1. Use the **sidebar** on the left to upload files
2. Click the file uploader and select one or more .txt or .pdf files
3. Click **"Process Documents"** to create the vector database
4. Wait for the success message

### Asking Questions

1. Once documents are processed, use the **chat input** at the bottom
2. Type your question and press Enter
3. The assistant will retrieve relevant information and generate a response
4. Click **"View Sources"** to see which document chunks were used

### Best Practices

- **Document Size**: The app handles large files by chunking them automatically
- **Multiple Documents**: You can upload and query across multiple documents simultaneously
- **Specific Questions**: Ask specific questions for better results
- **Reprocessing**: Click "Process Documents" again to add or replace documents

## Configuration

### Chunking Parameters (in `chat_with_pdf.py`)
```python
CHUNK_SIZE = 1000        # Size of each text chunk
CHUNK_OVERLAP = 200      # Overlap between chunks
```

### Retrieval Parameters
```python
TOP_K_CHUNKS = 5         # Number of chunks to retrieve per query
```

### Model Configuration
```python
model="openai.gpt-4o"    # LLM for response generation
temperature=0.2          # Lower = more focused responses
embedding="text-embedding-3-large"  # Embedding model
```

## File Structure

```
INFO-5940-Codespace/
├── chat_with_pdf.py              # Main application (RAG implementation)
├── langgraph_chroma_retreiver.ipynb  # Example notebook
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── ref-log.md                    # Reference log
├── data/                         # Sample data files
│   ├── combined_transcript.txt
│   └── RAG_source.txt
└── .devcontainer/                # Codespace configuration
```

## Technical Details

### LangChain Components Used
- `TextLoader`: For loading .txt files
- `PyPDFLoader`: For loading .pdf files
- `RecursiveCharacterTextSplitter`: For chunking documents
- `OpenAIEmbeddings`: For creating vector embeddings
- `Chroma`: For vector storage and retrieval
- `ChatOpenAI`: For LLM-based response generation
- `PromptTemplate`: For formatting prompts

### RAG Pipeline Flow
```
User Upload - Document Loading - Chunking - Embedding - ChromaDB Storage
                                                              ↓
User Query - Vector Search (Retrieval) - Context Formation - LLM Generation - Response
```