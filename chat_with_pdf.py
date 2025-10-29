import streamlit as st
import os
from openai import OpenAI
from os import environ

# imports for RAG pipeline components
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import tempfile
import shutil
from typing import List


# Initial configurations for the OpenAI client and model
openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url="https://api.ai.it.cornell.edu",
)

# Primary chat model used for responses
conversational_model = ChatOpenAI(
    model="openai.gpt-4o",
    temperature=0.2,
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url="https://api.ai.it.cornell.edu",
)


CHUNK_CHARACTER_LIMIT = 1000
CHUNK_CHARACTER_OVERLAP = 200


TOP_K_RETRIEVAL = 5


def load_uploaded_document(uploaded_file) -> List[Document]:
    """
    Load a document from an uploaded file.
    
    Supports .txt and .pdf file formats. Creates a temporary file to handle
    the upload and uses appropriate LangChain loader based on file type.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        List of LangChain Document objects
    """
    # Create a temporary file to save the uploaded content
    suffix = os.path.splitext(uploaded_file.name)[1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Load document based on file type
        if uploaded_file.name.endswith('.txt'):
            loader = TextLoader(tmp_file_path)
        elif uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        else:
            raise ValueError(f"Unsupported file type: {uploaded_file.name}")
        
        documents = loader.load()
        
        # Adding filename to metadata for source tracking
        for doc in documents:
            doc.metadata['source'] = uploaded_file.name
            
        return documents
    finally:
        os.unlink(tmp_file_path)


def split_into_chunks(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for efficient retrieval.
    
    Uses RecursiveCharacterTextSplitter to intelligently split text while
    maintaining context through overlapping chunks.
    
    Args:
        documents: List of LangChain Document objects
        
    Returns:
        List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARACTER_LIMIT,
        chunk_overlap=CHUNK_CHARACTER_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """
    ChromaDB vector store from document chunks.
    
    Embeds all chunks using OpenAI embeddings and stores them in ChromaDB
    for efficient similarity search.
    
    Args:
        chunks: List of chunked Document objects
        
    Returns:
        Chroma vectorstore object
    """
    embeddings = OpenAIEmbeddings(
        model="openai.text-embedding-3-large",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url="https://api.ai.it.cornell.edu",
    )
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    return vectorstore


def join_documents(docs: List[Document]) -> str:
    """
    Format retrieved documents into a single context string.
    
    Args:
        docs: List of retrieved Document objects
        
    Returns:
        Formatted string with all document contents
    """
    return "\n\n".join(doc.page_content for doc in docs)


def ensure_session_defaults() -> None:
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'total_chunks' not in st.session_state:
        st.session_state.total_chunks = 0
    if 'document_names' not in st.session_state:
        st.session_state.document_names = []
    if 'num_documents' not in st.session_state:
        st.session_state.num_documents = 0


def generate_rag_response(question: str, vectorstore: Chroma) -> tuple[str, List[Document]]:
    """
    Generate a response using the RAG pipeline.
    
    Retrieves relevant document chunks and uses them as context for
    generating an accurate, grounded response.
    
    Args:
        question: User's question
        vectorstore: ChromaDB vectorstore containing document chunks
        
    Returns:
        Tuple of (response text, list of source documents)
    """
    # Retrieve relevant chunks
    retrieved_docs = vectorstore.similarity_search(question, k=TOP_K_RETRIEVAL)
    
    # Format context from retrieved documents
    context = join_documents(retrieved_docs)
    
    # Compute document inventory from session (if available)
    document_names = st.session_state.get("document_names", [])
    num_documents = st.session_state.get("num_documents", len(set([d.metadata.get("source", "Unknown") for d in retrieved_docs])))
    doc_list_str = ", ".join(document_names) if document_names else ", ".join(sorted(set([d.metadata.get("source", "Unknown") for d in retrieved_docs])))

    # Creating prompt template per requested guidance
    template = """
You are a careful assistant that answers ONLY from the retrieved context.

RULES
1) Grounding: Use information ONLY from “Retrieved Snippets” below. Do NOT rely on outside knowledge.
2) Citations: After any non-trivial fact, add citation markers like [S1], [S3]. If multiple snippets support a statement, cite them all, e.g., [S2,S5].
3) Conflicts: If snippets conflict, prefer the one that is (i) more specific, (ii) directly on-topic, and (iii) later/explicitly dated. Briefly note the discrepancy.
4) Gaps: If the answer is not in the snippets, say: “I can’t find this in the provided documents.” Optionally suggest ONE precise follow-up the user could ask.
5) Multi-doc: If multiple sources contribute, name all of the sources that contributed to the answer with references.
6) Numbers & small calculations: Quote numbers verbatim; show any tiny calculation inline (e.g., “12 + 8 = 20”). Never invent numbers.
7) Definitions: Define a term only if a definition appears in snippets; otherwise state that the term isn’t defined in the provided documents.
8) Lists/Counts: If asked to list or count, report exactly what appears in the snippets and say if the list may be partial.
9) Tone/Length: Be concise (3–7 sentences unless the user asked for more). Use Markdown.
10) When asked to summarize, summarize the content from both documents in a numbered list, unless asked to summarize from a specific document.

DOCUMENT INVENTORY
- Total documents: {num_documents}
- Document names: {document_names}

QUESTION
{question}

RETRIEVED SNIPPETS
Numbered S1..Sk in the order given. Each snippet may include its source name and page if available.

{context}

RESPONSE FORMAT
- **Answer:** concise answer with inline [S#] citations.
- **Sources:** bullet list “S# — <source name> (page if known)” for every S# you cited, in order of first appearance.
"""
    
    prompt = PromptTemplate.from_template(template)
    
    # Generating output
    messages = prompt.invoke({
        "question": question,
        "context": context,
        "num_documents": num_documents,
        "document_names": doc_list_str,
    })
    response = conversational_model.invoke(messages.text)
    
    return response.content, retrieved_docs


# Streamlit UI chat interface

st.set_page_config(
    page_title="RAG Chat Application",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2d3748 0%, #4a5568 100%);
        padding: 2rem;
        border-radius: 10px;
        color: #e2e8f0;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #4a5568;
    }
    
    .main-header h1 {
        color: #e2e8f0;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: #cbd5e0;
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        padding: 1rem;
        border-radius: 8px;
        color: #e2e8f0;
        margin-bottom: 1rem;
        border: 1px solid #4a5568;
    }
    
    .sidebar-header h3 {
        color: #e2e8f0;
        margin: 0;
        font-size: 1.3rem;
    }
    
    .status-card {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        padding: 1rem;
        border-radius: 8px;
        color: #e2e8f0;
        margin: 1rem 0;
        border: 1px solid #4a5568;
    }
    
    .settings-card {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #68d391;
        margin: 1rem 0;
        color: #e2e8f0;
        border: 1px solid #4a5568;
    }
    
    .chat-container {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        margin: 1rem 0;
        border: 1px solid #4a5568;
        color: #e2e8f0;
    }
    
    .footer {
        background: linear-gradient(90deg, #2d3748 0%, #4a5568 100%);
        padding: 1rem;
        border-radius: 8px;
        color: #e2e8f0;
        text-align: center;
        margin-top: 2rem;
        border: 1px solid #4a5568;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #68d391 0%, #48bb78 100%);
        color: #1a202c;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(104, 211, 145, 0.4);
    }
    
    .stExpander {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        border-radius: 8px;
        border: 1px solid #4a5568;
        color: #e2e8f0;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        color: #68d391;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #68d391;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        color: #63b3ed;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #63b3ed;
    }
    
    .stError {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        color: #fc8181;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #fc8181;
    }
    
    .main .block-container {
        background: #1a202c;
        padding: 2rem;
        border-radius: 10px;
        color: #e2e8f0;
    }
    
    .stApp {
        background: #1a202c;
    }
    
    .stMarkdown {
        color: #e2e8f0;
    }
    
    .stTextInput > div > div > input {
        background: #2d3748;
        color: #e2e8f0;
        border: 1px solid #4a5568;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #68d391;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>Document Chatbot</h1>
    <p>Upload documents and ask questions about their content with AI-powered RAG based document retrieval</p>
</div>
""", unsafe_allow_html=True)


# Centered uploader section (replaces sidebar)
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    st.markdown("""
    <div class="sidebar-header">
        <h3>Document Upload</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Upload one or more documents (.txt or .pdf)**")
    
    # File uploader supporting multiple files
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        help="Upload .txt or .pdf files to chat with"
    )
    
    # Display uploaded files
    if uploaded_files:
        st.markdown(f"""
        <div class="status-card">
            <strong>{len(uploaded_files)} file(s) uploaded successfully!</strong>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("View uploaded files", expanded=False):
            for file in uploaded_files:
                st.write(f"{file.name} ({file.size:,} bytes)")
    
    # Upload documents button
    process_button = st.button("Upload Documents", type="primary", use_container_width=True)
    
    # Show processing status
    if 'vectorstore' in st.session_state and st.session_state.vectorstore is not None:
        st.markdown(f"""
        <div class="status-card">
            <strong>Documents processed and ready!</strong>
        </div>
        """, unsafe_allow_html=True)
        
        if 'total_chunks' in st.session_state:
            st.markdown(f"""
            <div class="status-card">
                <strong>Total chunks: {st.session_state.total_chunks}</strong>
            </div>
            """, unsafe_allow_html=True)
    
    # Settings expander
    with st.expander("Configuration Settings", expanded=False):
        st.markdown("""
        <div class="settings-card">
            <strong>Current Settings</strong><br><br>
            <strong>Chunk Size:</strong> {chunk_size}<br>
            <strong>Chunk Overlap:</strong> {chunk_overlap}<br>
            <strong>Retrieval K:</strong> {top_k}
        </div>
        """.format(chunk_size=CHUNK_CHARACTER_LIMIT, chunk_overlap=CHUNK_CHARACTER_OVERLAP, top_k=TOP_K_RETRIEVAL), 
        unsafe_allow_html=True)

# Document Processing

# Initialize session state
ensure_session_defaults()

# Processing documents flow when button is clicked
if process_button and uploaded_files:
    with st.spinner("Processing documents... This may take a moment."):
        try:
            all_documents = []
            
            # Load all uploaded files
            for uploaded_file in uploaded_files:
                docs = load_uploaded_document(uploaded_file)
                all_documents.extend(docs)
            
            # Chunk documents
            chunks = split_into_chunks(all_documents)
            st.session_state.total_chunks = len(chunks)
            
            # Create vector store
            st.session_state.vectorstore = build_vectorstore(chunks)
            
            # Persist document inventory for future questions
            unique_sources = sorted({doc.metadata.get('source', 'Unknown') for doc in all_documents})
            st.session_state.document_names = unique_sources
            st.session_state.num_documents = len(unique_sources)
            
            # Clear previous chat messages when new documents are processed
            st.session_state.messages = []
            
            st.markdown(f"""
            <div class="status-card">
                <strong>Successfully processed {len(uploaded_files)} document(s) into {len(chunks)} chunks!</strong>
            </div>
            """, unsafe_allow_html=True)
            st.rerun()
            
        except Exception as e:
            st.markdown(f"""
            <div class="status-card" style="background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); color: #991b1b;">
                <strong>Error processing documents: {str(e)}</strong>
            </div>
            """, unsafe_allow_html=True)

# Chat Interface

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View Sources", expanded=False):
                for i, doc in enumerate(message["sources"], 1):
                    source = doc.metadata.get('source', 'Unknown')
                    st.markdown(f"**Source {i}:** {source}")
                    st.text(doc.page_content[:200] + "...")

# Chat input
if st.session_state.vectorstore is not None:
    question = st.chat_input("enter your query")
    
    if question:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response, sources = generate_rag_response(
                        question, 
                        st.session_state.vectorstore
                    )
                    st.markdown(response)
                    
                    # Show sources
                    with st.expander("View Sources", expanded=False):
                        for i, doc in enumerate(sources, 1):
                            source = doc.metadata.get('source', 'Unknown')
                            st.markdown(f"**Source {i}:** {source}")
                            st.text(doc.page_content[:200] + "...")
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
else:
    pass