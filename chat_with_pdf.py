import streamlit as st
import os
from openai import OpenAI
from os import environ

# LangChain imports for RAG pipeline
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import tempfile
import shutil
from typing import List


# Initial configurations for the OpenAI client and LLM
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url="https://api.ai.it.cornell.edu",
)

llm = ChatOpenAI(
    model="openai.gpt-4o",
    temperature=0.2,
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url="https://api.ai.it.cornell.edu",
)


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


TOP_K_CHUNKS = 5


def load_document(uploaded_file) -> List[Document]:
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
        
        # Add filename to metadata for source tracking
        for doc in documents:
            doc.metadata['source'] = uploaded_file.name
            
        return documents
    finally:
        # Clean up temporary file
        os.unlink(tmp_file_path)


def chunk_documents(documents: List[Document]) -> List[Document]:
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
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vectorstore(chunks: List[Document]) -> Chroma:
    """
    Create a ChromaDB vector store from document chunks.
    
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


def format_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents into a single context string.
    
    Args:
        docs: List of retrieved Document objects
        
    Returns:
        Formatted string with all document contents
    """
    return "\n\n".join(doc.page_content for doc in docs)


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
    retrieved_docs = vectorstore.similarity_search(question, k=TOP_K_CHUNKS)
    
    # Format context from retrieved documents
    context = format_docs(retrieved_docs)
    
    # Creating prompt template
    template = """You are a helpful assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer based on the context, just say that you don't know. 
Keep the answer concise and accurate.

Question: {question}

Context: {context}

Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    
    # Generating response
    messages = prompt.invoke({"question": question, "context": context})
    response = llm.invoke(messages.text)
    
    return response.content, retrieved_docs


# Streamlit UI chat interface

st.set_page_config(
    page_title="RAG Chat Application",
    layout="wide"
)

st.title("RAG-based Document Chat - SC3455")
st.markdown("Upload documents and ask questions about their content!")


# Sidebar for Document Upload

with st.sidebar:
    st.header("Document Upload")
    st.markdown("Upload one or more documents (.txt or .pdf)")
    
    # File uploader supporting multiple files
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        help="Upload .txt or .pdf files to chat with"
    )
    
    # Display uploaded files
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded")
        with st.expander("View uploaded files"):
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size} bytes)")
    
    # Process documents button
    process_button = st.button("Process Documents", type="primary", use_container_width=True)
    
    # Show processing status
    if 'vectorstore' in st.session_state and st.session_state.vectorstore is not None:
        st.success("Documents processed and ready!")
        if 'total_chunks' in st.session_state:
            st.info(f"Total chunks: {st.session_state.total_chunks}")
    
    # Settings expander
    with st.expander("Settings"):
        st.markdown(f"**Chunk Size:** {CHUNK_SIZE}")
        st.markdown(f"**Chunk Overlap:** {CHUNK_OVERLAP}")
        st.markdown(f"**Retrieval K:** {TOP_K_CHUNKS}")

# Document Processing

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'total_chunks' not in st.session_state:
    st.session_state.total_chunks = 0

# Processing documents flow when button is clicked
if process_button and uploaded_files:
    with st.spinner("Processing documents... This may take a moment."):
        try:
            all_documents = []
            
            # Load all uploaded files
            for uploaded_file in uploaded_files:
                docs = load_document(uploaded_file)
                all_documents.extend(docs)
            
            # Chunk documents
            chunks = chunk_documents(all_documents)
            st.session_state.total_chunks = len(chunks)
            
            # Create vector store
            st.session_state.vectorstore = create_vectorstore(chunks)
            
            # Clear previous chat messages when new documents are processed
            st.session_state.messages = []
            
            st.success(f"Successfully processed {len(uploaded_files)} document(s) into {len(chunks)} chunks!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

# Chat Interface

st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View Sources"):
                for i, doc in enumerate(message["sources"], 1):
                    source = doc.metadata.get('source', 'Unknown')
                    st.markdown(f"**Source {i}:** {source}")
                    st.text(doc.page_content[:200] + "...")

# Chat input
if st.session_state.vectorstore is not None:
    question = st.chat_input("Ask a question about your documents...")
    
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
                    with st.expander("View Sources"):
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
    # Show instructions when no documents are uploaded
    st.info("Please upload documents using the sidebar and click 'Process Documents' to start chatting!")

# Footer

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    Built with Streamlit, LangChain, and ChromaDB
    </div>
    """,
    unsafe_allow_html=True
)
