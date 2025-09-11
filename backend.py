import os
import tempfile
import gc
import time
from functools import wraps
from typing import List, Optional
from dotenv import load_dotenv

# Updated LangChain imports for current versions
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Initialize LLM with error handling
def initialize_llm():
    """Initialize Google Generative AI LLM with proper error handling"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required")
        if len(api_key) < 20:  # Basic validation
            raise ValueError("Invalid API key format")  
              
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        return llm
    except Exception as e:
        raise Exception(f"Failed to initialize LLM: {str(e)}")

# Initialize embeddings with error handling
def initialize_embeddings():
    """Initialize HuggingFace embeddings with proper error handling"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        raise Exception(f"Failed to initialize embeddings: {str(e)}")

# Initialize components
try:
    llm = initialize_llm()
    embeddings = initialize_embeddings()
except Exception as e:
    print(f"Initialization error: {e}")
    llm = None
    embeddings = None

# Cleanup of vector stores, embeddings stay in memory
def cleanup_resources():
    if 'vectorstore' in globals():
        del vectorstore
    if 'embeddings' in globals():
        del embeddings
    gc.collect()

# ✅ Health Checks
def health_check():
    """Check if all systems are working"""
    try:
        # Test LLM
        if not llm:
            return {"status": "unhealthy", "reason": "LLM not initialized"}
        
        # Test simple query
        test_response = llm.invoke("test")
        if not test_response:
            return {"status": "unhealthy", "reason": "LLM not responding"}
        
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)}

# Enhanced RAG prompt template
template = """You are a helpful AI assistant that provides accurate and informative responses.

Use the following context to answer the question. If the context doesn't contain enough information to answer completely, say so and provide what information you can based on your general knowledge.

Context:
{context}

Question: {question}

Instructions:
- Be concise but comprehensive
- Cite relevant parts of the context when applicable
- If context is insufficient, clearly state this limitation
- Provide helpful and accurate information like Google Gemini would

Answer:"""

prompt = PromptTemplate.from_template(template)

def load_and_chunk_documents(file_paths: List[str]) -> List[Document]:
    """
    Load and chunk documents from file paths with improved error handling
    
    Args:
        file_paths: List of file paths to process
        
    Returns:
        List of chunked Document objects
    """
    if not file_paths:
        return []
    
    docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased chunk size for better context
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    for path in file_paths:
        try:
            if not os.path.exists(path):
                print(f"Warning: File {path} does not exist")
                continue
                
            if path.lower().endswith('.pdf'):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path, encoding='utf-8')
            
            # Load documents
            raw_docs = loader.load()
            if not raw_docs:
                print(f"Warning: No content loaded from {path}")
                continue
            
            # Split documents
            chunked = splitter.split_documents(raw_docs)
            
            # Add metadata
            for doc in chunked:
                doc.metadata["source"] = os.path.basename(path)
                doc.metadata["chunk_id"] = len(docs)
            
            docs.extend(chunked)
            
        except Exception as e:
            print(f"Error processing file {path}: {str(e)}")
            continue
    
    return docs

def create_vectorstore(docs: List[Document]) -> Optional[FAISS]:
    """
    Create FAISS vectorstore from documents
    
    Args:
        docs: List of documents to index
        
    Returns:
        FAISS vectorstore or None if creation fails
    """
    if not docs or not embeddings:
        return None
    
    try:
        # Create FAISS vectorstore
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore
    except Exception as e:
        print(f"Error creating vectorstore: {str(e)}")
        return None

def create_hybrid_retriever(docs: List[Document], vectorstore: Optional[FAISS], k: int = 5):
    """
    Create hybrid retriever combining BM25 and vector search
    
    Args:
        docs: List of documents
        vectorstore: FAISS vectorstore
        k: Number of documents to retrieve
        
    Returns:
        Retriever object
    """
    retrievers = []
    weights = []
    
    # Add BM25 retriever if documents available
    if docs:
        try:
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = k
            retrievers.append(bm25_retriever)
            weights.append(0.5)
        except Exception as e:
            print(f"Warning: Could not create BM25 retriever: {str(e)}")
    
    # Add vector retriever if vectorstore available
    if vectorstore:
        try:
            vector_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            retrievers.append(vector_retriever)
            weights.append(0.7 if len(retrievers) == 1 else 0.5)
        except Exception as e:
            print(f"Warning: Could not create vector retriever: {str(e)}")
    
    # Create ensemble retriever if we have multiple retrievers
    if len(retrievers) > 1:
        try:
            ensemble_retriever = EnsembleRetriever(
                retrievers=retrievers,
                weights=weights
            )
            return ensemble_retriever
        except Exception as e:
            print(f"Warning: Could not create ensemble retriever: {str(e)}")
            return retrievers[0] if retrievers else None
    elif retrievers:
        return retrievers[0]
    else:
        return None

def get_response(query: str, file_paths: List[str] = None) -> str:
    """
    Generate response using RAG pipeline with comprehensive error handling
    
    Args:
        query: User question
        file_paths: Optional list of file paths for context
        
    Returns:
        Generated response string
    """
    if not llm:
        return "❌ Error: Language model not properly initialized. Please check your GOOGLE_API_KEY."
    
    if not query.strip():
        return "Please provide a valid question."
    
    try:
        # Process documents if provided
        docs = []
        vectorstore = None
        
        if file_paths:
            docs = load_and_chunk_documents(file_paths)
            if docs:
                vectorstore = create_vectorstore(docs)
        
        # If no documents provided or processed, use LLM directly
        if not docs:
            try:
                response = llm.invoke(query)
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                return f"❌ Error generating response: {str(e)}"
        
        # Create retriever for RAG
        retriever = create_hybrid_retriever(docs, vectorstore)
        
        if not retriever:
            # Fallback to direct LLM response with document context
            context = "\n\n".join([doc.page_content for doc in docs[:3]])
            enhanced_query = f"Context: {context}\n\nQuestion: {query}"
            try:
                response = llm.invoke(enhanced_query)
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                return f"❌ Error with fallback response: {str(e)}"
        
        # Create RAG chain
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            # Generate response
            result = qa_chain.invoke({"query": query})
            
            # Extract response
            if isinstance(result, dict):
                response_text = result.get('result', 'No response generated')
                source_docs = result.get('source_documents', [])
                
                # Add source information if available
                if source_docs:
                    sources = set([doc.metadata.get('source', 'Unknown') for doc in source_docs])
                    if sources and sources != {'Unknown'}:
                        response_text += f"\n\n📚 *Sources: {', '.join(sources)}*"
                
                return response_text
            else:
                return str(result)
                
        except Exception as e:
            return f"❌ Error in RAG pipeline: {str(e)}"
    
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

# Test function for debugging
def test_setup():
    """Test the setup and return status"""
    status = {
        "llm_initialized": llm is not None,
        "embeddings_initialized": embeddings is not None,
        "google_api_key_set": bool(os.getenv("GOOGLE_API_KEY"))
    }
    return status

if __name__ == "__main__":
    # Test the setup when running directly
    status = test_setup()
    print("Backend Setup Status:")
    for key, value in status.items():
        print(f"  {key}: {'✓' if value else '✗'}")
    
    if all(status.values()):
        print("\n✅ Backend setup successful!")
        # Test with a simple query
        test_response = get_response("Hello, how are you?")
        print(f"Test response: {test_response[:100]}...")
    else:
        print("\n❌ Backend setup incomplete. Please check your configuration.")