# Import required libraries
import os
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from anon import anonymize_text


# Function to load, split, and embed data from PDF documents into Chroma vector store
def process_documents(docs):
    """
    Process PDF documents through loading, splitting, and embedding.
    Returns vector store instance.
    """
    # Create temporary directory for PDF storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded files to temp directory after initially anonymising them with langchain
        doc_paths = []
        for doc in docs:
            path = os.path.join(temp_dir, doc.name)

            with open(path, "wb") as f:
                f.write(doc.getbuffer())
            doc_paths.append(path)
        
        # Load the documents
        documents = []
        for path in doc_paths:
            print(f"Loading document: {path}")
            if path.endswith(".pdf"):
                loader = PDFPlumberLoader(path)
            else: # Handle other document types here
                #load doc and docx files
                loader = Docx2txtLoader(path) 
        # Anonymise the documents using langchain's anonymise method by looping through each document and calling the anonymise method
        loaded_documents = loader.load()
        for doc in loaded_documents:
            doc.page_content = anonymize_text(doc.page_content)
        anonymized_documents = loaded_documents
        documents.extend(anonymized_documents)
        # Split documents into chunks using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  
            chunk_overlap=150  
        )
        splits = text_splitter.split_documents(documents)
        
        # Instantiate the embeddings model
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        # if a chroma db already exists, load it and add the new documents
        if os.path.exists("./chroma_db"):
            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory="./chroma_db"
            )
            vector_store.add_documents(splits)
        # Create embeddings and vector store
        else:
            vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        return vector_store

# Initialize and returns a retriever for the vector store, which will be used to fetch relevant chunks from the stored embeddings based on user queries. 
def get_retriever():
    """Initialize and return the vector store retriever"""
    # Initialize the embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    try:
        # Initialize the vector store
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )

        # Return the retriever with MMR (Maximum Marginal Relevance) search and k=3
        return vector_store.as_retriever(search_type="MMR", search_kwargs={"k": 3})

    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return None
    