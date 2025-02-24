import os
from langchain_chroma import Chroma
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from utils import process_documents, get_retriever
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders.telegram import text_to_docs as Loader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document


# Custom prompt template
def get_custom_prompt():
    # Define and return the custom prompt template if chroma db has been loaded then use prompt that forces the context to be used
    if st.session_state.vector_store:
        """Define and return the custom prompt template."""
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
            "You are a consultants assistant designed to help experts make recommendations based on knowledge from past psychology reports. Follow these guidelines:\n"
"1. Answer questions using only quotes and extracts from the uploaded documents.\n"
"2. If the answer isn't in the documents, say: 'I cannot find relevant information in the provided documents.'\n"
"3. Do not speculate, assume, or invent information.\n"
"4. Maintain a professional tone and organize responses clearly.\n"
"5. Keep answers concise, focused, and relevant to the question."
        ),
        HumanMessagePromptTemplate.from_template(
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Provide a 4 to 5 paragraph answer based on the context above that is precise and well-structured. "
        )
    ])
    else:
        return ChatPromptTemplate.from_messages(
    [("user", "Write a 4 or 5 paragraph summary of the following context suitable for a consultants report covering all areas of the context. Use academic and professional language and avoid including your thought process but give a clear explanation of all points referencing the relevant details from the context :\\n\\n{context}")]
)

# Initialize QA Chain
def initialize_qa_chain():
    if not st.session_state.qa_chain and st.session_state.vector_store:
        llm = ChatOllama(model="deepseek-r1:7b", temperature=0.1)
        retriever = get_retriever()

        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": get_custom_prompt()}
        )
    else:
        llm = ChatOllama(model="deepseek-r1:7b", temperature=0.6)
        st.session_state.qa_chain = create_stuff_documents_chain(
            llm,
            prompt=get_custom_prompt()
        )
    return st.session_state.qa_chain

# Initialize the chatbot's memory (session states)
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None




# Chat interface section
def chat_interface():
    st.title("Psych.ai")
    st.markdown("Your personal textbook AI chatbot powered by Deepseek 7B")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle user input
    if prompt := st.chat_input("Ask about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            
            with st.spinner("Fetching information..."):
                if st.session_state.vector_store: 
                    try:
                        qa_chain = initialize_qa_chain()
                        response = qa_chain.invoke({"query": prompt})
                        full_response = response["result"]
                    except Exception as e:
                        full_response = f"Error: {str(e)}"
                else:
                    try:
                        sum_chain = initialize_qa_chain()
                        #create Document object
                        print("prompt",prompt)
                        docs = Loader(prompt)
                        response = sum_chain.invoke({"context": docs})
                        full_response = response
                    except Exception as e:
                        full_response = f"Error: {str(e)}"
                        raise e
            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Main function
def main():
    initialize_session_state()
    with st.sidebar:
        # Instructions
        st.markdown("### Instructions")
        st.info("""
        1. Upload PDF documents.
        2. Click 'Create Knowledge Base'.
        3. Once documents are processed, start chatting with the bot!
        """)
        
        # Streamlit file uploader widget for pdfs and doc files
        docs = st.file_uploader(
            "Upload PDF or DOC documents", 
            type=["pdf", "doc", "docx"],
            accept_multiple_files=True  # Allow multiple file uploads
        )
        
        # Action Button for user to kick off the knowledge base creation process
        # Action Button for user to kick off the knowledge base creation process
        if st.button("Create or Load Knowledge Base"):
            if not docs:
                # load existing chroma db if it exists
                if os.path.exists("./chroma_db"):
                    st.session_state.vector_store = Chroma(persist_directory="./chroma_db")
                    st.session_state.qa_chain = None  # Reset QA chain when new documents are processed
                    st.success("Knowledge base loaded!")
                else:
                    st.error("Please upload documents first.")
                    return
            else:
                try:
                    with st.spinner("Creating knowledge base... This may take a moment."):
                        vector_store = process_documents(docs)
                        st.session_state.vector_store = vector_store
                        st.session_state.qa_chain = None  # Reset QA chain when new documents are processed

                    st.success("Knowledge base created!")  # Simple success message after completion

                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")  # Show error if something goes wrong
                    raise e
        if st.button("Unload context"):
            st.session_state.vector_store = None
            st.session_state.qa_chain = None
            st.success("Context unloaded!")
    chat_interface()
   
 


if __name__ == "__main__":
    main()