import os
from dotenv import load_dotenv
import streamlit as st
import langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI 
from langchain.vectorstores import FAISS  
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import faiss

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# ---- Step 1: Load and Process Documents ----
def load_and_split_documents(file_path):
    # Load text from a file
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    return chunks

# ---- Step 2: Embed the Chunks and Create Vector Database ----
def create_vector_db(chunks, persist_directory="faiss_db"):
    # Generate embeddings using OpenAI
    embeddings = OpenAIEmbeddings()

    # Initialize FAISS index
    faiss_index = FAISS.from_documents(chunks, embeddings)

    # Optionally, you can save the index to a file
    faiss.write_index(faiss_index.index, persist_directory)

    return faiss_index

# ---- Step 3: Build RAG Pipeline ----
def build_rag_pipeline(vector_db):
    # Load the FAISS index and create a retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 matches

    # Initialize the OpenAI GPT model
    gpt_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)

    # Combine retriever with the model in a QA pipeline
    qa_chain = RetrievalQA.from_chain_type(llm=gpt_model, retriever=retriever, return_source_documents=True)

    return qa_chain

# ---- Streamlit App ----
def main():
    st.title("RAG Application with Streamlit")
    st.sidebar.header("Upload and Query")

    # upload the text document
    uploaded_file = st.sidebar.file_uploader("Upload a Text Document", type=["txt"])
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        file_path = f"{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.sidebar.success(f"File '{uploaded_file.name}' uploaded successfully!")

        # Process the uploaded document
        st.header("Processing Document...")
        chunks = load_and_split_documents(file_path)
        st.success(f"Document processed into {len(chunks)} chunks.")

        # Embed chunks and build vector database
        st.header("Building Vector Database...")
        vector_db = create_vector_db(chunks)
        st.success("Vector database created and persisted.")

        # Build RAG pipeline
        st.header("Initializing RAG Pipeline...")
        rag_pipeline = build_rag_pipeline(vector_db)
        st.success("RAG pipeline ready!")

        # Query section
        st.header("Ask a Question")
        query = st.text_input("Enter your question:", placeholder="Type your question here...")

        if st.button("Submit") and query:
            with st.spinner("Fetching answer..."):
                response = rag_pipeline.invoke({"query": query})
                answer = response["result"]
                sources = response["source_documents"]

                st.write("### Answer:")
                st.write(answer)

                st.write("### Sources:")
                unique_sources = set() 
                for source in sources:
                    source_metadata = source.metadata.get('source', 'Unknown')
                    unique_sources.add(source_metadata)  

                for source in unique_sources:
                    st.write(f"- {source}")

if __name__ == "__main__":
    main()
