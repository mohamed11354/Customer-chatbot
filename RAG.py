from typing import List, Union
import requests
from langchain_community.document_loaders import WebBaseLoader, pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_together import TogetherEmbeddings
from langchain.docstore.document import Document
import os

class RAGManager:
    """Retrieval-Augmented Generation (RAG) Manager for handling document ingestion and indexing."""

    def __init__(self, embedding_model: str = "togethercomputer/m2-bert-80M-8k-retrieval"):
        """
        Initialize the RAG Manager with an embedding model.
        :param embedding_model: Name of the embedding model
        """
        self.embeddings = TogetherEmbeddings(model=embedding_model)
        self.vector_store = None  # Lazy initialization of FAISS store
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    def load_documents(self, sources: Union[str, List[str]]) -> List[Document]:
        """
        Load documents from URLs or text file paths.
        :param sources: A URL (str) or a list of text file paths (List[str]).
        :return: List of Document objects.
        """
        documents = []
        
        if isinstance(sources, str):  # Single URL or file
            sources = [sources]

        for source in sources:
            if source.startswith("http"):  # Load from URL
                loader = WebBaseLoader(source)
            elif os.path.exists(source):  # Load from a text file
                loader = pdf.PDFMinerLoader(source)
            else:
                raise ValueError(f"Invalid source: {source}. Must be a URL or a valid file path.")

            documents.extend(loader.load())

        return documents
    
    def process_and_index(self, sources: Union[str, List[str]]):
        """
        Load, split, and index documents into the FAISS vector store.
        :param sources: A URL or a list of document file paths.
        """
        docs = self.load_documents(sources)
        split_docs = self.text_splitter.split_documents(docs)

        # Create or update the FAISS index
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        else:
            self.vector_store.add_documents(split_docs)
    
    def search(self, query: str, k: int = 4):
        """
        Perform a similarity search on the vector store.
        :param query: The search query.
        :param k: Number of results to return.
        :return: List of relevant document chunks.
        """
        if not self.vector_store:
            raise RuntimeError("Vector store is empty. Please add documents first.")

        results = self.vector_store.similarity_search(query, k=k)
        return [res.page_content for res in results]


rag = RAGManager()

# Test load url
rag.process_and_index("https://cp-algorithms.com/graph/topological-sort.html")
results = rag.search("What is the main topic?", k = 1)
print(results)

# Test load document
rag.process_and_index("seamcarving.pdf")
results = rag.search("what are seams?", k = 1)
print(results)
    