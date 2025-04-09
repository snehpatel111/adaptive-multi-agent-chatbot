import os
from dotenv import load_dotenv
load_dotenv()
import warnings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from wikipediaapi import Wikipedia
from uuid import uuid4
import chromadb

warnings.filterwarnings('ignore', category=RuntimeWarning)

class ETLPipeline:
    def __init__(self):
        db_path = os.getenv("CHROMA_DB_PATH")
        os.makedirs(db_path, exist_ok=True)
        self.persistent_client = chromadb.PersistentClient(
            path=db_path)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.wikipedia = Wikipedia(
            user_agent=os.getenv('USER_AGENT'), language='en')

    def load_web_data(self, urls, collection_name):
        """ETL for admission websites [[7]][[9]]"""
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=OllamaEmbeddings(
                model="llama3.2",
                base_url=os.getenv("OLLAMA_URL")
            ),
            client=self.persistent_client,
            persist_directory=os.getenv("CHROMA_DB_PATH"),
        )
        loader = WebBaseLoader(urls)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)

        # Add to ChromaDB
        for chunk in chunks:
            vector_store.aadd_documents(
                documents=[chunk],
                ids=[str(uuid4())]
            )

    def load_wikipedia_data(self, queries, collection_name):
        """ETL for Wikipedia knowledge [[1]][[6]]"""
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=OllamaEmbeddings(
                model="llama3.2",
                base_url=os.getenv("OLLAMA_URL")
            ),
            client=self.persistent_client,
            persist_directory=os.getenv("CHROMA_DB_PATH"),
        )
        documents = []

        for query in queries:
            page = self.wikipedia.page(query)
            if page.exists():
                documents.append(Document(
                    page_content=page.text,
                    metadata={"source": "wikipedia", "title": page.title}
                ))

        chunks = self.text_splitter.split_documents(documents)

        # # Add to ChromaDB
        for chunk in chunks:
            vector_store.aadd_documents(
                documents=[chunk],
                ids=[str(uuid4())]
            )
    
    def perform_etl(self):
        print("Starting ETL Pipeline...")
        self.load_wikipedia_data(["Concordia_University"], "admission")
        self.load_web_data(
            [
                # BCompSc
                "https://www.concordia.ca/academics/undergraduate/computer-science.html",
                "https://www.concordia.ca/students/financial/scholarships-funding/out-of-province-awards.html",
                "https://www.concordia.ca/academics/degrees/program-length.html",
                "https://www.concordia.ca/ginacody/computer-science-software-eng/programs/computer-science/bachelor/bcompsc-honours.html",
                "https://www.concordia.ca/academics/undergraduate/calendar/current/section-71-gina-cody-school-of-engineering-and-computer-science/section-71-70-department-of-computer-science-and-software-engineering/section-71-70-10-computer-science-and-software-engineering-courses.html",
                "https://www.concordia.ca/academics/undergraduate/calendar/current/section-71-gina-cody-school-of-engineering-and-computer-science/section-71-70-department-of-computer-science-and-software-engineering/section-71-70-2-degree-requirements-bcompsc-.html#12135",
                "https://www.concordia.ca/admissions/undergraduate/requirements/cegep-students.html",
                "https://www.concordia.ca/admissions/undergraduate/requirements/canada.html",
                "https://www.concordia.ca/admissions/undergraduate/requirements/international.html",
                "https://www.concordia.ca/admissions/undergraduate/requirements.html",

                # MApCompSc
                "https://www.concordia.ca/academics/graduate/computer-science-mcompsci-applied.html",
                "https://www.concordia.ca/gradstudies/future-students/how-to-apply/english-language-proficiency.html",
                "https://www.concordia.ca/gradstudies/future-students/how-to-apply/requirements.html",
                "https://www.concordia.ca/students/international/immigration.html",
                "https://www.concordia.ca/students/international/applying-for-your-immigration-documents.html",
            ], "admission")
        self.load_wikipedia_data([
            "Artificial_intelligence",
            "Machine_learning",
            "Neural_network",
            "Natural_language_processing",
            "Deep_learning",
            "Large_language_model"
        ], "ai_knowledge")
        self.load_wikipedia_data([
            "Science",
            "Technology",
            "History",
            "Geography",
            "Mathematics",
            "Politics",
            "Economics",
            "Literature",
            "Art",
            "Music",
            "Sports",
            "Food",
            "Health",
        ], "general")
        print("ETL process completed")
