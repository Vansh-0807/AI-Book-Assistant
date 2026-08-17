#load the document
#convert them into chunks
#create embeddings
#store them in the database.

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

#loading the documents
data = PyPDFLoader("document loaders/7BCEE2A-Digital Principles and Computer Organization.pdf")
load = data.load()

# split the documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(load)

# using embedding model
embedding_model = MistralAIEmbeddings()

# storing them in the database
vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embedding_model,
    persist_directory = "chroma-db"
)