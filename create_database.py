#load the document
#convert them into chunks
#create embeddings
#store them in the database.

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import time

#loading the documents
data = PyPDFLoader("document loaders/7BCEE2A-Digital Principles and Computer Organization.pdf")
load = data.load()

# split the documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(load)
print(f"Total chunks: {len(chunks)}")

# using embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# storing them in the database in batches to avoid rate limits
BATCH_SIZE = 80  # stay under 100/min free-tier limit
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    print(f"Processing batch {i // BATCH_SIZE + 1}: chunks {i+1}-{i+len(batch)} of {len(chunks)}")
    
    if i == 0:
        # First batch: create the vectorstore
        vectorstore = Chroma.from_documents(
            documents=batch,
            embedding=embedding_model,
            persist_directory="chroma-db"
        )
    else:
        # Subsequent batches: add to existing vectorstore
        vectorstore.add_documents(batch)
    
    # Wait 65 seconds before next batch to respect rate limit
    if i + BATCH_SIZE < len(chunks):
        print("Waiting 65 seconds for rate limit reset...")
        time.sleep(65)

print("Database created successfully!")