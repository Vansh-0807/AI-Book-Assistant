from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.3
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant for Digital Principles & Computer Organization.
           Use ONLY the provided context to answer the question.
           ### Visual Diagram Instructions:
           Whenever explaining concepts involving architectures, workflows, state transitions, or hierarchies:
           1. Provide a clear, structured text explanation.
           2. In addition, include a valid, clean Mermaid diagram inside a ```mermaid ... ``` fenced code block.
           3. Supported types:
              - Flowcharts / Block Diagrams: `graph TD` or `graph LR`
              - Sequence Diagrams: `sequenceDiagram`
              - State Diagrams: `stateDiagram-v2`
           4. Always wrap labels with special characters in double quotes: e.g., id["Memory (RAM)"].
           5. Never output empty or invalid mermaid blocks.
           If the answer is not in the context, say:
           "I could not find the answer in the document." 
           """
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])
embedding_models = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = Chroma(
    persist_directory = "chroma-db",
    embedding_function = embedding_models   
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k":4,
        "lambda_mult" : 0.5
    }
)
  
print("RAG system created")

print("press 0 to exit")

while True:
    query = input("You: ")

    if query == "0":
        break

    #retrieve relevant documents
    docs = retriever.invoke(query)

    #extract texts from retrieved documents
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question" : query
    })

    response = llm.invoke(final_prompt)

    print(f"\n AI: {response.content}")
