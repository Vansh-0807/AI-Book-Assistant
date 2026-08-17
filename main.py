from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings

llm = ChatMistralAI(
    model = "mistral-small-latest",
    temperature = 0.3
)

prompt = ChatPromptTemplate.from_messages(
    [("system",
    """You are a helpful AI assistant.
    Use ONLY the provided context to answer the question.
    
    If the answer is not present in the context,
     say: "I could not find the answer in the document."  
     """),

     ("human", 
      """Context: 
       {context}
       Question:
       {question}
       """)]
)

embedding_models = MistralAIEmbeddings()

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
