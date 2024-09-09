from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load documents from 'data' folder and split them to the chunks
document_loader = PyPDFDirectoryLoader(path="data")
documents = document_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Initialize models
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store = FAISS.from_documents(documents=texts, embedding=embedding_model)
llm = ChatOllama(model='llama3')

template = """
You are an assistant for question-answering tasks about some prompt engineering topics. Use the following pieces of  
retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use five sentences
maximum and keep the answer concise.
Context: {context} 
Question: {question} 
Answer:
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def qa_chain_with_sources(query):
    retrieved_docs = vector_store.as_retriever().invoke(query)
    context_text = format_docs(retrieved_docs)
    answer = llm.invoke(prompt.format_prompt(context=context_text, question=query))
    return answer, retrieved_docs


while True:
    user_query = input("What do you want to ask? If nothing, write 'exit' to quit: ")
    if user_query.lower() == "exit":
        break

    result, source_docs = qa_chain_with_sources(user_query)
    print(f"Answer: {result.content}\n")

    if not result.content.startswith("I don't know"):
        print("Source Documents:")
        for doc in source_docs:
            source = doc.metadata.get("source")
            page = doc.metadata.get("page", "unknown")  # Get the page number if available
            print(f"Source: {source}, Page: {page + 1}") # Page indices start from 0
