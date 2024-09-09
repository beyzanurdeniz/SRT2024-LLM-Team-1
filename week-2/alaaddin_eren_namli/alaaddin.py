"""
This is a simple RAG application which gets up-to-date data from documents
Google's Gemini 1.5 Flash model is used in this project.
Using the power of LangChain framework and Retrieval Augmented Generation, the model gets
relevant data according to the prompt and gives up-to-date answers
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain


# Function that creates vector database
def create_vector_db(docs):
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",google_api_key=YOUR_API_KEY)
    vector_store = FAISS.from_documents(docs,embedding=embedding)
    return vector_store

# Create sample document
my_document = Document(
    page_content="""
    MOCK PAGE CONTENT
""")

# Split the text into chunks and store them as embeddings
docs = [my_document]
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
split_docs = splitter.split_documents(docs)
vector_database = create_vector_db(split_docs)

# Initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=YOUR_API_KEY,temperature=0.5)

# Create the prompt
prompt = ChatPromptTemplate.from_template("""
Answer the user's question:
Context: {context}
Question: {input}
""")

# Construct the chain
chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

# Use retriever to get the most relevant data and construct the retrieval chain
retriever = vector_database.as_retriever(search_kwargs={"k":2})
retrieval_chain = create_retrieval_chain(retriever,chain)

# Get the response and print it
response = retrieval_chain.invoke({
    "input":"USER INPUT"
})
print(response["answer"])