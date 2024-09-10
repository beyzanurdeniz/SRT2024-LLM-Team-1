from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from local import gemini_api_key, root_path

file_paths = [
    f"{root_path}/cphb.pdf",
    f"{root_path}/inzva-bundles/01-intro.pdf",
    f"{root_path}/inzva-bundles/02-algorithms-1.pdf",
    f"{root_path}/inzva-bundles/03_math1.pdf",
    f"{root_path}/inzva-bundles/04_Graph1.pdf",
    f"{root_path}/inzva-bundles/05_DP1.pdf",
    f"{root_path}/inzva-bundles/06-data-structures-1.pdf",
    f"{root_path}/inzva-bundles/07_Graph2.pdf",
    f"{root_path}/inzva-bundles/08-data-structure-2.pdf",
    f"{root_path}/inzva-bundles/10_dp_2.pdf",
    f"{root_path}/inzva-bundles/11-graph-3.pdf",
    f"{root_path}/inzva-bundles/12_Math3.pdf",
    f"{root_path}/inzva-bundles/14-algorithms-5.pdf",
    f"{root_path}/inzva-bundles/Data-Structures-3.pdf"
]

all_documents = []
for path in file_paths:
    loader = PyPDFLoader(file_path=path)
    documents = loader.load()
    all_documents.extend(documents)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.split_documents(all_documents)

embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=gemini_api_key)
vector_store = FAISS.from_documents(split_docs, embedding=embedding)

vector_store.save_local("faiss_index")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api_key, temperature=0.5)

prompt_template = ChatPromptTemplate.from_template("""
Answer the user's question based on the following context. If you cannot find the answer, respond with 'no answer found'. If found, organize it as readble as possible.".
Context: {context}
Question: {input}
""")

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

while True:
    user_query = input("Ask a question (or type 'exit' to quit): ")
    if user_query.lower() == 'exit':
        break

    result = qa_chain.invoke({"query": user_query})
    print(f"Answer: {result['result']}\n")
