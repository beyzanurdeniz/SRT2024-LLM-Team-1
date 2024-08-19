'''
This a simple application using LangChain to summarize an article.
Google's gemini-pro model is used for summarization.
Takes the link of the article as input and prints the summary.
'''

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

llm = ChatGoogleGenerativeAI(model = "gemini-pro")

link = input("Please enter the link of the article you want to summarize: ")
loader = WebBaseLoader(link)
docs = loader.load()

template = """Summarize the following article:
"{text}"
SUMMARY:"""

prompt = PromptTemplate(input_variables=["text"], template=template)

chain = prompt | llm

for doc in docs:
    result = chain.invoke({"text": doc.page_content})
    print(result.content)