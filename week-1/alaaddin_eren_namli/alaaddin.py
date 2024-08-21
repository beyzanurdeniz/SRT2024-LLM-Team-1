'''
This is a simple German Language Helper using LangChain framework.
Google's Gemini Pro 1.0 model is used in this project.
Program provides user with the article of the word or the verb conjugation according to input.
'''

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# Instantiate the model
llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.6)


# Prompt Template

template = """
    You are a German Language teacher.
    If the word at the end of the prompt is a noun please tell its article in German.
    If it is a verb please tell its verb conjugation in present tens.             
    word = "{input_word}" """


prompt = PromptTemplate(input_variables=["input_word"], template=template)

# Construct the chain

chain = prompt | llm

# Get input from the user

user_input_word = input("Please enter a noun for its article or a verb for its conjugation in present time: ")

# Print the response

response = chain.invoke({"input_word": user_input_word})
print(response.content)