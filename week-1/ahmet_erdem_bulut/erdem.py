"""
This application generates personalized book and movie recommendations to users based on
their preferences, past likes and current mood using LangChain.
The Ollama model is used as the language model to generate recommendations.
"""


from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama3")
template="""                             
    Based on the user's preferences: {preferences}, past likes: {past_likes} and current mood: {current_mood},
    recommend some books and movies, and for each recommendation give a short summary, tell what it is about, and
    tell why it would be a good choice according to user's inputs. It would be better if you recommend some additional  
    underappreciated works and also your recommendations don't have to be fictional.
    """

prompt_template = PromptTemplate(input_variables=["preferences", "past_likes", "current_mood"], template=template)
parser = StrOutputParser()
chain = prompt_template | llm | parser

preferences = input("Write your preferred types of books and movies: ")
past_likes = input("Write some of your favorite books or movies: ")
current_mood = input("How are you feeling currently?: ")

recommendations = chain.invoke({"preferences": preferences, "past_likes": past_likes, "current_mood": current_mood})
print(recommendations)