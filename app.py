from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import ollama
from langchain_community.llms import Ollama

from langchain_core.output_parsers import StrOutputParser
import os
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
# os.environ['LANG_CHAIN_API_KEY'] = os.getenv('LANG_CHAIN_API_KEY')
langchain_api_key = st.secrets.get("LANG_CHAIN_API_KEY", os.getenv("LANG_CHAIN_API_KEY"))
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
os.environ['LANG_CHAIN_API_KEY'] = "true"
os.environ["LANG_CHAIN_PROJECT"] = "project Q&A"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are healpful assistant"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,engine,temperature,max_tokens):
    llm = Ollama(model=engine)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question":question})
    return answer

engine=st.sidebar.selectbox("Select Open AI model", ["mistral"])
## Adjust response parameter
temperature=st.sidebar.slider("Temperature", min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)
## MAin interface for user input
st.write("Goe ahead and ask any question")
user_input=st.text_input("You:")
if user_input :
    response= generate_response(user_input, engine, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the user input")