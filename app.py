import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_api_key = os.getenv("GROQ_API_KEY")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Question: {question}")
])

def generate_response(question, model, temperature, max_tokens):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

st.title("Q&A Chatbot (Groq)")

model = st.sidebar.selectbox(
    "Select Groq model",
    ["llama3-8b-8192", "mixtral-8x7b-32768"]
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 200)

st.write("Ask me anything:")
user_input = st.text_input("You:")

if user_input:
    with st.spinner("Thinking..."):
        response = generate_response(user_input, model, temperature, max_tokens)
        st.success(response)
