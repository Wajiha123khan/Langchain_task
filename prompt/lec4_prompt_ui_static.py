from langchain_groq import ChatGroq     #  Groq is openAI whcih is free to use wher i am using that model like =model="llama-3.1-8b-instant  
from dotenv import load_dotenv
import streamlit as st 

load_dotenv(r"E:\AI_ML_Python\Deeplearning\Langchain_Model\.env")
# ✅ Pehle model banao!
# model = ChatOpenAI(model="gpt-4o-mini")
model = ChatGroq(model="llama-3.1-8b-instant")

st.header("Reasearch Paper Summarization")
user_input = st.text_input("Enter the research paper content:")
if st.button("summarize"):
    result = model.invoke(user_input)
    st.write(result.content)
