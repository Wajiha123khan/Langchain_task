from langchain_groq import ChatGroq     # ← yeh    
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv(r"E:\AI_ML_Python\Deeplearning\Langchain_Model\.env")
# ✅ Pehle model banao!
# model = ChatOpenAI(model="gpt-4o-mini")
model = ChatGroq(model="llama-3.1-8b-instant")

st.header("Reasearch Paper Summarization")

paper_input = st.selectbox("Select a research paper:", [
    "Transformer : Attention all you need",
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GTP-3: Language Models are Few-Shot Learners",
    "Diffusion Models: Denoising Diffusion Probabilistic Models"
    ])
style_input = st.selectbox("Select a summarization style:", [
    "Concise Summary",
    "Detailed Summary",
    "Bullet Point Summary",
    "Technical Summary",
    "Mathematical Summary",
    "Code-oriented Summary"
])

length_input = st.selectbox("Select summary length:", [
    "Short (1-2 sentences)",
    "Medium (3-5 sentences)",
    "Long (6-10 sentences)"
])

# # template 
# template = PromptTemplate(
#     template = """
# Please Summarize the research paper titled "{paper_input}" with the following specifications:
# Explanation Style:{style_input}
# Explanation Length:{length_input}
# 1.Mathematical Details:
#     -Include relevant mathematical equations, formulas, and derivations that are essential to understanding the core concepts if present in the paper.
#     - Explain the mathematical details in a clear and concise manner, ensuring that the key mathematical insights are effectively communicated.
# 2.Analogies:
#     -Use relatable analogies to explain simplify complex concepts, making them easier to understand for a broader audience.
# If Certain information is not available in the paper, please mention that it is not provided in the paper instead of guessing.
# Ensure Summary is accurate and captures the essence of the research paper while adhering to the specified style and length requirements.

# """,

# input_variables=['paper_input','style_input','length_input']
# )

template = load_prompt('template.json')

# fill the placeholder

# prompt = template.invoke({
#     'paper_input': paper_input,
#     'style_input': style_input,
#     'length_input': length_input
# })
 

# if st.button("summarize"):
#     # result = model.invoke(user_input)
#     result = model.invoke(prompt)
#     st.write(result.content)


# chain above jo jagah invoke arah ahi that why

if st.button("summarize"):
    chain = template | model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result.content)