# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(r"E:\AI_ML_Python\Deeplearning\Langchain_Model\.env")

Prompt = PromptTemplate(
    template="Generate 5 intresting facts about {topic}",
    input_variables=['topic']
)

model = ChatGroq(model="llama-3.1-8b-instant")
parser = StrOutputParser()

chain = Prompt | model | parser
result = chain.invoke({'topic':'CNN'})
print(result)

# pip install grandalf
chain.get_graph().print_ascii()