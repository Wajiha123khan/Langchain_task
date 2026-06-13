from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# openai ki api use hain so it create error cause it not free 
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimension=32,
)

result = embedding.embed_query("Hello world")
print(str(result))