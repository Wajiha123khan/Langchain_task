# Package / Library , langchain_huggingface
# Class HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# 384 dimensional vector for sentence-transformers/all-MiniLM-L6-v2 = this is locally run
#  Instance of HuggingFaceEmbeddings with the specified model name
embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
)

docs = [
    "My name is wajiha khan",
    "I am on a mission to learn langchain and huggingface",
    "Langchain is a powerful framework for building language model applications",
]
#text = "My name is wajiha khan"
#result_vector = embedding.embed_query(text)
result_vector = embedding.embed_documents(docs)

print(str(result_vector))
