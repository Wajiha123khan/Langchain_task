from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
embedding = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2',
    # dimension = 300  
    # temprature = 0,
    # max_tokens = 12 these are not used in this embedding model 
)

# this is 2D Array of shape (3, 384) where 3 is the number of documents and 384 is the dimension of the embedding
List_docs = [
    "The old lighthouse keeper had a habit of talking to the sea. ",
    "Every evening, just before the fog rolled in, he would stand at the edge of the rocks and whisper the names of ships that never came back. ",
    "Nobody in the village thought it strange anymore — they had grown used to his rituals the way you grow used to the sound of rain on a tin roof.",
    "His cat, a fat orange thing named Biscuit, would sit beside him and stare at the horizon with the same quiet intensity, as if it too were keeping some private record of the lost."
]
# this is 1D Array of shape (384,) where 384 is the dimension of the embedding
# User_Query = "What did the lighthouse keeper do every evening?"

User_Query = "tell me about the rain"

doc_embedding = embedding.embed_documents(List_docs)
query_embedding = embedding.embed_query(User_Query)

# print(cosine_similarity([query_embedding], doc_embedding))
# so query_embedding use square beacause it is 1D array and doc_embedding is
# 2D array so we need to make query_embedding 2D array by using square brackets
score = cosine_similarity([query_embedding],doc_embedding)[0] # [0] ye row 1 jo uthaeya jo ke 1D bna jayega 
print(score)
# print(score[0])
# print(score[1])

#print(list(enumerate(score)))

# lambda funstion is temporary function jo ek normal def dfunction jo multiple line of code hoata usko wo dhort line main convert krdea hai:
# lambda <inputs> : <kya return karna hai>
#index =sorted(list(enumerate(score)), key=lambda x:x[1])[-1][0] 

index, score =sorted(list(enumerate(score)), key=lambda x:x[1])[-1]

print("the user questions is:",User_Query)
print(List_docs[index])
print("similarity score is ",score)


# index, score = ...[-1] = result main — sirf ek number so ismein overwrite hoayega score jo ke ek hi similarity score dikayega 
# index = ...[-1][0] =  — poora array yeh overwrite nahi hoga, sirf index ko overwrite karega jo ke 1D array main hoga