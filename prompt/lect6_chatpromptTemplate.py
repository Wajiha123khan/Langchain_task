from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage

# chat promptTemplate = **ChatPromptTemplate** — ek structured message format jo 
# System, Human, aur AI roles define karta hai taake LLM ko consistent aur clear prompts mile.

chat_template = ChatPromptTemplate([
    ('system', 'Your are helpful {domain} Expert'), # AI ko role deta hai
    ('human', 'Explain in simple term , What is {topic}') # User ka sawaal hota hai
    
])

prompt = chat_template.invoke({'domain':'AI', 'topic':'Transformer'})
print(prompt)