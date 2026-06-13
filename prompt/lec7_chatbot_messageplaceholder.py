from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# MessagesPlaceholder = MessagesPlaceholder ek blank space hai 
# prompt mein jahan poori chat history inject hoti hai dynamically.

# chat template
chat_temp = ChatPromptTemplate([
    
    ('system', "You are a helpful customer support agent"),
    #  Yahan par runtime pe puri purani conversation aa jaati hai  taake AI ko pata ho pehle kya baat hui th
    MessagesPlaceholder(variable_name='chat_history'), 
    ('human', '{query}')

])


# load chat History
chat_history = []
with open('prompt\chatbot_chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

# create prompt

prompt = chat_temp.invoke({'chat_history':chat_history, 'query':HumanMessage(content='where is my refund')})
print(prompt)