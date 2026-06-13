from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv(r"E:\AI_ML_Python\Deeplearning\Langchain_Model\.env")

model1 = ChatGroq(model="llama-3.1-8b-instant")
model2 = ChatGroq(model="llama-3.1-8b-instant")

Prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=['text']
)

Prompt2 = PromptTemplate(
    template="Generate 5 short questions from the following text \n {text}",
    input_variables=['text']
)

Prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': Prompt1 | model1 | parser,
    'quiz': Prompt2 | model2 | parser
})

merge_chain = Prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """he introduction of AlexNet by Krizhevsky et al. [4] marked a significant milestone in
deep learning. AlexNet’s success in the ImageNet challenge demonstrated the potential
of deep CNNs to handle complex image recognition tasks, leading to increased interest
and research into deep learning across various domains. RNNs gained prominence with
the work of Hochreiter and Schmidhuber [26], who introduced Long Short-Term Memory
(LSTM) networks. LSTM networks were designed to overcome the vanishing gradient
problem associated with standard RNNs, enabling them to learn long-range dependencies
within data sequences. This capability has been critical in advancing sequence modeling
tasks such as natural language processing and speech recognition.
The concept of self-supervised learning was further explored by Devlin et al. [27] in
their development of Bidirectional Encoder Representations from Transformers (BERT),
Information 2024, 15, 755
3 of 45
which revolutionized natural language processing. BERT utilizes transformer architectures
in a novel training approach that leverages unlabeled data, setting new state-of-the-art
benchmarks for a variety of NLP tasks. Recent innovations have also focused on improving
the efficiency and adaptability of DL models. Howard et al. [28] introduced MobileNets,
which uses depth-wise separable convolutions to build lightweight deep neural networks
for mobile and edge-device applications. This work shows the industry’s shift towards
developing computationally efficient models that do not compromise performance.
Furthermore, the integration of deep learning with reinforcement learning has led to
the development of models capable of mastering complex games and tasks. Mnih et al. [29]
presented a model that combined Q-learning with deep learning to create Deep Q-Networks
(DQNs), enabling these models to perform at human-level capabilities on Atari games.
This integration showcases the robustness of DL in understanding and interacting with
environments in a way that mimics human learning. Other recent reviews have explored
different architectures and applications [18–21,23]. While recent reviews have made signifi
cant contributions to our understanding of deep learning architectures, they often focus
on specific subsets of the field, such as enhancements in neural network efficiency or
applications within specific domains."""
result = chain.invoke({'text': text})
print(result)

chain.get_graph().print_ascii()