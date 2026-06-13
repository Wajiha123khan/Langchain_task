from langchain_core.prompts import PromptTemplate

# template 
template = PromptTemplate(
    template = """
Please Summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style:{style_input}
Explanation Length:{length_input}
1.Mathematical Details:
    -Include relevant mathematical equations, formulas, and derivations that are essential to understanding the core concepts if present in the paper.
    - Explain the mathematical details in a clear and concise manner, ensuring that the key mathematical insights are effectively communicated.
2.Analogies:
    -Use relatable analogies to explain simplify complex concepts, making them easier to understand for a broader audience.
If Certain information is not available in the paper, please mention that it is not provided in the paper instead of guessing.
Ensure Summary is accurate and captures the essence of the research paper while adhering to the specified style and length requirements.

""",

input_variables=['paper_input','style_input','length_input']
)

template.save('template.json')
