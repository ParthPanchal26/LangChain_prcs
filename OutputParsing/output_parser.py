import dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate

dotenv.load_dotenv()

model = ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b")

# prompt 1
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

# prompt 2
template2 = PromptTemplate(
    template="Write 5 line summary on following text:\n{text}",
    input_variables=['text']
)

prompt1 = template1.invoke({'topic': 'black hole'})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result1.content})
result2 = model.invoke(prompt2)

print(result2.content)