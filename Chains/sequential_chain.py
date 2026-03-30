import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

dotenv.load_dotenv()

prompt1 = PromptTemplate(
    template="Write a paragraph on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Write summary on this paragraph:\n{paragraph}",
    input_variables=['paragraph']
)

model = ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b")

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "coding"})

print(result)