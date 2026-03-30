import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

dotenv.load_dotenv()

template = PromptTemplate(
    template="Create me 5 facts about {topic}",
    input_variables=['topic']
)

model = ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b")

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({"topic": "coding"})

print(result)