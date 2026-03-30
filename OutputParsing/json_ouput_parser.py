import dotenv
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
import warnings

warnings.filterwarnings('ignore')

dotenv.load_dotenv()

model = ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b")

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a person\n{format_instructions}",
    input_variables=[],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)


print('--------------------------------------------------------------')
chain = template | model | parser
result = chain.invoke({})
print(result)
print('--------------------------------------------------------------')