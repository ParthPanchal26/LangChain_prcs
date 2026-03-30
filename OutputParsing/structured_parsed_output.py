import dotenv
# from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
# from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
import warnings

warnings.filterwarnings('ignore')

dotenv.load_dotenv()

model = ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b")

schema = [
    ResponseSchema(name='name', description='Full name of person'),
    ResponseSchema(name='age', description='Age of the person in integer data type'),
    ResponseSchema(name='city', description='City of the person'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give me some details about a person\n{format_instructions}",
    input_variables=[],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({})
print(result)
