import dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel, Field
import json

dotenv.load_dotenv()

class schema(BaseModel):
    name: str = Field(default=None, description="Full name of the person")
    age: int = Field(default=None, description="Age of the person (must be > 18)", gt=18)
    city: str = Field(default=None, description="Write city of the person")

model = ChatNVIDIA(model="nvidia/nemotron-3-super-120b-a12b")

parser = PydanticOutputParser(pydantic_object=schema)

template = PromptTemplate(
    template="Give me some details of random person\n{formatting_instructions}",
    input_variables=[],
    partial_variables={"formatting_instructions": parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({})
print(result)
print(result.name)