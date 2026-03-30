from typing import Literal

import dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel, Field

dotenv.load_dotenv()

model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")
parser = StrOutputParser()

class classification_schema(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Return sentiment of feedback")

parser2 = PydanticOutputParser(pydantic_object=classification_schema)

prompt_1 = PromptTemplate(
    template="Classify this feedback in terms of 'positive' or 'negative'\n{feedback}\n{format_instructions}",
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt_1 | model | parser2

prompt_2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback:\n{feedback}",
    input_variables=['feedback']
)

prompt_3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback:\n{feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x : x.sentiment == 'positive', prompt_2 | model | parser),
    (lambda x : x.sentiment == 'negative', prompt_3 | model | parser),
    RunnableLambda(lambda x:"could not judge feedback")
)   

chain = classifier_chain | branch_chain
result = chain.invoke({'feedback': "Instagram is the worst app i ever installed"})
print(result)