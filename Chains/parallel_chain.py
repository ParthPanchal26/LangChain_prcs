import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_nvidia_ai_endpoints import ChatNVIDIA

dotenv.load_dotenv()

prompt_1 = PromptTemplate(
    template="Write me notes on these:\n{text}",
    input_variables=['text']
)

prompt_2 = PromptTemplate(
    template="Create me quiz for these notes:\n{text}",
    input_variables=['text']
)

prompt_3 = PromptTemplate(
    template="""Merge the provided notes and quiz into single document:
    notes:
    {notes}
    quiz:
    {quiz}
    """,
    input_variables=['notes', 'quiz']
)

model = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt_1 | model | parser,
    'quiz': prompt_2 | model | parser
})

merge_chain = prompt_3 | model | parser

chain = parallel_chain | merge_chain

text = "Linear regression"

result = chain.invoke({'text': text})

print(result)