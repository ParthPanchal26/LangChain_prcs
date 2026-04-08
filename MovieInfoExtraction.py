# Import Libraries
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import streamlit as st

# Load secrets
load_dotenv()

# Create pydantic schema
class Cast(BaseModel):
    name: str = Field(description="Cast name")
    role: str = Field(description="Role of the cast")

class MovieData(BaseModel):
    title: str = Field(description="Write the movie title, NA if not available")
    year: str = Field(description="Write movie release year, NA if not available")
    director: str = Field(description="Write description of movie, NA if not available")
    genre: list[str] = Field(description="List all genre")
    cast: list[Cast] = Field(description="List movie cast here, NA if not available")
    rating: int = Field(description="Write rating of movie, NA if not available")
    duration: str = Field(description="Duration of movie, NA if not available")
    language: str = Field(description="Write language of movie, NA if not available")
    country: str = Field(description="Country of origin, Na if not available")
    summary: str = Field(description="Summary of movie, NA if not available")

# Create system prompt
systemPrompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """
        You are a movie data extraction assistant. You will receive a long freeform text about a movie. Your task is to extract all relevant movie details and output them strictly in JSON format.

        Use the following output format as a placeholder:

        {output_format}

        Rules:
        1. Always respond ONLY in valid JSON that matches the schema in <OUTPUT_FORMAT_HERE>.
        2. Do not include explanations, extra text, or markdown formatting.
        3. If some information is missing or cannot be inferred from the text, use null.
        4. Ensure the JSON is syntactically correct.
        5. For cast, if roles are unknown, you may leave "role" as null.
        6. Parse all relevant details from the text, including title, year, director, genre, cast, rating, duration, language, country, and summary.
    """
        ),
        HumanMessagePromptTemplate.from_template("{paragraph}"),
    ]
)

# Create parser
parser = PydanticOutputParser(pydantic_object=MovieData)

# Get movie info
st.title("Movie Data Extraction")
paragraph = st.text_area("Enter movie description text here:")

# Extract data
if st.button("Extract Movie Data"):
    final_system_prompt = systemPrompt.invoke(
        {"output_format": parser.get_format_instructions(), "paragraph": paragraph}
    )

    model = init_chat_model(
        model="mistralai/mistral-small-4-119b-2603",
        model_provider="nvidia",
        temperature=0.2,
    )

    response = model.invoke(final_system_prompt)
    st.code(response.content, language="json")
