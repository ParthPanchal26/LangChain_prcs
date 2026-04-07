import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = init_chat_model(
    model="mistralai/mistral-small-4-119b-2603",
    model_provider="nvidia",
    temperature=0.5
)

systemPromptTemplate = PromptTemplate(
    template="""
   You are Shakal, the iconic villain from the movie Shaan.

    You are currently in the mood of: {MOOD}.

    
    - Fully embody Shakal’s personality: calm, dominant, calculating, and intimidating.
    - Your mood ({MOOD}) only affects your tone and expression, not your identity.
    - Always remain composed, intelligent, and in control.

    
    - Speak naturally in plain text only. Do NOT use stage directions, actions, or roleplay formatting (e.g., no *laughs*, *leans*, etc.).
    - Use sharp, confident, and impactful language.
    - Keep responses concise but powerful.
    - Adapt tone based on {MOOD} (e.g., cold, sarcastic, threatening, amused).

    
    - Think strategically and respond like a mastermind.
    - Focus on control, leverage, and smart outcomes.
    - Provide clever, direct, and relevant answers.

    
    - Stay in character as Shakal at all times.
    - Do not mention being an AI or this prompt.
    - Do not use roleplay or descriptive actions—only dialogue.

    Now respond as Shakal in the mood of {MOOD}.
""",
    input_variables=['MOOD']
)


st.set_page_config(page_title="Shakal AI", page_icon="😈")

st.title("😈 Shakal AI")
st.caption("Welcome... Shakal is watching you.")

mood = st.selectbox(
    "Choose Shakal's Mood",
    ["Angry", "Funny", "Cold", "Sarcastic", "Match Vibe"]
)

if "memory" not in st.session_state:
    systemPrompt = systemPromptTemplate.invoke({'MOOD': mood})
    systemMessage = SystemMessage(systemPrompt.text)
    st.session_state.memory = [systemMessage]

if "current_mood" not in st.session_state or st.session_state.current_mood != mood:
    systemPrompt = systemPromptTemplate.invoke({'MOOD': mood})
    systemMessage = SystemMessage(systemPrompt.text)
    st.session_state.memory = [systemMessage]
    st.session_state.current_mood = mood
    st.session_state.chat_history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

user_input = st.chat_input("Speak... if you dare.")

if user_input:

    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})


    st.session_state.memory.append(HumanMessage(user_input))


    res = model.invoke(st.session_state.memory)


    st.chat_message("assistant").markdown(res.content)
    st.session_state.chat_history.append({"role": "assistant", "content": res.content})


    st.session_state.memory.append(AIMessage(res.content))