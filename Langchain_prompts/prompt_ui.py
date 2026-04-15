from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)

st.header("Research Tool")

user_input = st.text_input("Enter your query:")

if st.button("Search"):
    model = ChatHuggingFace(llm=llm)
    result = model.invoke(user_input)
    st.subheader("Answer:")
    st.write(result.content)