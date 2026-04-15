# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv

# load_dotenv()
# model = ChatGoogleGenerativeAI(model="gemini-pro")
# result = model.invoke("What is the capital of India?")
# print(result.content)

# import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# os.environ["GOOGLE_API_KEY"] = "AIzaSyA6ib7UMCnTbymkO915YMug4IaNpp8mQH8"

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

response = model.invoke("Why do parrots talk?")

print(response.content)