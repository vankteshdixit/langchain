from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

model = ChatHuggingFace(llm=HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
))

messages =[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about LangChain.")
]

result = model.invoke(messages)

messages.append(AIMessage(content = result.content))

print(messages)

