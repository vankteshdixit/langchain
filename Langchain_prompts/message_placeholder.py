from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# chat template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

chat_history = []

# load chat history properly
with open("chat_history.txt") as f:
    for line in f:
        if line.startswith("Human:"):
            chat_history.append(HumanMessage(content=line.replace("Human:", "").strip()))
        elif line.startswith("AI:"):
            chat_history.append(AIMessage(content=line.replace("AI:", "").strip()))

print(chat_history)

# Example usage
prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": "What is my last question?"
})

print(prompt)

