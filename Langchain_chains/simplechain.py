from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv  

load_dotenv()

prompt = PromptTemplate(
    template = 'Generate 5 interesting facts about {topic}',
    input_variables = ['topic']
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "space"})

print(result)

chain.get_graph().print_ascii()