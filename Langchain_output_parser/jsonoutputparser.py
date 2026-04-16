from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template = 'Give me the name , age and city of a fictional person \n {format_instructions}',
    input_variables = [],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# prompt = template.format()

# result = model.invoke(prompt)

# parser.parse(result.content)

chain = template | model | parser

result = chain.invoke({})

print(result)

print(type(result))