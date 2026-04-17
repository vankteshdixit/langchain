from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableParallel,
    RunnableSequence,
    RunnablePassthrough,
    RunnableLambda
)
from dotenv import load_dotenv
load_dotenv()

loader = TextLoader('rag\\cricket.txt', encoding= 'utf-8')

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template = 'Summarize the following text: \n {text}',
    input_variables = ['text']
)
parser = StrOutputParser()

docs = loader.load()

print(docs)

print(len(docs))

print(docs[0])

chain = prompt | model | parser

print("-" * 50)

print(chain.invoke({"text": docs[0].page_content}))