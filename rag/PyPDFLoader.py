from langchain_community.document_loaders import PyPDFLoader
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

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm = llm)
parser = StrOutputParser()
loader = PyPDFLoader('rag\\Vanktesh Resume.pdf')

docs = loader.load()

# print(docs)

# print(len(docs))

print(docs[0])
