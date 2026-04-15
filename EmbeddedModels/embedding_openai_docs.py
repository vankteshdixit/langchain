from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

documents = [
    "The capital of India is New Delhi.",
    "The capital of France is Paris.",
    "The capital of Japan is Tokyo."
]

result = embeddings.embed_documents(documents)

for doc_result in result:
    print(str(doc_result))