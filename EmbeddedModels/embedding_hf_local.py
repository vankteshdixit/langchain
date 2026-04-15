from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name= 'sentence-transformers/all-MiniLM-L6-v2')

# text = "Delhi is the capital of India."

documents = [
    "The capital of India is New Delhi.",
    "The capital of France is Paris.",
    "The capital of Japan is Tokyo."
]
vectors = embedding.embed_documents(documents)

for vector in vectors:
    print(str(vector))