# # from langchain_openai import OpenAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
# # from dotenv import load_dotenv
# from sklearn.metrics.pairwise import cosine_similarity

# # load_dotenv()

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", dimensions=300)

# documents = [
#     "The capital of India is New Delhi.",
#     "The capital of France is Paris.",
#     "The capital of Japan is Tokyo."
# ]

# query = "What is the capital of India?"

# documents_embeddings = embeddings.embed_documents(documents)
# query_embedding = embeddings.embed_query(query)

# similarities = cosine_similarity([query_embedding], documents_embeddings)[0]

# print("Similarities:", similarities)

from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "The capital of India is New Delhi.",
    "The capital of France is Paris.",
    "The capital of Japan is Tokyo."
]

query = "What is the capital of India?"

documents_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

similarities = cosine_similarity([query_embedding], documents_embeddings)

print("Similarities:", similarities)

# Find most similar document
best_match_index = similarities.argmax()
print("Best match:", documents[best_match_index])