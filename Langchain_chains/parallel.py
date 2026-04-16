# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.schema.runnable import RunnableParallel
# from dotenv import load_dotenv  

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#     task="text-generation",
# )

# model1 = ChatHuggingFace(llm=llm)
# model2 = ChatHuggingFace(llm=llm)

# prompt1 = PromptTemplate(
#     template = 'Generate short and simple notes from the following text \n {text}',
#     input_variables = ['text']
# )

# prompt2 = PromptTemplate(
#     template = 'Generate a 5 short question answers from the following text\n {text}',
#     input_variables = ['text']
# )

# prompt3 = PromptTemplate(
#     template = 'Merge the provided notes and quiz into the a single document \n {notes} and {quiz}',
#     input_variables = ['notes', 'quiz']
# )
# parser = StrOutputParser()

# parallel_chain = RunnableParallel({
#     "notes": prompt1 | model1 | parser,
#     "quiz": prompt2 | model2 | parser
# })

# merge_chain = prompt3 | model1 | parser

# chain = parallel_chain | merge_chain

# result = chain.invoke({"text": "Langchain is a framework for developing applications powered by language models. It provides a standard interface for all types of language models, as well as tools to connect them to other sources of data and to manage their interactions."})

# print("Final Output:\n", result)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv  

load_dotenv()

# LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=200
)

model1 = ChatHuggingFace(llm=llm)
model2 = ChatHuggingFace(llm=llm)

# Prompts
prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text:\n{text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text:\n{text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document:\n\nNotes:\n{notes}\n\nQuiz:\n{quiz}',
    input_variables=['notes', 'quiz']
)

# Parser
parser = StrOutputParser()

# Chains
notes_chain = prompt1 | model1 | parser
quiz_chain = prompt2 | model2 | parser

parallel_chain = RunnableParallel(
    notes=notes_chain,
    quiz=quiz_chain
)

# Merge
merge_chain = prompt3 | model1 | parser

# Final chain
chain = parallel_chain | merge_chain

# Run
result = chain.invoke({
    "text": "Langchain is a framework for developing applications powered by language models. It provides a standard interface for all types of language models, as well as tools to connect them to other sources of data and to manage their interactions."
})

print("Final Output:\n", result)