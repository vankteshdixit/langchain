from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableSequence,
    RunnablePassthrough
)
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Write a Joke about a {topic}',
    input_variables = ['topic']
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="text-generation",
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = 'Write a short summary on the following joke: \n {text}',
    input_variables = ['text']
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)

explanation_chain = RunnableSequence(prompt2, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': explanation_chain
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({"topic": "cat"}))