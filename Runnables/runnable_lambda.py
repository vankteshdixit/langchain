from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnableSequence,
    RunnablePassthrough,
    RunnableLambda
)
from dotenv import load_dotenv

load_dotenv()

def word_count(text):
    return len(text.split())

# runnable_word_count = RunnableLambda(word_counter)

# print(runnable_word_count.invoke("Hello world, this is a test"))

prompt = PromptTemplate(
    template = 'Write a Joke about a {topic}',
    input_variables = ['topic']
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct", 
    task="text-generation",
)

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

# parallel_chain = RunnableParallel({
#     'joke': RunnablePassthrough(),
#     'word_count': RunnableLambda(word_count)
# })

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(lambda x: len(x.split()))
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({"topic": "cat"}))