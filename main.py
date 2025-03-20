import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# prompt template
system_template = "영어 문장을 {language}로 번역해."
prompt = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# llm
# llm = ChatOpenAI(model="gpt-4o-mini")
# llm = ChatOllama(model="llama3.2")
llm = ChatOllama(model="llama-3-ko")

# parser
output_parser = StrOutputParser()

# LangChain Expression Language(LCEL) chaining
chain = prompt | llm | output_parser


result = chain.invoke({"language": "korean", "text": "Hi There!"})
print(result)
