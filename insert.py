import os
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

if __name__ == "__main__":
    print("인덱싱")
    embeddings = OllamaEmbeddings(model="llama-3-ko")
    llm = ChatOllama(model="llama-3-ko")

    markdown_path = "data/fire.md"
    # document가 여러 종류를 사용할 수 있다.
    loader = UnstructuredMarkdownLoader(markdown_path)
    # loader = TextLoader("notion.txt")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")
