import os
from typing import Any, List, Dict

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings()

    llm = ChatOpenAI(verbose=True, temperature=0)

    vectorstore = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings
    )

    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    #
    # combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    #
    # retrival_chain = create_retrieval_chain(
    #     retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    # )

    # result = retrival_chain.invoke(input={"input": query})

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=True
    )

    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    chat_history = [
        {"speaker": "user", "message": "is cool"}
    ]
    print(run_llm(query="What is ? phone number?"))
