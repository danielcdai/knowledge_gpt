from typing import List
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document

from knowledge_gpt.core.debug import FakeChatModel
from langchain.chat_models.base import BaseChatModel


def pop_docs_upto_limit(
    query: str, chain: StuffDocumentsChain, docs: List[Document], max_len: int
) -> List[Document]:
    """Pops documents from a list until the final prompt length is less
    than the max length."""

    token_count: int = chain.prompt_length(docs, question=query)  # type: ignore

    while token_count > max_len and len(docs) > 0:
        docs.pop()
        token_count = chain.prompt_length(docs, question=query)  # type: ignore

    return docs


def get_llm(model: str, provider: str, ollama_base_url: str | None, **kwargs) -> BaseChatModel:
    if model == "debug":
        return FakeChatModel()
    
    match provider:
        case "openai":
            from langchain_community.chat_models import ChatOpenAI
            return ChatOpenAI(model=model, **kwargs)  # type: ignore
        case "ollama":
            return get_ollama_llm(model=model, ollama_base_url=ollama_base_url, **kwargs)
        case _:
            raise NotImplementedError(f"Provider {provider} not supported.")

    raise NotImplementedError(f"Model {model} not supported!")


def get_ollama_llm(model: str, ollama_base_url: str, **kwargs) -> BaseChatModel:
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(base_url=ollama_base_url, model=model, **kwargs)