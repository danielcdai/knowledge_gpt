from langchain.vectorstores import VectorStore
from knowledge_gpt.core.parsing import File
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.base import Embeddings
from typing import List, Type, Literal
from langchain.docstore.document import Document
from knowledge_gpt.core.debug import FakeVectorStore


class FolderIndex:
    """Index for a collection of files (a folder)"""

    def __init__(self, files: List[File], index: VectorStore):
        self.name: str = "default"
        self.files = files
        self.index: VectorStore = index

    @staticmethod
    def _combine_files(files: List[File]) -> List[Document]:
        """Combines all the documents in a list of files into a single list."""

        all_texts = []
        for file in files:
            for doc in file.docs:
                doc.metadata["file_name"] = file.name
                doc.metadata["file_id"] = file.id
                all_texts.append(doc)

        return all_texts

    @classmethod
    def from_files(
        cls, files: List[File], embeddings: Embeddings, vector_store: Type[VectorStore]
    ) -> "FolderIndex":
        """Creates an index from files."""

        all_docs = cls._combine_files(files)

        index = vector_store.from_documents(
            documents=all_docs,
            embedding=embeddings,
        )

        return cls(files=files, index=index)


def embed_files(
    files: List[File], embedding: str, vector_store: str, model: str, 
    ollama_embedding_url: str | None = None, 
    **kwargs
) -> FolderIndex:
    """Embeds a collection of files and stores them in a FolderIndex."""
    print(f"Embedding: {embedding}, model: {model}, vector store: {vector_store}")
    match embedding:
        case "openai":
            from langchain.embeddings import OpenAIEmbeddings
            # TODO: Align the openai embedding model here
            _embeddings = OpenAIEmbeddings(
                **kwargs
            )
        case "ollama":
            from langchain_community.embeddings import OllamaEmbeddings
            _embeddings = OllamaEmbeddings(
                base_url=ollama_embedding_url,
                model=model,
            )
        case "debug":
            from knowledge_gpt.core.debug import FakeEmbeddings
            _embeddings = FakeEmbeddings()
        case _:
            from knowledge_gpt.core.debug import FakeEmbeddings
            _embeddings = FakeEmbeddings()

    supported_vector_stores: dict[str, Type[VectorStore]] = {
        "faiss": FAISS,
        "debug": FakeVectorStore,
    }
    if vector_store in supported_vector_stores:
        _vector_store = supported_vector_stores[vector_store]
    else:
        raise NotImplementedError(f"Vector store {vector_store} not supported.")

    return FolderIndex.from_files(
        files=files, embeddings=_embeddings, vector_store=_vector_store
    )


def get_model_list(provider: Literal['ollama', 'openai'], openai_api_key: str | None, ollama_base_url: str | None) -> List[str]:
    if provider == "ollama":
        from ollama import Client
        ollama_client = Client(host=ollama_base_url)
        return [m.model for m in ollama_client.list().models]
    elif provider == "openai":
        import openai
        openai.api_key = openai_api_key
        response = openai.Model.list()
        model_list = [model['id'] for model in response['data']]
        return model_list
    else:
        raise NotImplementedError(f"Provider {provider} not supported.")