import streamlit as st
from knowledge_gpt.components.sidebar import sidebar

from knowledge_gpt.ui import (
    wrap_doc_in_html,
    is_query_valid,
    is_file_valid,
    is_open_ai_key_valid,
    display_file_read_error,
)

from knowledge_gpt.core.caching import bootstrap_caching

from knowledge_gpt.core.parsing import read_file
from knowledge_gpt.core.chunking import chunk_file
from knowledge_gpt.core.embedding import embed_files, get_model_list
from knowledge_gpt.core.qa import query_folder
from knowledge_gpt.core.utils import get_llm


VECTOR_STORE = "faiss"
MODEL_LIST = ["gpt-3.5-turbo", "gpt-4"]
latest_success_message = None

# Uncomment to enable debug mode
# MODEL_LIST.insert(0, "debug")

st.set_page_config(page_title="Playground", page_icon="📖", layout="wide")
st.header("📖Embedding Playground")

# Enable caching for expensive functions
bootstrap_caching()

sidebar()
EMBEDDING = str(st.session_state["LLM_PROVIDER"]).lower()

uploaded_file = st.file_uploader(
    "Upload a pdf, docx, or txt file",
    type=["pdf", "docx", "txt"],
    help="Scanned documents are not supported yet!",
)

if EMBEDDING == "openai":
    ollama_base_url = None
    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    if not openai_api_key:
        st.warning(
            "Enter your OpenAI API key in the sidebar. You can get a key at"
            " https://platform.openai.com/account/api-keys."
        )
elif EMBEDDING == "ollama":
    openai_api_key = None
    ollama_base_url = st.session_state.get("OLLAMA_BASE_URL")
    if not ollama_base_url:
        st.warning("Enter your Ollama base URL in the sidebar.")

if not uploaded_file:
    st.stop()

# Embedding metadata for persistence
st.subheader("Embedding Settings")
# TODO: Persist the index by the source name for later chat selection
source_name = st.text_input("Source Name", placeholder="Give your knowledge a name!")
model_list = []
try:
    model_list = get_model_list(EMBEDDING, openai_api_key, ollama_base_url)
except Exception as e:
    st.error(f"error fetching models: {e}")
model: str = st.selectbox("Model", options=model_list)
# TODO: Support different loader here
loader = st.selectbox("Loader", options=["Text", "CSV"])
with st.expander("Advanced Options"):
    chunk_size = st.number_input("Chunk size", min_value=0, value=500, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, value=50, step=10)
    return_all_chunks = st.checkbox("Show all chunks retrieved from vector search", value=True)
    show_full_doc = st.checkbox("Show parsed contents of the document")

if EMBEDDING == "openai" and not is_open_ai_key_valid(openai_api_key, model):
    st.stop()

with st.spinner("Chunking document..."):
    try:
        file = read_file(uploaded_file)
    except Exception as e:
        display_file_read_error(e, file_name=uploaded_file.name)

    chunked_file = chunk_file(file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

if not is_file_valid(file) or source_name == "":
    st.stop()

latest_success_message = st.success("Document chunking completed successfully!")

with st.spinner("Indexing document... This may take a while⏳"):
    # TODO: Trigger the embeddding by a button click, not by file upload or name change
    folder_index = embed_files(
        files=[chunked_file],
        embedding=EMBEDDING if model != "debug" else "debug",
        vector_store=VECTOR_STORE if model != "debug" else "debug",
        model=model,
        ollama_embedding_url=ollama_base_url,
        openai_api_key=openai_api_key,
    )
    st.session_state["FOLDER_INDEX"] = folder_index
    latest_success_message.empty()
    # TODO: Create general function to display success message, at the bottom of the page
    latest_success_message = st.success("Document indexing completed successfully!")

query = st.text_area("Ask a question about the document")
k = st.number_input(label="TOP_K", min_value=1, max_value=10, value=5, step=1)
chat_model = st.selectbox("Chat Model", options=model_list)
submit = st.button("Submit")

if show_full_doc:
    with st.expander("Document"):
        # Hack to get around st.markdown rendering LaTeX
        st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)


if submit:
    with st.spinner(f"Waiting for answer now..."):
        if not is_query_valid(query):
            st.stop()

        # Output Columns
        answer_col, sources_col = st.columns(2)
        llm = get_llm(model=chat_model, provider=EMBEDDING, openai_api_key=openai_api_key, ollama_base_url=ollama_base_url, temperature=0)
        result = query_folder(
            folder_index=st.session_state["FOLDER_INDEX"],
            query=query,
            return_all=return_all_chunks,
            llm=llm,
            top_k=k,
        )

        with answer_col:
            st.markdown("#### Answer")
            st.markdown(result.answer)

        with sources_col:
            st.markdown("#### Sources")
            for source in result.sources:
                st.markdown(source.page_content)
                st.markdown(source.metadata["source"])
                st.markdown("---")
