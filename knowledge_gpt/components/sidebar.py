import streamlit as st

from knowledge_gpt.components.faq import faq
from dotenv import load_dotenv
import os

load_dotenv()

PROVIDER_LIST=["OPENAI", "OLLAMA"]

def sidebar():
    with st.sidebar:
        
        # Provider selection
        provider = st.selectbox("LLM Provider", options=PROVIDER_LIST)

        # Provider environment variable
        # provider = os.getenv("LLM_PROVIDER", "OPENAI")

        if provider == "OPENAI":
            st.markdown(
                "## How to use\n"
                "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"  # noqa: E501
                "2. Upload a pdf, docx, or txt file📄\n"
                "3. Ask a question about the document💬\n"
            )
            api_key_input = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="Paste your OpenAI API key here (sk-...)",
                help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
                value=os.environ.get("OPENAI_API_KEY", None)
                or st.session_state.get("OPENAI_API_KEY", ""),
            )
            st.session_state["LLM_PROVIDER"] = "OPENAI"
            st.session_state["OPENAI_API_KEY"] = api_key_input
            faq()
            st.markdown("---")
        elif provider == "OLLAMA":
            ollama_base_url = st.text_input(
                "Ollama Base URL",
                type="default",
                placeholder="Paste your ollama base url here (http://localhost:11434)",
                value="http://localhost:11434"
                or st.session_state.get("OLLAMA_BASE_URL", ""),
            )
            st.session_state["LLM_PROVIDER"] = "OLLAMA"
            st.session_state["OLLAMA_BASE_URL"] = ollama_base_url

        st.markdown("# About")
        st.markdown(
            "📖KnowledgeGPT allows you to ask questions about your "
            "documents and get accurate answers with instant citations. "
        )
        st.markdown(
            "This tool is a work in progress. "
            "You can contribute to the project on [GitHub](https://github.com/mmz-001/knowledge_gpt) "  # noqa: E501
            "with your feedback and suggestions💡"
        )
        st.markdown("Made by [mmz_001](https://twitter.com/mm_sasmitha)")
        st.markdown("---")
