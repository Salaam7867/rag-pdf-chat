"""Application secret loading."""

import os
from typing import Optional

import streamlit as st


def get_groq_api_key() -> Optional[str]:
    """Read the Groq key from Streamlit secrets, then environment variables."""
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return str(key)
    except Exception:
        # Streamlit secrets may be unavailable during local development.
        pass

    return os.getenv("GROQ_API_KEY")

def get_qdrant_credentials() -> tuple[str, str]:
    """Load Qdrant Cloud credentials from Streamlit secrets or environment variables."""

    try:
        url = st.secrets.get("QDRANT_URL")
        api_key = st.secrets.get("QDRANT_API_KEY")
    except Exception:
        url = None
        api_key = None

    url = url or os.getenv("QDRANT_URL")
    api_key = api_key or os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        raise RuntimeError(
            "Qdrant Cloud credentials are missing. "
            "Configure QDRANT_URL and QDRANT_API_KEY."
        )

    return str(url), str(api_key)
