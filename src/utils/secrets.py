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
