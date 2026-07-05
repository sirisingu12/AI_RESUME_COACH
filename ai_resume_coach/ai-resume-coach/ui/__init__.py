"""
ui
--
Streamlit page modules for the AI Resume Analyzer & Career Coach.

Each module exposes a single `render() -> None` function, called by
app.py based on the sidebar navigation selection. Pages read and write
all shared state exclusively through utils.session_state helpers —
never access st.session_state directly from a page module.
"""

from __future__ import annotations
