"""
app.py
------
Main Streamlit entry point for the AI Resume Analyzer & Career Coach.

Responsibilities:
    - Configure the Streamlit page (title, layout, sidebar state).
    - Initialize session_state via utils.session_state.init_session_state().
    - Ensure local data directories exist (config.ensure_directories()).
    - Render the sidebar: branding, navigation, resume/JD status, and
      a "Start Over" control.
    - Dispatch to the selected page's render() function.

Each page module under ui/ exposes a single `render() -> None` function
that draws that page's content using st.* calls and reads/writes state
exclusively through utils.session_state helpers.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from config import (
    APP_TITLE,
    APP_SUBTITLE,
    PAGE_LAYOUT,
    INITIAL_SIDEBAR_STATE,
    NAV_PAGES,
    PAGE_UPLOAD,
    PAGE_ATS,
    PAGE_JD_MATCH,
    PAGE_SUGGESTIONS,
    PAGE_INTERVIEW,
    PAGE_ROADMAP,
    PAGE_CHATBOT,
    ensure_directories,
)
from utils.session_state import (
    init_session_state,
    reset_session_state,
    set_current_page,
    get_current_page,
    resume_is_uploaded,
    jd_is_provided,
    get_uploaded_file_name,
    get_debug_snapshot,
)

# ---------------------------------------------------------------------------
# Page modules
# ---------------------------------------------------------------------------
# Each module exposes a `render()` function. Imported up front (rather
# than dynamically) so import errors surface immediately on startup
# rather than only when a user navigates to a broken page.
from ui import upload_page
from ui import ats_page
from ui import jd_match_page
from ui import suggestions_page
from ui import interview_page
from ui import roadmap_page
from ui import chatbot_page


# Maps each NAV_PAGES label to its render function.
PAGE_RENDERERS = {
    PAGE_UPLOAD: upload_page.render,
    PAGE_ATS: ats_page.render,
    PAGE_JD_MATCH: jd_match_page.render,
    PAGE_SUGGESTIONS: suggestions_page.render,
    PAGE_INTERVIEW: interview_page.render,
    PAGE_ROADMAP: roadmap_page.render,
    PAGE_CHATBOT: chatbot_page.render,
}


# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
def configure_page() -> None:
    """
    Set Streamlit's global page configuration.

    Must be called before any other st.* command. Values are sourced
    from config.py so branding/layout can be changed in one place.
    """
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧭",
        layout=PAGE_LAYOUT,
        initial_sidebar_state=INITIAL_SIDEBAR_STATE,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    """
    Render the sidebar: branding, status indicators, navigation, and
    the "Start Over" control.

    Returns:
        The currently selected page name (one of NAV_PAGES).
    """
    with st.sidebar:
        st.title(APP_TITLE)
        st.caption(APP_SUBTITLE)

        st.divider()

        # --- Status indicators -------------------------------------
        if resume_is_uploaded():
            filename = get_uploaded_file_name() or "resume.pdf"
            st.success(f"Resume loaded: **{filename}**", icon="📄")
        else:
            st.info("No resume uploaded yet.", icon="📄")

        if jd_is_provided():
            st.success("Job description provided.", icon="📋")
        else:
            st.caption("No job description entered yet.")

        st.divider()

        # --- Navigation -----------------------------------------------
        current = get_current_page()
        default_index = NAV_PAGES.index(current) if current in NAV_PAGES else 0

        selected_page = st.radio(
            "Navigate",
            options=NAV_PAGES,
            index=default_index,
            label_visibility="collapsed",
        )
        set_current_page(selected_page)

        st.divider()

        # --- Start over -------------------------------------------------
        if st.button("🔄 Start Over", use_container_width=True):
            reset_session_state(keep_resume=False)
            st.rerun()

        # --- Debug expander (developer aid) -----------------------------
        with st.expander("🛠️ Debug: session state"):
            st.json(get_debug_snapshot())

    return selected_page


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Application entry point: configure the page, initialize state,
    render the sidebar, and dispatch to the selected page.
    """
    configure_page()
    ensure_directories()
    init_session_state()

    selected_page = render_sidebar()

    st.title(selected_page)

    renderer = PAGE_RENDERERS.get(selected_page)
    if renderer is None:
        st.error(f"No page registered for '{selected_page}'.")
        return

    renderer()


if __name__ == "__main__":
    main()
