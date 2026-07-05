# AI Resume Analyzer & Career Coach

A fully local, AI-powered resume analyzer and career coach. Upload a PDF
resume and get an ATS score, job-description match analysis, keyword
gap analysis, AI-generated improvement suggestions, interview
questions, a career roadmap, and a RAG-based chatbot that can answer
questions about your resume — all running on your own machine, with no
paid APIs.

## Features

1. Resume PDF upload & parsing
2. ATS score calculation
3. Job description matching (semantic + keyword)
4. Keyword gap analysis
5. AI resume improvement suggestions
6. Interview question generator
7. Career roadmap generator
8. Resume chatbot (RAG over your resume)

## Tech Stack

- **Python 3.10+**
- **Streamlit** — UI
- **LM Studio** — local LLM, OpenAI-compatible API
- **sentence-transformers** (`all-MiniLM-L6-v2`) — embeddings
- **FAISS** — vector search for the RAG chatbot
- **pypdf** — PDF text extraction
- **Plotly** — charts (ATS gauge, match score, roadmap timeline)

No database, no authentication, no paid/cloud APIs. Everything runs
locally.

## Project Structure

```
ai-resume-coach/
├── app.py                  # Streamlit entry point
├── config.py                # Central configuration
├── embeddings.py             # sentence-transformers wrapper
├── llm.py                    # LM Studio client + LLM task functions
├── requirements.txt
├── core/                      # Non-UI logic
│   ├── __init__.py
│   ├── pdf_parser.py
│   ├── resume_structurer.py
│   ├── vector_store.py
│   ├── ats_scorer.py
│   ├── jd_matcher.py
│   ├── keyword_gap.py
│   ├── suggestion_engine.py
│   ├── interview_generator.py
│   ├── career_roadmap.py
│   └── rag_chatbot.py
├── ui/                         # Streamlit pages
│   ├── __init__.py
│   ├── upload_page.py
│   ├── ats_page.py
│   ├── jd_match_page.py
│   ├── suggestions_page.py
│   ├── interview_page.py
│   ├── roadmap_page.py
│   └── chatbot_page.py
├── utils/
│   ├── __init__.py
│   ├── text_cleaning.py
│   ├── session_state.py
│   └── prompts.py
├── data/
│   ├── sample_resumes/
│   ├── sample_job_descriptions/
│   └── faiss_index/            # created automatically
└── tests/
```

## Setup

### 1. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

The first run will download the `all-MiniLM-L6-v2` embedding model
(~80 MB) from Hugging Face and cache it locally — this requires an
internet connection **once**. After that, everything runs offline.

### 2. Install and start LM Studio

1. Download LM Studio from https://lmstudio.ai and install it.
2. Download a local model (e.g. a Llama 3 8B, Mistral 7B, or Phi-3
   instruct model — anything that fits your hardware).
3. Go to the **Local Server** tab in LM Studio, load your model, and
   click **Start Server**.
4. Note the server URL (default `http://localhost:1234/v1`) and the
   exact model name shown — you'll need the model name in step 3.

### 3. Configure the app

Edit `config.py` (or set environment variables) so `LM_STUDIO_MODEL`
matches the exact model name shown in LM Studio:

```bash
export LM_STUDIO_URL="http://localhost:1234/v1"     # if not default
export LM_STUDIO_MODEL="your-model-name-here"
```

### 4. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Usage Walkthrough

1. **📄 Upload Resume** — upload a PDF resume. It's parsed and
   structured locally (skills, education, experience, projects,
   certifications, contact info).
2. **📊 ATS Score** — see your overall ATS score and a breakdown by
   category (skills, experience, projects, education, structure).
3. **🎯 JD Match** — paste a job description to get a semantic match
   score, matched/missing skills, and keyword coverage.
4. **💡 Suggestions** — get AI-generated strengths, weaknesses, missing
   sections, and improvement suggestions.
5. **🎤 Interview Prep** — generate technical, project, and behavioral
   interview questions tailored to your resume.
6. **🗺️ Career Roadmap** — enter an optional target role and get a
   staged learning roadmap with recommended technologies and projects.
7. **💬 Resume Chatbot** — ask questions about your resume; answers are
   generated using retrieval-augmented generation over your resume
   content.

## Troubleshooting

- **"[LLM Error] Could not connect to LM Studio..."** — make sure LM
  Studio's local server is running (Local Server tab → Start Server)
  and that `LM_STUDIO_BASE_URL` in `config.py` matches its address.
- **"Could not extract any text from this PDF"** — the PDF may be a
  scanned image with no embedded text layer. Try a different file or
  an OCR'd version.
- **LM Studio returns HTTP 400** — the `model` name sent by the app
  doesn't match what LM Studio expects. Set `LM_STUDIO_MODEL` to the
  exact name shown in LM Studio's Local Server tab.
- **Slow responses** — local models can be slow on CPU-only machines.
  Increase `LLM_TIMEOUT_SECONDS` in `config.py` if requests are timing
  out, or use a smaller model.

## Notes for Students / Contributors

- All scoring/matching logic (`ats_scorer.py`, `jd_matcher.py`,
  `keyword_gap.py`) is independently testable without LM Studio
  running — only `suggestion_engine.py`, `interview_generator.py`,
  `career_roadmap.py`, and `rag_chatbot.py` require LM Studio.
- `config.py` is the single source of truth for tunable values
  (weights, thresholds, token budgets, paths). Avoid hardcoding these
  elsewhere.
- `utils/session_state.py` owns every `st.session_state` key — pages
  should never access `st.session_state` directly.
