"""
llm.py
------
Module responsible for all interactions with the local LLM served by
LM Studio via its OpenAI-compatible REST API
(typically http://localhost:1234/v1).

Provides:
    - LLMClient: a reusable, configurable client wrapping the
      /chat/completions endpoint, with error handling.
    - High-level task functions:
        - generate_resume_review()
        - generate_interview_questions()
        - generate_career_roadmap()
        - chat_with_resume()

Design notes:
    - No external/paid APIs are used. All requests go to a local
      LM Studio server over HTTP.
    - All task functions return plain strings (model output) and
      never raise on LLM/network failure — instead they return a
      clearly marked error message string, so the Streamlit UI can
      display it directly without crashing.
    - System prompts are merged into the first user message so the
      module works with models whose jinja templates only support
      "user" and "assistant" roles (e.g. Mistral, Phi-3). This is
      the safest approach for local models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import requests

from config import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_MODEL,
    LLM_TIMEOUT_SECONDS,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_MAX_TOKENS_REVIEW,
    LLM_MAX_TOKENS_INTERVIEW,
    LLM_MAX_TOKENS_ROADMAP,
    LLM_MAX_TOKENS_CHAT,
)


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL: str = LM_STUDIO_BASE_URL
DEFAULT_MODEL: str = LM_STUDIO_MODEL
DEFAULT_TEMPERATURE: float = LLM_TEMPERATURE
DEFAULT_MAX_TOKENS: int = LLM_MAX_TOKENS
DEFAULT_TIMEOUT_SECONDS: int = LLM_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class LLMConnectionError(Exception):
    """
    Raised internally when the LM Studio server cannot be reached or
    returns an unexpected/invalid response.
    """


# ---------------------------------------------------------------------------
# Reusable LLM client
# ---------------------------------------------------------------------------
@dataclass
class LLMClient:
    """
    A reusable client for talking to a local LM Studio server via its
    OpenAI-compatible /chat/completions endpoint.

    System prompts are merged into the first user message because many
    local models (Mistral, Phi-3, Gemma, etc.) use jinja chat templates
    that only accept "user" and "assistant" roles and return HTTP 400
    when a "system" role message is present.
    """

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS

    _endpoint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._endpoint = f"{self.base_url.rstrip('/')}/chat/completions"

    @staticmethod
    def _merge_system_into_user(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert any "system" role messages into "user" role messages by
        prepending their content to the next user message (or creating a
        standalone user message if none follows immediately).

        This ensures compatibility with local models whose jinja prompt
        templates only support "user" and "assistant" roles.

        Args:
            messages: The original list of chat messages.

        Returns:
            A new list of messages with no "system" role entries.
        """
        result: List[Dict[str, str]] = []
        pending_system: str = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                # Accumulate system content to prepend to next user msg.
                pending_system = (
                    f"{pending_system}\n{content}".strip()
                    if pending_system
                    else content
                )
            elif role == "user":
                if pending_system:
                    # Prepend system instruction to this user message.
                    merged_content = (
                        f"[Instructions]: {pending_system}\n\n"
                        f"[Request]: {content}"
                    )
                    result.append({"role": "user", "content": merged_content})
                    pending_system = ""
                else:
                    result.append({"role": "user", "content": content})
            else:
                # "assistant" and any other roles pass through unchanged.
                if pending_system:
                    # Flush pending system content as a standalone user msg.
                    result.append({"role": "user", "content": pending_system})
                    pending_system = ""
                result.append({"role": role, "content": content})

        # Any remaining system content becomes a final user message.
        if pending_system:
            result.append({"role": "user", "content": pending_system})

        return result

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request to the local LM Studio server.

        Automatically merges system messages into user messages for
        compatibility with models that don't support the system role.

        Args:
            messages: A list of OpenAI-style chat messages.
            temperature: Optional override for sampling temperature.
            max_tokens: Optional override for max tokens to generate.

        Returns:
            The generated text content from the model's first choice.

        Raises:
            LLMConnectionError: If the request fails for any reason.
        """
        # Merge system messages so all local models work correctly.
        safe_messages = self._merge_system_into_user(messages)

        payload = {
            "model": self.model,
            "messages": safe_messages,
            "temperature": (
                temperature if temperature is not None else self.temperature
            ),
            "max_tokens": (
                max_tokens if max_tokens is not None else self.max_tokens
            ),
            "stream": False,
        }

        try:
            response = requests.post(
                self._endpoint,
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise LLMConnectionError(
                f"Could not connect to LM Studio at {self.base_url}. "
                "Make sure LM Studio is running and the local server "
                "is started."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LLMConnectionError(
                f"Request to LM Studio timed out after {self.timeout} "
                "seconds. The model may be too slow or stuck."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMConnectionError(
                f"Unexpected error while contacting LM Studio: {exc}"
            ) from exc

        if response.status_code != 200:
            raise LLMConnectionError(
                f"LM Studio returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
            raise LLMConnectionError(
                f"Received an unexpected response format from LM Studio: {exc}"
            ) from exc

        return content.strip()


# ---------------------------------------------------------------------------
# Default shared client instance
# ---------------------------------------------------------------------------
_default_client: LLMClient = LLMClient()


def get_default_client() -> LLMClient:
    """Get the module-level default LLMClient instance."""
    return _default_client


def set_default_client(client: LLMClient) -> None:
    """Replace the module-level default LLMClient instance."""
    global _default_client
    _default_client = client


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------
def _safe_generate(
    client: Optional[LLMClient],
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Run a chat completion, converting LLMConnectionError into a
    "[LLM Error] ..." string rather than raising.
    """
    active_client = client or get_default_client()
    try:
        return active_client.chat_completion(
            messages, temperature=temperature, max_tokens=max_tokens
        )
    except LLMConnectionError as exc:
        return f"[LLM Error] {exc}"


def _resume_to_text(resume_data: Dict[str, Union[str, List[str]]]) -> str:
    """
    Convert a structured resume dictionary into a readable plain-text
    block suitable for inclusion in an LLM prompt.
    """
    def _section(title: str, items: Union[str, List[str]]) -> str:
        if isinstance(items, list):
            if not items:
                return f"{title}: (none listed)"
            bullet_items = "\n".join(f"  - {item}" for item in items)
            return f"{title}:\n{bullet_items}"
        return f"{title}: {items or '(not provided)'}"

    lines = [
        _section("Name", resume_data.get("name", "")),
        _section("Email", resume_data.get("email", "")),
        _section("Phone", resume_data.get("phone", "")),
        _section("Skills", resume_data.get("skills", [])),
        _section("Education", resume_data.get("education", [])),
        _section("Projects", resume_data.get("projects", [])),
        _section("Experience", resume_data.get("experience", [])),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# High-level task functions
# ---------------------------------------------------------------------------
def generate_resume_review(
    resume_data: Dict[str, Union[str, List[str]]],
    job_description: str = "",
    client: Optional[LLMClient] = None,
) -> str:
    """
    Generate resume improvement suggestions using the local LLM.

    Args:
        resume_data: Structured resume dictionary from parser.py.
        job_description: Optional job description text.
        client: Optional LLMClient instance.

    Returns:
        The model's review as a string, or "[LLM Error] ..." on failure.
    """
    resume_text = _resume_to_text(resume_data)

    jd_section = (
        f"\n\nTarget Job Description:\n{job_description.strip()}"
        if job_description.strip()
        else ""
    )

    system_message = (
        "You are an expert resume reviewer and career coach. You give "
        "specific, actionable, and honest feedback. You do not invent "
        "facts about the candidate; you only work with the information "
        "provided."
    )

    user_message = (
        "Review the following resume and provide improvement "
        "suggestions. Cover: (1) weak or vague bullet points and how "
        "to strengthen them, (2) any missing or underdeveloped "
        "sections, (3) formatting/structure issues for ATS "
        "compatibility, and (4) alignment with the target job "
        "description if one is provided. Present your feedback as a "
        "clear, organized list.\n\n"
        f"Resume:\n{resume_text}"
        f"{jd_section}"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    return _safe_generate(client, messages, max_tokens=LLM_MAX_TOKENS_REVIEW)


def generate_interview_questions(
    resume_data: Dict[str, Union[str, List[str]]],
    job_description: str = "",
    num_questions: int = 8,
    client: Optional[LLMClient] = None,
) -> str:
    """
    Generate likely interview questions tailored to the candidate's
    resume and an optional target job description.

    Args:
        resume_data: Structured resume dictionary from parser.py.
        job_description: Optional job description text.
        num_questions: Approximate number of questions to generate.
        client: Optional LLMClient instance.

    Returns:
        The model's generated questions as a string, or "[LLM Error] ..."
        on failure.
    """
    resume_text = _resume_to_text(resume_data)

    jd_section = (
        f"\n\nTarget Job Description:\n{job_description.strip()}"
        if job_description.strip()
        else ""
    )

    system_message = (
        "You are an experienced technical interviewer and career "
        "coach. You generate realistic, role-appropriate interview "
        "questions based on a candidate's background."
    )

    user_message = (
        f"Based on the resume below"
        f"{' and the target job description' if job_description.strip() else ''}, "
        f"generate approximately {num_questions} interview questions. "
        "Include a mix of: technical/skill-based questions, project "
        "deep-dive questions, and behavioral questions. For each "
        "question, briefly note (in one line) what the interviewer is "
        "trying to assess. Format the output as a numbered list.\n\n"
        f"Resume:\n{resume_text}"
        f"{jd_section}"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    return _safe_generate(client, messages, max_tokens=LLM_MAX_TOKENS_INTERVIEW)


def generate_career_roadmap(
    resume_data: Dict[str, Union[str, List[str]]],
    target_role: str,
    client: Optional[LLMClient] = None,
) -> str:
    """
    Generate a step-by-step career roadmap toward a target role.

    Args:
        resume_data: Structured resume dictionary from parser.py.
        target_role: The job title the candidate wants to work toward.
        client: Optional LLMClient instance.

    Returns:
        The model's generated roadmap as a string, or "[LLM Error] ..."
        on failure.
    """
    resume_text = _resume_to_text(resume_data)

    system_message = (
        "You are a career coach who creates realistic, actionable "
        "career development roadmaps based on a person's current "
        "skills and experience."
    )

    user_message = (
        f"The candidate's goal is to become a '{target_role}'. Based "
        "on their current resume below, create a career roadmap with "
        "clearly labeled stages (e.g. Stage 1, Stage 2, Stage 3...). "
        "For each stage, specify: the skills/technologies to learn, "
        "suggested projects or certifications, and an estimated "
        "timeframe. Keep the roadmap realistic and tailored to their "
        "current level shown in the resume.\n\n"
        f"Resume:\n{resume_text}"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    return _safe_generate(client, messages, max_tokens=LLM_MAX_TOKENS_ROADMAP)


def chat_with_resume(
    question: str,
    context_chunks: List[str],
    chat_history: Optional[List[Dict[str, str]]] = None,
    client: Optional[LLMClient] = None,
) -> str:
    """
    Answer a user's question about their resume using retrieved resume
    chunks as context (RAG).

    Args:
        question: The user's question about their resume.
        context_chunks: Relevant resume text chunks from VectorStore.search().
        chat_history: Optional prior conversation turns.
        client: Optional LLMClient instance.

    Returns:
        The model's answer as a string, or "[LLM Error] ..." on failure.
    """
    context_text = (
        "\n---\n".join(chunk.strip() for chunk in context_chunks if chunk.strip())
        or "(No relevant resume content was found.)"
    )

    system_message = (
        "You are a helpful assistant that answers questions about a "
        "specific person's resume. Use ONLY the provided resume "
        "context to answer. If the answer is not contained in the "
        "context, say that the resume does not provide that "
        "information rather than guessing."
    )

    user_message = (
        f"Resume context:\n{context_text}\n\n"
        f"Question: {question.strip()}"
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_message}]

    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": user_message})

    return _safe_generate(client, messages, max_tokens=LLM_MAX_TOKENS_CHAT)


# ---------------------------------------------------------------------------
# Manual test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_resume: Dict[str, Union[str, List[str]]] = {
        "name": "Jane Doe",
        "email": "jane@example.com",
        "phone": "+1 555 123 4567",
        "skills": ["Python", "Machine Learning", "FAISS", "Streamlit"],
        "education": ["B.Tech in Computer Science, 2024"],
        "projects": ["Built a resume analyzer using NLP and FAISS"],
        "experience": ["Data Science Intern at Acme Corp, worked on ML pipelines"],
    }

    print("=== Resume Review ===")
    print(generate_resume_review(sample_resume))

    print("\n=== Interview Questions ===")
    print(generate_interview_questions(sample_resume, num_questions=5))

    print("\n=== Career Roadmap ===")
    print(generate_career_roadmap(sample_resume, target_role="Machine Learning Engineer"))

    print("\n=== Resume Chatbot ===")
    print(
        chat_with_resume(
            "What programming languages does this candidate know?",
            context_chunks=["Skills: Python, Machine Learning, FAISS, Streamlit"],
        )
    )