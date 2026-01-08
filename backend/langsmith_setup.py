"""
LangSmith OpenAI client setup utility.

Provides a centralized function to initialize OpenAI client with LangSmith tracing.
"""

import os
from openai import OpenAI


def setup_openai_client() -> OpenAI:
    """
    Initialize OpenAI client with LangSmith tracing.

    Returns:
        OpenAI: Configured OpenAI client, wrapped with LangSmith if tracing is enabled.

    Raises:
        RuntimeError: If LANGSMITH_API_KEY environment variable is not set.
    """
    # Initialize base OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Setup LangSmith tracing - REQUIRED
    if not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError("LANGSMITH_API_KEY environment variable is required but not set")

    if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
        from langsmith.wrappers import wrap_openai

        # Set project name if specified, otherwise use default
        project_name = os.getenv("LANGSMITH_PROJECT", "arcscan")
        os.environ["LANGSMITH_PROJECT"] = project_name

        client = wrap_openai(client)
        print(f"✓ LangSmith tracing enabled for project: {project_name}")
    else:
        print("⚠ LangSmith tracing is disabled (LANGSMITH_TRACING=false)")

    return client
