"""Offline startup check for the environment variables this repo actually reads.

Only booleans are returned and only variable names are printed; values are
never exposed, logged or committed.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

# Keep in sync with .env.example. These are the only variables read from the
# environment by the repository today.
ENV_VARS = ("LLM_MODEL", "RESEARCH_PAPERS_PATH")


def check_env() -> dict:
    """Report which known environment variables are set.

    Purpose:
        Quick startup check that never reveals a secret value.
    Args:
        None.
    Returns:
        dict[str, bool], e.g. ``{"LLM_MODEL": False, "RESEARCH_PAPERS_PATH": True}``.
    Owner:
        Ibraheem.
    """
    load_dotenv()
    return {name: bool(os.environ.get(name)) for name in ENV_VARS}


def main() -> None:
    """Print ``OK`` or the names of variables that are not set."""
    status = check_env()
    missing = [name for name, is_set in status.items() if not is_set]
    if missing:
        print("Not set: " + ", ".join(missing))
    else:
        print("OK")


if __name__ == "__main__":
    main()
