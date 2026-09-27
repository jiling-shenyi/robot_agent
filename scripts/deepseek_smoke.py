"""Make one small DeepSeek Chat Completions request to verify local config."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)


def main() -> int:
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY is missing. Set it in .env or the process environment.")
        return 2

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        timeout=30.0,
    )

    try:
        response = client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            messages=[
                {"role": "system", "content": "Reply with the single word READY."},
                {"role": "user", "content": "Check that the API connection works."},
            ],
            stream=False,
            reasoning_effort=os.getenv("DEEPSEEK_REASONING_EFFORT", "high"),
            extra_body={
                "thinking": {
                    "type": os.getenv("DEEPSEEK_THINKING", "enabled"),
                }
            },
        )
    except AuthenticationError:
        print("DeepSeek rejected the credential (HTTP 401). Check or rotate the key.")
        return 3
    except RateLimitError:
        print("DeepSeek rate-limited the request. Check quota and try again later.")
        return 4
    except APIConnectionError:
        print("Could not connect to DeepSeek. Check network access and base URL.")
        return 5
    except APIStatusError as exc:
        print(f"DeepSeek returned HTTP {exc.status_code}; request_id={exc.request_id}")
        return 6

    content = response.choices[0].message.content or ""
    print(f"DeepSeek API connection OK; model={response.model}")
    print(f"Response: {content.strip()}")
    return 0 if content.strip() else 7


if __name__ == "__main__":
    sys.exit(main())
