import os


AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT"
)

AZURE_OPENAI_KEY = os.getenv(
    "AZURE_OPENAI_KEY"
)

AZURE_OPENAI_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_DEPLOYMENT"
)


if not AZURE_OPENAI_ENDPOINT:
    raise RuntimeError(
        "AZURE_OPENAI_ENDPOINT is not configured."
    )

if not AZURE_OPENAI_KEY:
    raise RuntimeError(
        "AZURE_OPENAI_KEY is not configured."
    )

if not AZURE_OPENAI_DEPLOYMENT:
    raise RuntimeError(
        "AZURE_OPENAI_DEPLOYMENT is not configured."
    )