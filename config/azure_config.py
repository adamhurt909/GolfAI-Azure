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
    print(
        "WARNING: AZURE_OPENAI_ENDPOINT not configured."
    )

if not AZURE_OPENAI_KEY:
    print(
        "WARNING: AZURE_OPENAI_KEY not configured."
    )

if not AZURE_OPENAI_DEPLOYMENT:
    print(
        "WARNING: AZURE_OPENAI_DEPLOYMENT not configured."
    )