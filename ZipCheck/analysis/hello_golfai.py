import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from openai import AzureOpenAI

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from config.azure_config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_DEPLOYMENT
)

print("")
print("CONNECTING TO GOLFAI...")
print("")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2025-04-01-preview"
)

response = client.chat.completions.create(
    model=AZURE_OPENAI_DEPLOYMENT,
    messages=[
        {
            "role": "user",
            "content": "Say Hello GolfAI and one short sentence about golf."
        }
    ]
)

print("")
print("RESPONSE")
print("--------")
print("")

print(
    response.choices[0].message.content
)