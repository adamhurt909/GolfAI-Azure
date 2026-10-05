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

#Azure GPT config, comment out below and commend in above when pushing.
#AZURE_OPENAI_ENDPOINT = "https://adamhurt909-0409-resource.services.ai.azure.com"
# 
#AZURE_OPENAI_KEY = "4UZJIXdpAWHUqAV1DyGBzWSMOl06471xI6uyVtmI0EjSOCareYQxJQQJ99CGACfhMk5XJ3w3AAAAACOG1LqM"
# 
#AZURE_OPENAI_DEPLOYMENT = "gpt-chat-latest"
 
 
 
#import os
 
#AZURE_OPENAI_ENDPOINT = os.getenv(
#    "AZURE_OPENAI_ENDPOINT"
#)
 
#AZURE_OPENAI_KEY = os.getenv(
#    "AZURE_OPENAI_KEY"
#)
 
#AZURE_OPENAI_DEPLOYMENT = os.getenv(
#    "AZURE_OPENAI_DEPLOYMENT"
#)