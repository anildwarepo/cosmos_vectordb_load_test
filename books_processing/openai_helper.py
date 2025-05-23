import os
from openai import AsyncAzureOpenAI, RateLimitError
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
import backoff
load_dotenv()

aoai_credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(aoai_credential, "https://cognitiveservices.azure.com/.default")
aoai_client = AsyncAzureOpenAI(
    api_version="2024-02-15-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_ad_token_provider=token_provider
)
embedding_model = "text-embedding-ada-002"

@backoff.on_exception(backoff.constant, RateLimitError, interval=60, max_tries=10)
async def get_embeddings(texts, model=embedding_model):
    embeddings = await aoai_client.embeddings.create(input=texts, model=model)
    return embeddings.data[0].embedding



