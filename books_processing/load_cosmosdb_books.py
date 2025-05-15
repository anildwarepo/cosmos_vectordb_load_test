from azure.cosmos import exceptions, PartitionKey
import json
from azure.identity import DefaultAzureCredential


import os
import pandas as pd
import uuid
import tiktoken
import asyncio
import time
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI, RateLimitError
import backoff
from dotenv import load_dotenv
from azure.cosmos.aio import CosmosClient
from asyncio import Semaphore
load_dotenv()

# Define your Cosmos DB account information
cosmosdb_endpoint = os.environ["AZURE_COSMOSDB_ENDPOINT"]

aoai_credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(aoai_credential, "https://cognitiveservices.azure.com/.default")

aoai_client = AsyncAzureOpenAI(
            api_version="2024-02-15-preview",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_ad_token_provider=token_provider
        )

embedding_model = "text-embedding-ada-002"
encoding = tiktoken.encoding_for_model(embedding_model)

MAX_TOKENS = 8192

BATCH_SIZE = 100
PARTITION_KEY_VALUE = "books_items"

MAX_RETRIES = 100
MAX_BATCH_BYTES = 2 * 1024 * 1024  # 2MB

container_name = "books_perftest"

vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path":"/textVector",
            "dataType":"float32",
            "distanceFunction":"cosine",
            "dimensions":1536
        }
    ]
}


vector_indexing_policy = {
    
    "indexingMode": "consistent",
    "automatic": True,
    "includedPaths": [
        {
            "path": "/*"
        }
    ],
    "excludedPaths": [
        {
            "path": "/_etag/?"
        },
        {
            "path": "/textVector/*"
        }
        
    ],
    "vectorIndexes": [
        {
            "path": "/textVector",
            "type": "quantizedFlat"
        }
    ]
}

full_text_paths_policy = {
   "defaultLanguage": "en-US",
   "fullTextPaths": [
       {
           "path": "/fileName",
           "language": "en-US"
       },
       {
           "path": "/text",
           "language": "en-US"
       }
   ]
}


vector_indexing_policy_diskANN = {
    
    "indexingMode": "consistent",
    "automatic": True,
    "includedPaths": [
        {
            "path": "/*"
        }
    ],
    "excludedPaths": [
        {
            "path": "/_etag/?"
        },
        {
            "path": "/textVector/*"
        }
    ],
    "fullTextIndexes": [
        {
            "path": "/text"
        }
    ],
    "vectorIndexes": [
        {
            "path": "/textVector",
            "type": "diskANN"
        }
    ]
}


# Initialize the Cosmos client
async def init_cosmos():
    database_name = "booksdb"
    

    credential = DefaultAzureCredential()
    cosmosdb_client = CosmosClient(cosmosdb_endpoint, credential=credential)

    # Create DB if it doesn't exist
    await cosmosdb_client.create_database_if_not_exists(id=database_name)

    # Get database client (not async)
    database = cosmosdb_client.get_database_client(database_name)

    # Now you can create container, upsert items, etc.
    print("✅ Connected to Cosmos DB database:", database_name)
    try:
        container = await database.create_container(
            id=container_name,
            partition_key=PartitionKey(path="/partitionKey"),
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=vector_indexing_policy_diskANN,
            full_text_policy=full_text_paths_policy,
            offer_throughput=10000
        )
        print(f"✅ Created container: {container.container_link}")
        return container
    except exceptions.CosmosResourceExistsError:
        print("Container already exists. Using existing container.")
        container = database.get_container_client(container_name)
        return container


@backoff.on_exception(
    backoff.constant,
    RateLimitError,
    interval=60,
    max_tries=MAX_RETRIES,
    jitter=None
)
async def get_embeddings_with_retry(aoai_client, texts, model):
    return await aoai_client.embeddings.create(
        input=texts,
        model=model
    )




async def start_processing1(container):
    overall_start = time.monotonic()
    count = 0
    print("Current working directory:", os.getcwd())
    csv_files_path = os.path.join('.', 'all_golden_data_csv')
    total = len(os.listdir(csv_files_path))
    start_index = 10179
    batch_items = []
    for i, file in enumerate(os.listdir(csv_files_path)):
        # if i < start_index:
        #     count += 1
        #     continue

        print(f"Processing file {i + 1} of {total}: {file}")
        file_start = time.monotonic()
        try:
            df = pd.read_csv(os.path.join(csv_files_path, file))
            text = df['text'].iloc[0]
            tokens = encoding.encode(text)

            if len(tokens) > MAX_TOKENS:
                tokens = tokens[:MAX_TOKENS]
                text = encoding.decode(tokens)

            embedding_result = await get_embeddings_with_retry(aoai_client, text, embedding_model)  #await openai_helper.generate_embeddings(text, embedding_model)
            book_item = {
                "id": str(uuid.uuid4()),
                "partitionKey": PARTITION_KEY_VALUE,
                "fileName": file,
                "text": text,
                "textVector": list(embedding_result.data[0].embedding),
            }
            batch_items.append(book_item)

            if len(batch_items) >= BATCH_SIZE:
                await flush_batch_with_execute_item_batch(container, batch_items)
                count += len(batch_items)
                batch_items = []
                file_duration = time.monotonic() - file_start
                print(f"Processed {count} so far | File: {file} | Time taken: {file_duration:.2f} seconds")

        except Exception as e:
            print(f"File processing error: {file} | Error: {e}")

    

    # Flush any remaining items
    if batch_items:
        await flush_batch_with_execute_item_batch(container, batch_items)
        count += len(batch_items)

    overall_duration = time.monotonic() - overall_start
    print(f"\n✅ Completed processing {count} files in {overall_duration:.2f} seconds")

EMBED_BATCH_SIZE = 16  # OpenAI limit
MAX_BATCH_BYTES = 2 * 1024 * 1024
MAX_PARALLEL_EMBEDDING_TASKS = 4  # Tune based on OpenAI rate limits

async def start_processing2(container):
    overall_start = time.monotonic()
    count = 0
    batch_items = []
    batch_size_bytes = 0

    print("Current working directory:", os.getcwd())
    csv_files_path = os.path.join('.', 'all_golden_data_csv')
    files = sorted(os.listdir(csv_files_path))  # sort for stability

    start_index = 10179
    files = files[start_index:]

    for i in range(0, len(files), EMBED_BATCH_SIZE):
        file_batch = files[i:i + EMBED_BATCH_SIZE]
        texts, metadata = [], []
        file_start = time.monotonic()
        for file in file_batch:
            try:
                df = pd.read_csv(os.path.join(csv_files_path, file))
                full_text = df['text'].iloc[0]
                tokens = encoding.encode(full_text)

                if len(tokens) > MAX_TOKENS:
                    tokens = tokens[:MAX_TOKENS]
                    text = encoding.decode(tokens)
                else:
                    text = full_text

                texts.append(text)
                metadata.append(file)
            except Exception as e:
                print(f"Failed to process file: {file} | Error: {e}")

        if not texts:
            continue

        try:
            embedding_result = await get_embeddings_with_retry(aoai_client, texts, embedding_model)
        except Exception as e:
            print(f"Embedding API failed: {e}")
            continue

        for text, file, embedding in zip(texts, metadata, embedding_result.data):
            item = {
                "id": str(uuid.uuid4()),
                "partitionKey": PARTITION_KEY_VALUE,
                "fileName": file,
                "text": full_text,
                "textVector": list(embedding.embedding)
            }

            item_size = len(json.dumps(item).encode("utf-8"))
            if batch_size_bytes + item_size > MAX_BATCH_BYTES:
                await flush_batch_with_execute_item_batch(container, batch_items)
                count += len(batch_items)
                batch_items = []
                batch_size_bytes = 0
                file_duration = time.monotonic() - file_start
                print(f"Processed {count} so far | Time taken: {file_duration:.2f} seconds, total_time: {time.monotonic() - overall_start:.2f} seconds")

            batch_items.append(item)
            batch_size_bytes += item_size

    # Final flush
    if batch_items:
        await flush_batch_with_execute_item_batch(container, batch_items)
        count += len(batch_items)
        file_duration = time.monotonic() - file_start
        print(f"Processed {count} so far | Time taken: {file_duration:.2f} seconds")

    overall_duration = time.monotonic() - overall_start
    print(f"\n✅ Completed processing {count} files in {overall_duration:.2f} seconds")


async def start_processing3(container):
    overall_start = time.monotonic()
    count = 0
    batch_items = []
    batch_size_bytes = 0

    print("Current working directory:", os.getcwd())
    csv_files_path = os.path.join('.', 'all_golden_data_csv')
    files = sorted(os.listdir(csv_files_path))

    start_index = 10179
    files = files[start_index:]

    semaphore = Semaphore(MAX_PARALLEL_EMBEDDING_TASKS)

    async def process_batch(file_batch):
        nonlocal batch_items, batch_size_bytes, count
        async with semaphore:
            texts, metadata = [], []
            for file in file_batch:
                file_start = time.monotonic()
                try:
                    df = pd.read_csv(os.path.join(csv_files_path, file))
                    full_text = df['text'].iloc[0]
                    tokens = encoding.encode(full_text)

                    if len(tokens) > MAX_TOKENS:
                        tokens = tokens[:MAX_TOKENS]
                        text = encoding.decode(tokens)
                    else:
                        text = full_text

                    texts.append(text)
                    metadata.append((file, text))
                except Exception as e:
                    print(f"Failed to read file {file}: {e}")

            if not texts:
                return

            try:

                embedding_result = await get_embeddings_with_retry(aoai_client, texts, embedding_model)
            except Exception as e:
                print(f"Embedding failed: {e}")
                return

            for (file, text), embedding in zip(metadata, embedding_result.data):
                item = {
                    "id": str(uuid.uuid4()),
                    "partitionKey": PARTITION_KEY_VALUE,
                    "fileName": file,
                    "text": text,
                    "textVector": list(embedding.embedding)
                }

                item_size = len(json.dumps(item).encode("utf-8"))
                if batch_size_bytes + item_size > MAX_BATCH_BYTES or len(batch_items) >= BATCH_SIZE:
                    await flush_batch_with_execute_item_batch(container, batch_items)
                    count += len(batch_items)
                    batch_items = []
                    batch_size_bytes = 0
                    file_duration = time.monotonic() - file_start
                    print(f"Processed {count} so far | Time taken: {file_duration:.2f} seconds, total_time: {time.monotonic() - overall_start:.2f} seconds")


                batch_items.append(item)
                batch_size_bytes += item_size

    # Schedule all batches
    tasks = []
    for i in range(0, len(files), EMBED_BATCH_SIZE):
        file_batch = files[i:i + EMBED_BATCH_SIZE]
        tasks.append(asyncio.create_task(process_batch(file_batch)))

    await asyncio.gather(*tasks)

    # Final flush
    if batch_items:
        file_start = time.monotonic()
        await flush_batch_with_execute_item_batch(container, batch_items)
        count += len(batch_items)
        file_duration = time.monotonic() - file_start
        print(f"Processed {count} so far | Time taken: {file_duration:.2f} seconds, total_time: {time.monotonic() - overall_start:.2f} seconds")


    overall_duration = time.monotonic() - overall_start
    print(f"\n✅ Completed processing {count} files in {overall_duration:.2f} seconds")



async def flush_batch_with_execute_item_batch(container, items):
    try:
        current_batch = []
        current_batch_size = 0

        for item in items:
            item_size = len(json.dumps(item).encode("utf-8"))

            if current_batch_size + item_size > MAX_BATCH_BYTES:
                # Flush current batch
                await execute_batch(container, current_batch)
                current_batch = []
                current_batch_size = 0

            current_batch.append(item)
            current_batch_size += item_size

        # Final flush
        if current_batch:
            await execute_batch(container, current_batch)

    except Exception as e:
        print(f"Error preparing batch: {e}")

async def execute_batch(container, items):
    try:
        batch_operations = [("upsert", (item,)) for item in items]

        response = await container.execute_item_batch(
            batch_operations=batch_operations,
            partition_key=PARTITION_KEY_VALUE
        )

        for idx, op_result in enumerate(response):
            status = op_result["statusCode"]
            if status not in (200, 201):
                print(f"⚠️ Operation {idx + 1} failed: Status {status}, Details: {op_result}")
            else:
                print(f"✅ Operation {idx + 1} succeeded: Status {status}")
    except Exception as e:
        print(f"Error executing item batch: {e}")

async def main():
    container = await init_cosmos()
    await start_processing(container)

if __name__ == "__main__":
    asyncio.run(main())



