import os
import json
import uuid
import tiktoken
import pandas as pd
import asyncio
import aiofiles
from dotenv import load_dotenv
from azure.identity.aio import DefaultAzureCredential
from azure.cosmos.aio import CosmosClient
from azure.cosmos import PartitionKey, exceptions

from asyncio import Semaphore
import books_processing.openai_helper as openai_helper

load_dotenv()

# Constants
MAX_TOKENS = 8192
EMBED_BATCH_SIZE = 16
MAX_CONCURRENT_TASKS = 4
COSMOS_BATCH_SIZE = 100
MAX_BATCH_BYTES = 2 * 1024 * 1024
PARTITION_KEY_VALUE = "books_items"
STAGING_FILE = "staging_embeddings.jsonl"

encoding = tiktoken.encoding_for_model(openai_helper.embedding_model)

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



async def embed_batch(batch_files, csv_path, semaphore):
    async with semaphore:
        texts, meta = [], []
        print(f"🔄 Processing batch: {batch_files[0]} to {batch_files[-1]}")
        for file in batch_files:
            try:
                df = pd.read_csv(os.path.join(csv_path, file))
                text = df["text"].iloc[0]
                tokens = encoding.encode(text)
                if len(tokens) > MAX_TOKENS:
                    tokens = tokens[:MAX_TOKENS]
                    text = encoding.decode(tokens)
                texts.append(text)
                meta.append((file, text))
            except Exception as e:
                print(f"Error reading {file}: {e}")

        if not texts:
            return []

        try:
            embeddings = await openai_helper.get_embeddings(texts)
            return [
                {
                    "id": str(uuid.uuid4()),
                    "partitionKey": PARTITION_KEY_VALUE,
                    "fileName": file,
                    "text": text,
                    "textVector": list(embedding.embedding)
                }
                for (file, text), embedding in zip(meta, embeddings.data)
            ]
        except Exception as e:
            print(f"Embedding error for batch starting with {batch_files[0]}: {e}")
            return []


async def stage_embeddings():
    csv_path = "all_golden_data_csv"
    files = sorted(os.listdir(csv_path))
    semaphore = Semaphore(MAX_CONCURRENT_TASKS)
    tasks = []

    for i in range(0, len(files), EMBED_BATCH_SIZE):
        batch_files = files[i:i + EMBED_BATCH_SIZE]
        tasks.append(embed_batch(batch_files, csv_path, semaphore))

    results = await asyncio.gather(*tasks)

    async with aiofiles.open(STAGING_FILE, "w") as f:
        for batch in results:
            for record in batch:
                await f.write(json.dumps(record) + "\n")


async def flush_to_cosmos(container):
    async with aiofiles.open(STAGING_FILE, "r") as f:
        items, size = [], 0
        async for line in f:
            item = json.loads(line)
            item_size = len(line.encode("utf-8"))

            if len(items) >= COSMOS_BATCH_SIZE or size + item_size > MAX_BATCH_BYTES:
                await execute_batch(container, items)
                items, size = [], 0

            items.append(item)
            size += item_size

        if items:
            await execute_batch(container, items)


async def execute_batch(container, items):
    try:
        operations = [("upsert", (item,)) for item in items]
        response = await container.execute_item_batch(batch_operations=operations, partition_key=PARTITION_KEY_VALUE)

        for idx, res in enumerate(response):
            status = res["statusCode"]
            if status not in (200, 201):
                print(f"Item {idx + 1} failed: {status}")
            else:
                print(f"✅ Item {idx + 1} succeeded: {status}")
    except Exception as e:
        print(f"Batch execution error: {e}")


async def init_cosmos():
    cosmosdb_endpoint = os.environ["AZURE_COSMOSDB_ENDPOINT"]
    database_name = "booksdb"
    container_name = "books_perftest"

    client = CosmosClient(cosmosdb_endpoint, credential=DefaultAzureCredential())
    await client.create_database_if_not_exists(id=database_name)
    db = client.get_database_client(database_name)

    try:
        container = await db.create_container(
            id=container_name,
            partition_key=PartitionKey(path="/partitionKey"),
            offer_throughput=10000
        )
    except exceptions.CosmosResourceExistsError:
        container = db.get_container_client(container_name)

    return container


async def main():
    print("Starting embedding and staging...")
    await stage_embeddings()
    print("Embedding complete. Now writing to Cosmos DB...")
    container = await init_cosmos()
    await flush_to_cosmos(container)
    print("🎉 All data successfully written to Cosmos DB.")


if __name__ == "__main__":
    asyncio.run(main())
