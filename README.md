## Prerequisites
- Python 3.11 or higher if using venv
- Azure Subscription with Azure Cosmos DB provisioned
- Enable Vector Search and Full Text Search on your Azure Cosmos DB account
- Azure CLI installed. If you don't have it installed, please follow the [Azure CLI installation guide](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).
- Azure OpenAI Service provisioned with embedding model - text-embedding-ada-002
- Please review [Azure Cosmos DB Vector Search](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/vector-search) documentation for more details.



## Authentication to Azure Cosmos DB and Azure OpenAI Service

Cosmos DB and Azure OpenAI Service authentication is done using the `azure-identity` library. The code uses `DefaultAzureCredential` which supports multiple authentication methods including environment variables, managed identity, and Azure CLI.

For simplicity, you can use az login to authenticate your Azure CLI session, which will allow the `DefaultAzureCredential` to pick up the credentials automatically.


## Endpoints

Update the .env.example file with your Azure Cosmos DB and Azure OpenAI Service endpoints and keys. Rename the file to `.env` after updating it.


```bash
AZURE_OPENAI_ENDPOINT=https://<your-openai-resource-name>.openai.azure.com/
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_COSMOSDB_ENDPOINT=https://<your-cosmosdb-account-name>.documents.azure.com:443/
```



## Notes about Vector Search in Azure Cosmos DB


- The flat and quantizedFlat index types uses Azure Cosmos DB's index to store and read each vector when performing a vector search. Vector searches with a flat index are brute-force searches and produce 100% accuracy or recall. That is, it's guaranteed to find the most similar vectors in the dataset. However, there's a limitation of 505 dimensions for vectors on a flat index.

- The quantizedFlat index stores quantized (compressed) vectors on the index. Vector searches with quantizedFlat index are also brute-force searches, however their accuracy might be slightly less than 100% since the vectors are quantized before adding to the index. However, vector searches with quantized flat should have lower latency, higher throughput, and lower RU cost than vector searches on a flat index. This is a good option for smaller scenarios, or scenarios where you're using query filters to narrow down the vector search to a relatively small set of vectors. quantizedFlat is recommended when the number of vectors to be indexed is somewhere around 50,000 or fewer per physical partition. However, this is just a general guideline and actual performance should be tested as each scenario can be different.

- The diskANN index is a separate index defined specifically for vectors using DiskANN, a suite of high performance vector indexing algorithms developed by Microsoft Research. DiskANN indexes can offer some of the lowest latency, highest throughput, and lowest RU cost queries, while still maintaining high accuracy. In general, DiskANN is the most performant of all index types if there are more than 50,000 vectors per physical partition.


