# Rag_Learn
Basic RAG implementation with Langchain e Qdrant

You'll need to have:
  - Docker
  - Python 3.11
  - Groq API_Key
  - Langchain API_Key

To Run Qdrant with Docker:

```
docker pull qdrant/qdrant
```


```
docker run -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

It's valid to remember that this code have an API_Key to Qdrant, but this docker above doesn't support it. To enable you'll need to go into the Qdrant Docker Files /qdrant/config/config.yaml and uncomment api_key, changing the value:

```
service:
  # Set an api-key.
  # If set, all requests must include a header with the api-key.
  # example header: `api-key: <API-KEY>`
  #
  # If you enable this you should also enable TLS.
  # (Either above or via an external service like nginx.)
  # Sending an api-key over an unencrypted channel is insecure.
  api_key: testando_321
```

If you want to continue with non autenticated API, just mantain your original docker and erase every api_key param in QdrantClient call:

```
QdrantClient(url="http://localhost:6333", api_key="testando_321")
```

to:

```
QdrantClient(url="http://localhost:6333")
```

Remember installing every library with pip!
