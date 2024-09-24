## Setup

LANGCHAIN_TRACING_V2="true"
GROQ_API_KEY = "your_api_key"
LANGCHAIN_API_KEY="your_api_key"

import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

## Creating Chain

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

llm = ChatGroq(model="llama3-8b-8192")

## Prompt pronto para RAG simples
prompt = hub.pull("rlm/rag-prompt")
# print(prompt)

chain = prompt | llm | StrOutputParser()

## Manipulating data (getting, chunking, embedding)

import bs4
from langchain_community.document_loaders import WebBaseLoader
from sentence_transformers import SentenceTransformer
import numpy as np

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
doc = loader.load()

conteudo = doc[0].page_content.split("\n")
# Retirando espaços em branco
conteudo = [s for s in conteudo if s]
# Retirando titulo e dados do autor/artigo
conteudo = conteudo[3:]
# Retirando Rodapé
conteudo = conteudo[:conteudo.index("Citation#")]

payload = []

for idx, content in enumerate(conteudo):
    item = {
        'id': idx,
        'content': content
    }
    payload.append(item)

model = SentenceTransformer(
    "all-MiniLM-L6-v2", device="cpu"
)

embeddings = model.encode([obj['content'] for obj in payload])
np.save("startup_vectors.npy", embeddings, allow_pickle=False) # Nao eh necessario, apenas para ja economizar um tempo no notebook

## Qdrant (lembra de estar com o docker funcionando a API_Key)

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(url="http://localhost:6333", api_key="testando_321")

if not client.collection_exists:
    client.create_collection(
        collection_name="teste_RAG_Langchain",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    vectors = np.load("./startup_vectors.npy")

    client.upload_collection(
        collection_name="teste_RAG_Langchain",
        vectors=vectors,
        payload=payload,
        ids=None,
        batch_size=256,
    )

## Creating and testing searching class
from neural_searcher import NeuralSearcher

neural_searcher = NeuralSearcher(collection_name="teste_RAG_Langchain")

resultado = neural_searcher.search(text="Types of Memory")
# print(resultado)

## Uniting LangChain with Qdrant
question = "What is self-reflection in the context of LLMs?"

docs_relacionados = neural_searcher.search(text=question)

string_contexto = ''
for obj in docs_relacionados:
    string_contexto += obj['content'] + "\n"

print("\n--------------- Resposta RAG ---------------")
print(chain.invoke({"context": string_contexto, "question": question}))



