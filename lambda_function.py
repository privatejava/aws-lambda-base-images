import sys

import json
import logging
import sys
import os
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index import (
    ServiceContext,
    load_index_from_storage,
    StorageContext
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.retrievers import VectorIndexRetriever
from llama_index.vector_stores.faiss import FaissVectorStore



logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

## load your model(s) into vram here

#Loading FAISS model
model_id = "BAAI/llm-embedder"
# model_id = "BAAI/bge-large-en-v1.5"

CACHE_DIR=os.environ.get('CACHE_DIR', '/mnt/langchain/cache')
FAISS_INDEX_DIR=os.environ.get('FAISS_INDEX', '')
print(f"Faiss Index: {FAISS_INDEX_DIR}")

embed_model = HuggingFaceEmbedding(model_name=model_id, cache_folder=CACHE_DIR)
print(f"Loaded hugging face Model")
service_context = ServiceContext.from_defaults( embed_model=embed_model, llm=None)
print(f"Init service context")
vector_store = FaissVectorStore.from_persist_dir(FAISS_INDEX_DIR)
print(f"Init vector store")

# Service Context
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir=FAISS_INDEX_DIR
)
print(f"Init storage ctx")
index = load_index_from_storage(storage_context=storage_context, service_context=service_context)
print(f"Loaded index ctx")


def handler(event={}, context={}):
    assert "input" in event
    # assert "type" in event["input"]
    assert "message" in event["input"]

    message = event["input"]["message"]
    response = None

    retriever = VectorIndexRetriever(
        index=index, 
        similarity_top_k=5,
    )
    nodes = retriever.retrieve(message)
    response= [] 
    for node_with_score in nodes:
        node = node_with_score.node
        response.append({
            "metadata": node.metadata,
            # "relationships":  node.relationships,
            "content":  node.text,
            "score": str(node_with_score.score)
        })

    # response = {
    #     "message":  'Hello 2 from AWS Lambda using Python' + sys.version + '!'
    # }
    return {
        "query": message,
        "result": response
    }