import os
from typing import Dict, List
from chromadb.db import Documents
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from dataclasses import dataclass
from utils import load_from_jsonl

load_dotenv()

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ["OPENAI_API_KEY"],
                model_name="text-embedding-ada-002"
            )
id=0

client = chromadb.Client()

collection = client.create_collection("papers")

@dataclass
class Document:
    content: str
    metadata: Dict

def embed(documents: List[Document]):
    contents = [doc.content for doc in documents]
    collection.add(
        embeddings=openai_ef(contents),
        documents=contents,
        metadatas=[doc.metadata for doc in documents],
        ids=[str(one_id) for one_id in list(range(id, id+len(documents)))]
    )
    id += len(documents)
    print(f"Successfully saved {len(documents)} documents")
    

def query(q: str, filter: Dict, top_k: int=5):
    return collection.query(
        query_embeddings=openai_ef([q]),
        n_results=top_k,
        where=filter
    )


def main():
    text_per_page = load_from_jsonl("text_per_page.json")
    embed([Document(content=text, metadata={"page": index}) for index, text in enumerate(text_per_page)])
