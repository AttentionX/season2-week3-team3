import os
from typing import Dict, List, Optional
from chromadb.db import Documents
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from dataclasses import dataclass
from utils import load_from_jsonl

RESULTS_DIR = "results"

load_dotenv()

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-ada-002"
)
id = 0

client = chromadb.PersistentClient(path="./vectordb")

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
        ids=[str(one_id) for one_id in list(range(id, id + len(documents)))],
    )
    id += len(documents)
    print(f"Successfully saved {len(documents)} documents")


def query(q: str, filter_by: Optional[Dict] = None, top_k: int = 5):
    return collection.query(
        query_embeddings=openai_ef([q]), n_results=top_k, where=filter_by
    )


def main():
    for dirpath, dirnames, filenames in os.walk(RESULTS_DIR):
        for dirname in dirnames:
            # Get the directory name as title
            title = dirname
            # Construct the full file path
            file_path = os.path.join(dirpath, dirname, "text_per_page.jsonl")
            text_per_page = load_from_jsonl(file_path)
            embed(
                [
                    Document(content=text, metadata={"page": index, "title": title})
                    for index, text in enumerate(text_per_page)
                ]
            )
