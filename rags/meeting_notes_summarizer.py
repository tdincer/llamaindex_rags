"""
summarize_with_ollama.py

A document summarization pipeline using LlamaIndex with Ollama-powered models.
This script loads input documents, optionally splits them into chunks (depending
on the configured chunk size), builds a summary index, and queries it to produce
a concise yet informative summary.

Ollama must be running locally at http://localhost:11434 before execution.

Note:
    The SentenceSplitter here uses a very large chunk size (10**6), which means
    most documents will not actually be split. To enable finer-grained processing
    for large texts, reduce the chunk size value. Adjust the model and embedding
    names as needed to match your Ollama setup.
"""

import logging
import sys
from typing import Union
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, DocumentSummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "llama3.2:latest"
DEFAULT_EMBED_MODEL = "embeddinggemma:latest"
DEFAULT_CHUNK_SIZE = 10**6

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main(input_file: Union[str, Path]):
    """Generate a document summary using LlamaIndex and Ollama models.

    Args:
        input_path: Path to a file or directory containing the document(s) to summarize.

    Returns:
        llama_index.core.base.response.schema.Response:
            The complete LlamaIndex Response object containing the summary text
            and any associated metadata.

    Raises:
        FileNotFoundError: If the provided path does not exist.
    """

    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logging.info(f"Loading document: {input_file}")

    logging.info("Initializing Ollama LLM and embedding models...")
    llm = Ollama(
        model="llama3.2:latest",
        base_url=OLLAMA_BASE_URL,
        request_timeout=1024,
        temperature=0.3,
    )

    embedding = OllamaEmbedding(
        model_name="embeddinggemma:latest",
        base_url=OLLAMA_BASE_URL,
        request_timeout=1024,
    )

    logging.info("Loading input documents...")
    reader = SimpleDirectoryReader(input_files=[input_file])
    documents = reader.load_data()

    splitter = SentenceSplitter(chunk_size=10**6)
    whole_docs = splitter.get_nodes_from_documents(documents)

    logging.info("Building document summary index...")
    doc_summary_index = DocumentSummaryIndex(
        whole_docs,
        llm=llm,
        transformations=[splitter],
        embed_model=embedding,
    )
    query_engine = doc_summary_index.as_query_engine(select_top_k=1, llm=llm)

    logging.info("Querying summary from index...")
    response = query_engine.query(
        "Provide a detailed yet concise summary of the document."
    )
    return response


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error(
            "Usage: python summarize_with_ollama.py <path_to_document_or_directory>"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    summary = main(input_file)

    print("\n===== DOCUMENT SUMMARY =====\n")
    print(summary)
