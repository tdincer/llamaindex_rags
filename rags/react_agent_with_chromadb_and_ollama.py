"""
react_agent_with_chromadb_and_ollama.py
---------------------------------------
Builds and runs a ReActAgent powered by Ollama’s local LLM and embedding models.
The agent connects to a persistent Chroma vector database containing pre-embedded
web content (or other indexed documents) and answers queries using that data.

Key components:
- Ollama LLM: Provides reasoning and natural language understanding.
- Ollama Embedding model: Generates vector embeddings for semantic retrieval.
- Chroma: Vector database used to store and search document embeddings.
- ReActAgent: LlamaIndex’s reasoning agent that can use tools (like query engines)
  to decide when and how to retrieve information.

Usage:
    python react_agent_with_chromadb_and_ollama.py "What topics are covered on the site?"
"""

import sys
import logging
import asyncio

import chromadb
from chromadb.config import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.vector_stores.chroma import ChromaVectorStore

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "llama3.2:latest"
DEFAULT_EMBED_MODEL = "embeddinggemma:latest"
DEFAULT_TEMPERATURE = 0.3
REQUEST_TIMEOUT = 1024
PERSIST_DIR = "../data/ollama/chroma_bs_nodes_db"
COLLECTION_NAME = "web_nodes"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


async def agent_with_tool():
    """
    Initialize a ReActAgent configured with a Chroma-based query engine.

    Steps:
    1. Instantiate the Ollama LLM and embedding models.
    2. Connect to the persistent Chroma vector database containing indexed data.
    3. Wrap the Chroma vector store with a LlamaIndex VectorStoreIndex.
    4. Create a query engine that retrieves the most relevant context from the store.
    5. Build a ReActAgent that can use this query engine as a tool.

    Returns:
        ReActAgent: A ready-to-query LlamaIndex agent using Ollama and Chroma.
    """
    logging.info("Initializing Ollama LLM and embedding models...")

    llm = Ollama(
        model=DEFAULT_LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=REQUEST_TIMEOUT,
        temperature=DEFAULT_TEMPERATURE,
    )

    embedding = OllamaEmbedding(
        model_name=DEFAULT_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=REQUEST_TIMEOUT,
    )

    # Connect to the Chroma vector store containing embedded data
    chroma_client = chromadb.PersistentClient(
        path=PERSIST_DIR, settings=Settings(anonymized_telemetry=False)
    )
    collection = chroma_client.get_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Set up storage and index wrappers for querying
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embedding,
    )

    # Create a query engine that retrieves the most relevant chunks from the index
    query_engine = vector_index.as_query_engine(select_top_k=1, llm=llm)

    # Expose the query engine as a tool for the ReAct agent
    tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="website_nodes",
        description="Query parsed website content for answering web-related questions.",
    )

    agent = ReActAgent(tools=[tool], llm=llm)
    return agent


async def main(query: str):
    """
    Run a single query asynchronously through the ReActAgent.

    Args:
        query (str): Natural language query to send to the agent.

    Returns:
        str: Agent's generated answer or reasoning output.
    """
    agent = await agent_with_tool()
    response = await agent.run(query)
    return response


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error(
            "Usage: python react_agent_with_chromadb_and_ollama.py '<your_query>'"
        )
        sys.exit(1)

    query = sys.argv[1]
    summary = asyncio.run(main(query))

    print("\n===== QUERY RESPONSE =====\n")
    print(summary)
