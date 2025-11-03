from llama_index.core import (
    SimpleDirectoryReader,
    DocumentSummaryIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

OLLAMA_BASE_URL = "http://localhost:11434"


def main(input_files):
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

    reader = SimpleDirectoryReader(input_files=input_files)
    documents = reader.load_data()

    splitter = SentenceSplitter(chunk_size=10**6)
    whole_docs = splitter.get_nodes_from_documents(documents)
    doc_summary_index = DocumentSummaryIndex(
        whole_docs,
        llm=llm,
        transformations=[splitter],
        embed_model=embedding,
    )
    query_engine = doc_summary_index.as_query_engine(select_top_k=1, llm=llm)
    response = query_engine.query(
        "Provide a detailed yet concise summary of the document."
    )
    return response


if __name__ == "__main__":
    input_files = ["../sample_data/sample_meeting_notes.txt"]
    summary = main(input_files)
    print(summary)
