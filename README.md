# Zencognify

A Python package for document ingestion and processing with vector store capabilities.

## Installation

```bash
pip install zencognify
```

## Features

- Document processing from URLs (PDF, DOC, etc.)
- Vector storage with Qdrant
- Bulk document upload
- Document chunking and metadata management
- OpenAI embeddings integration

## Usage

### Basic Setup

```python
from cognify import VectorStore

vector_store = VectorStore(qdrant_url="http://localhost:6333")
```

### Create Collection

```python
vector_store.create_collection("my_documents")
```

### Bulk Upload Documents

```python
urls = [
    "https://example.com/document1.pdf",
    "https://example.com/document2.pdf"
]

results = vector_store.bulk_url_upload("my_documents", urls)
print(f"Uploaded {results['successful_uploads']} documents")
```

### Retrieve Document Chunks

```python
chunks = vector_store.get_document_chunks("my_documents", "document_id")
```

### Semantic Search

```python
results = vector_store.search("my_documents", "machine learning algorithms", top_k=5)
for doc, score in results:
    print(f"Score: {score:.3f}")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Source: {doc.metadata['source']}")
    print("---")

document_id = "your-document-id"
results = vector_store.search_by_document("my_documents", "neural networks", document_id, top_k=3)
```

### Delete Documents

```python
chunk_ids = ["chunk_id1", "chunk_id2"]
vector_store.delete_documents("my_documents", chunk_ids)
```

## Requirements

- Python 3.12+
- Qdrant vector database
- OpenAI API key (for embeddings)

## Dependencies

- langchain
- langchain-openai
- langchain-qdrant
- qdrant-client
- docling
- python-dotenv
- openai
- tiktoken

## Environment Variables

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## License

MIT License
