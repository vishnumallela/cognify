from typing import List, Dict, Any, Optional
import os
from uuid import uuid4
from datetime import datetime
from dotenv import load_dotenv
import openai

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class VectorStore:
    def __init__(self, qdrant_url: str, qdrant_api_key: Optional[str] = None):
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.qdrant_client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=200
        )
        openai.api_key = self.openai_api_key

    def create_collection(self, collection_name: str) -> None:

        try:
            collections = self.qdrant_client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            if collection_name in existing_collections:
                print(f"Collection '{collection_name}' already exists.")
                return
        except Exception as e:
            print(f"Error checking existing collections: {e}")
        

        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )
        print(f"Created collection: {collection_name}")

    def add_documents(self, collection_name: str, documents: List[str]) -> Dict[str, List[str]]:
        collection = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

        document_chunk_mapping = {}
        for doc_url in documents:
            converted = DocumentConverter()
            doc = converted.convert(doc_url).document
            markdown_content = doc.export_to_markdown()
            chunks = self.text_splitter.split_text(markdown_content)

            document_id = str(uuid4())
            base_metadata = {
                "source": doc_url,
                "filename": os.path.basename(doc_url),
                "document_id": document_id,
                "created_at": datetime.now().isoformat(),
            }

            documents_to_add = []
            chunk_ids = []

            for i, chunk in enumerate(chunks):
                chunk_uuid = str(uuid4())
                chunk_id = f"{document_id}_chunk_{i}"
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                    "chunk_id": chunk_id,
                })

                chunk_ids.append(chunk_uuid)
                documents_to_add.append(Document(page_content=chunk, metadata=chunk_metadata))

            collection.add_documents(documents=documents_to_add, ids=chunk_ids)
            document_chunk_mapping[doc_url] = chunk_ids

        return document_chunk_mapping

    def get_document_chunks(self, collection_name: str, document_id: str) -> List[Document]:
        collection = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

        qdrant_filter = Filter(
            should=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        )
        return collection.similarity_search("", k=1000, filter=qdrant_filter)

    def delete_documents(self, collection_name: str, document_ids: List[str]) -> bool:
        collection = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )
        collection.delete(ids=document_ids)
        return True

    def bulk_url_upload(self, collection_name: str, urls: List[str]) -> Dict[str, Any]:
        collection = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

        results = {
            "total_urls": len(urls),
            "successful_uploads": 0,
            "failed_uploads": 0,
            "document_mappings": {},
            "errors": []
        }

        for i, url in enumerate(urls):
            try:
                print(f"Processing URL {i+1}/{len(urls)}: {url}")
                
   
                converted = DocumentConverter()
                doc = converted.convert(url).document
                markdown_content = doc.export_to_markdown()
                chunks = self.text_splitter.split_text(markdown_content)

                document_id = str(uuid4())
                base_metadata = {
                    "source": url,
                    "filename": os.path.basename(url),
                    "document_id": document_id,
                    "created_at": datetime.now().isoformat(),
                }

                documents_to_add = []
                chunk_ids = []

                for j, chunk in enumerate(chunks):
                    chunk_uuid = str(uuid4())
                    chunk_id = f"{document_id}_chunk_{j}"
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": j,
                        "total_chunks": len(chunks),
                        "chunk_size": len(chunk),
                        "chunk_id": chunk_id,
                    })

                    chunk_ids.append(chunk_uuid)
                    documents_to_add.append(Document(page_content=chunk, metadata=chunk_metadata))

           
                collection.add_documents(documents=documents_to_add, ids=chunk_ids)
                results["document_mappings"][url] = chunk_ids
                results["successful_uploads"] += 1
                print(f"✓ Successfully processed: {url} ({len(chunks)} chunks)")

            except Exception as e:
                error_msg = f"Failed to process {url}: {str(e)}"
                results["errors"].append(error_msg)
                results["failed_uploads"] += 1
                print(f"✗ Error processing {url}: {str(e)}")

        print(f"\nBulk upload completed:")
        print(f"  Total URLs: {results['total_urls']}")
        print(f"  Successful: {results['successful_uploads']}")
        print(f"  Failed: {results['failed_uploads']}")
        
        return results

    def search(self, collection_name: str, query: str, top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        collection = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )
        
        qdrant_filter = None
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            qdrant_filter = Filter(should=conditions)
        
        results = collection.similarity_search_with_score(
            query=query, 
            k=top_k, 
            filter=qdrant_filter
        )
        
        return results

    def search_by_document(self, collection_name: str, query: str, document_id: str, top_k: int = 5) -> List[Document]:
        return self.search(
            collection_name=collection_name,
            query=query,
            top_k=top_k,
            filter_metadata={"document_id": document_id}
        )