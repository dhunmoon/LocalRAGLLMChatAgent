import os
import json
import hashlib
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime
import argparse

# Core dependencies you'll need to install:
# pip install sentence-transformers chromadb langchain-community python-docx PyPDF2 ollama

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import ollama

# Document processing
import PyPDF2
from docx import Document as DocxDocument
import docx


class DocumentProcessor:
    """Handles extraction of text from various document formats"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF files"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX files"""
        try:
            doc = DocxDocument(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error processing DOCX {file_path}: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from TXT files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error processing TXT {file_path}: {e}")
            return ""
    
    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """Extract text based on file extension"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return cls.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return cls.extract_text_from_docx(file_path)
        elif file_extension in ['.txt', '.md']:
            return cls.extract_text_from_txt(file_path)
        else:
            print(f"Unsupported file type: {file_extension}")
            return ""


class DocumentChunker:
    """Splits documents into chunks for better vector search"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, file_path: str) -> List[Dict]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > len(chunk_text) * 0.7:  # Only if break point is reasonable
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + len(chunk_text)
            
            chunks.append({
                'text': chunk_text.strip(),
                'file_path': file_path,
                'chunk_id': chunk_id,
                'start_pos': start,
                'end_pos': end
            })
            
            start = end - self.overlap
            chunk_id += 1
        
        return chunks


class LocalRAGAgent:
    """Main RAG agent class that handles document indexing and querying"""
    
    def __init__(self, db_path: str = "./rag_database", model_name: str = "llama3.2"):
        self.db_path = db_path
        self.model_name = model_name
        
        # Initialize components
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.document_processor = DocumentProcessor()
        self.chunker = DocumentChunker()
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize SQLite for metadata
        self.init_metadata_db()
        
        print(f"RAG Agent initialized with model: {model_name}")
        print(f"Database path: {db_path}")
    
    def init_metadata_db(self):
        """Initialize SQLite database for file metadata"""
        os.makedirs(self.db_path, exist_ok=True)
        self.conn = sqlite3.connect(f"{self.db_path}/metadata.db")
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                file_hash TEXT,
                indexed_at TIMESTAMP,
                chunk_count INTEGER
            )
        """)
        self.conn.commit()
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash of file content for change detection"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def is_file_indexed(self, file_path: str) -> bool:
        """Check if file is already indexed and unchanged"""
        cursor = self.conn.execute(
            "SELECT file_hash FROM documents WHERE file_path = ?",
            (file_path,)
        )
        result = cursor.fetchone()
        
        if result:
            stored_hash = result[0]
            current_hash = self.get_file_hash(file_path)
            return stored_hash == current_hash
        
        return False
    
    def index_document(self, file_path: str) -> bool:
        """Index a single document"""
        if self.is_file_indexed(file_path):
            print(f"File already indexed and unchanged: {file_path}")
            return True
        
        print(f"Indexing: {file_path}")
        
        # Extract text
        text = self.document_processor.extract_text(file_path)
        if not text:
            print(f"No text extracted from: {file_path}")
            return False
        
        # Create chunks
        chunks = self.chunker.chunk_text(text, file_path)
        if not chunks:
            print(f"No chunks created for: {file_path}")
            return False
        
        # Generate embeddings
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts).tolist()
        
        # Prepare data for ChromaDB
        ids = [f"{file_path}_{chunk['chunk_id']}" for chunk in chunks]
        metadatas = [
            {
                'file_path': chunk['file_path'],
                'chunk_id': chunk['chunk_id'],
                'start_pos': chunk['start_pos'],
                'end_pos': chunk['end_pos']
            }
            for chunk in chunks
        ]
        
        # Remove existing entries for this file
        try:
            existing_ids = self.collection.get(where={"file_path": file_path})['ids']
            if existing_ids:
                self.collection.delete(ids=existing_ids)
        except:
            pass  # Collection might be empty
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # Update metadata database
        file_hash = self.get_file_hash(file_path)
        self.conn.execute("""
            INSERT OR REPLACE INTO documents 
            (file_path, file_hash, indexed_at, chunk_count)
            VALUES (?, ?, ?, ?)
        """, (file_path, file_hash, datetime.now(), len(chunks)))
        self.conn.commit()
        
        print(f"Indexed {len(chunks)} chunks from: {file_path}")
        return True
    
    def index_folder(self, folder_path: str, recursive: bool = True):
        """Index all supported documents in a folder"""
        supported_extensions = {'.pdf', '.docx', '.txt', '.md'}
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"Folder does not exist: {folder_path}")
            return
        
        print(f"Indexing folder: {folder_path}")
        
        # Find all supported files
        files_to_index = []
        if recursive:
            for ext in supported_extensions:
                files_to_index.extend(folder_path.rglob(f"*{ext}"))
        else:
            for ext in supported_extensions:
                files_to_index.extend(folder_path.glob(f"*{ext}"))
        
        print(f"Found {len(files_to_index)} files to process")
        
        # Index each file
        successful = 0
        for file_path in files_to_index:
            if self.index_document(str(file_path)):
                successful += 1
        
        print(f"Successfully indexed {successful}/{len(files_to_index)} files")
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant document chunks"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format results
        search_results = []
        for i in range(len(results['ids'][0])):
            search_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return search_results
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using local LLM with context"""
        # Prepare context
        context = "\n\n".join([
            f"Document: {chunk['metadata']['file_path']}\n{chunk['text']}"
            for chunk in context_chunks
        ])
        
        # Create prompt
        prompt = f"""Based on the following documents, please answer the question. If the answer is not in the documents, say so clearly.

Context Documents:
{context}

Question: {query}

Answer:"""
        
        try:
            # Generate response using Ollama
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'max_tokens': 1000
                }
            )
            return response['response']
        except Exception as e:
            return f"Error generating response: {e}"
    
    def chat(self, query: str, n_results: int = 3) -> Dict:
        """Main chat interface"""
        print(f"\nProcessing query: {query}")
        
        # Search for relevant documents
        search_results = self.search_documents(query, n_results)
        
        if not search_results:
            return {
                'query': query,
                'answer': "I couldn't find any relevant documents to answer your question.",
                'sources': []
            }
        
        # Generate answer
        answer = self.generate_answer(query, search_results)
        
        # Prepare sources
        sources = []
        for result in search_results:
            sources.append({
                'file_path': result['metadata']['file_path'],
                'relevance_score': 1 - result['distance'] if result['distance'] else None,
                'text_preview': result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
            })
        
        return {
            'query': query,
            'answer': answer,
            'sources': sources
        }
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        cursor = self.conn.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        
        collection_count = self.collection.count()
        
        return {
            'indexed_documents': doc_count,
            'total_chunks': collection_count,
            'embedding_model': 'all-MiniLM-L6-v2',
            'llm_model': self.model_name
        }


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Local Document Chat Agent')
    parser.add_argument('--model', default='llama3.2', help='Ollama model name')
    parser.add_argument('--db-path', default='./rag_database', help='Database path')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index documents')
    index_parser.add_argument('path', help='File or folder path to index')
    index_parser.add_argument('--recursive', action='store_true', help='Index recursively')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Start interactive chat')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database stats')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = LocalRAGAgent(db_path=args.db_path, model_name=args.model)
    
    if args.command == 'index':
        if os.path.isfile(args.path):
            agent.index_document(args.path)
        elif os.path.isdir(args.path):
            agent.index_folder(args.path, recursive=args.recursive)
        else:
            print(f"Path not found: {args.path}")
    
    elif args.command == 'chat':
        print("Local Document Chat Agent")
        print("Type 'quit' to exit, 'stats' for database info")
        print("-" * 50)
        
        while True:
            query = input("\nYour question: ").strip()
            
            if query.lower() in ['quit', 'exit']:
                break
            
            if query.lower() == 'stats':
                stats = agent.get_stats()
                print(f"\nDatabase Stats:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            if not query:
                continue
            
            result = agent.chat(query)
            
            print(f"\nAnswer: {result['answer']}")
            
            if result['sources']:
                print(f"\nSources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source['file_path']}")
                    if source['relevance_score']:
                        print(f"   Relevance: {source['relevance_score']:.3f}")
    
    elif args.command == 'stats':
        stats = agent.get_stats()
        print("Database Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()