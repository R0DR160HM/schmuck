#!/usr/bin/env python3
"""
Hunsrik Language RAG System for Gemma 3
Extracts text from PDFs, creates a vector store, and enables RAG queries
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import PyPDF2
import ollama
from datetime import datetime

# ---------- CONFIG ----------
EMBED_MODEL = "embeddinggemma"
GEN_MODEL = "gemma3n:e2b"
PDF_DIR = "pdfs"
VECTOR_STORE_FILE = "hunsrik_vectors.json"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks
TOP_K = 5  # number of relevant chunks to retrieve
# ----------------------------


class HunsrikRAG:
    def __init__(self):
        self.vector_store = []
        self.load_vector_store()
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        print(f"📄 Extracting text from: {os.path.basename(pdf_path)}")
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                print(f"   ✓ Extracted {len(pdf_reader.pages)} pages")
        except Exception as e:
            print(f"   ✗ Error: {e}")
        return text
    
    def chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split text into overlapping chunks"""
        chunks = []
        text = text.strip()
        start = 0
        
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > CHUNK_SIZE * 0.5:  # Only if not too short
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            if chunk.strip():
                chunks.append({
                    'text': chunk.strip(),
                    'source': source,
                    'start_pos': start
                })
            
            start = end - CHUNK_OVERLAP
        
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text using Ollama"""
        try:
            response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
            return response['embedding']
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot_product / (mag1 * mag2)
    
    def process_pdfs(self):
        """Process all PDFs in the PDF directory"""
        pdf_dir = Path(PDF_DIR)
        if not pdf_dir.exists():
            print(f"❌ Directory '{PDF_DIR}' not found!")
            return
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in '{PDF_DIR}'")
            return
        
        print(f"\n🚀 Processing {len(pdf_files)} PDF file(s)...\n")
        
        all_chunks = []
        for pdf_file in pdf_files:
            text = self.extract_text_from_pdf(str(pdf_file))
            if text:
                chunks = self.chunk_text(text, pdf_file.name)
                all_chunks.extend(chunks)
                print(f"   → Created {len(chunks)} chunks\n")
        
        print(f"📊 Total chunks: {len(all_chunks)}")
        print(f"🔮 Generating embeddings with {EMBED_MODEL}...\n")
        
        self.vector_store = []
        for i, chunk in enumerate(all_chunks):
            embedding = self.get_embedding(chunk['text'])
            if embedding:
                self.vector_store.append({
                    'text': chunk['text'],
                    'source': chunk['source'],
                    'embedding': embedding
                })
                if (i + 1) % 10 == 0:
                    print(f"   Progress: {i + 1}/{len(all_chunks)}")
        
        print(f"\n✅ Created {len(self.vector_store)} embeddings")
        self.save_vector_store()
    
    def save_vector_store(self):
        """Save vector store to disk"""
        print(f"💾 Saving vector store to {VECTOR_STORE_FILE}...")
        with open(VECTOR_STORE_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.vector_store, f, ensure_ascii=False)
        print("   ✓ Saved successfully")
    
    def load_vector_store(self):
        """Load vector store from disk"""
        if os.path.exists(VECTOR_STORE_FILE):
            print(f"📂 Loading existing vector store...")
            with open(VECTOR_STORE_FILE, 'r', encoding='utf-8') as f:
                self.vector_store = json.load(f)
            print(f"   ✓ Loaded {len(self.vector_store)} chunks")
        else:
            print("ℹ️  No existing vector store found")
    
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """Retrieve most relevant chunks for a query"""
        if not self.vector_store:
            print("⚠️  Vector store is empty. Run process_pdfs() first.")
            return []
        
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        # Calculate similarities
        similarities = []
        for item in self.vector_store:
            sim = self.cosine_similarity(query_embedding, item['embedding'])
            similarities.append((sim, item))
        
        # Sort by similarity and get top_k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [{'text': item['text'], 'source': item['source'], 'similarity': sim} 
                for sim, item in similarities[:top_k]]
    
    def query(self, question: str, verbose: bool = True) -> str:
        """Query the RAG system"""
        if verbose:
            print(f"\n💬 Question: {question}\n")
            print("🔍 Retrieving relevant context...")
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve(question)
        
        if not relevant_chunks:
            return "No relevant information found. Please process PDFs first."
        
        if verbose:
            print(f"   ✓ Found {len(relevant_chunks)} relevant chunks\n")
            for i, chunk in enumerate(relevant_chunks[:3], 1):
                print(f"   [{i}] Similarity: {chunk['similarity']:.3f} | Source: {chunk['source']}")
        
        # Build context
        context = "\n\n---\n\n".join([
            f"[Source: {chunk['source']}]\n{chunk['text']}" 
            for chunk in relevant_chunks
        ])
        
        # Build prompt
        prompt = f"""Você é um especialista na língua hunsriqueana (Hunsrickisch). Use o contexto a seguir (materiais de aprendizado de hunsriqueano) para responder à pergunta.

Contexto dos materiais de hunsriqueano:
{context}

Pergunta: {question}

Responda com base no contexto acima. Responda sempre em português e, em seguida, traduza para hunsriqueano, seguindo as regras gramáticais e exemplos obtidos no contexto.
Seja breve e direto."""
        
        if verbose:
            print(f"\n🤖 Generating answer with {GEN_MODEL}...\n")
        
        # Generate response
        try:
            response = ollama.generate(
                model=GEN_MODEL,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                }
            )
            return response['response']
        except Exception as e:
            return f"Error generating response: {e}"
    
    def interactive_mode(self):
        """Start interactive Q&A session"""
        print("\n" + "="*60)
        print("🗣️  HUNSRIK RAG SYSTEM - Interactive Mode")
        print("="*60)
        print("\nCommands:")
        print("  - Type your question to get an answer")
        print("  - 'quit' or 'exit' to stop")
        print("  - 'reprocess' to reload PDFs")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'reprocess':
                    self.process_pdfs()
                    continue
                
                answer = self.query(user_input)
                print(f"\n🤖 Answer:\n{answer}\n")
                print("-" * 60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def main():
    """Main function"""
    print("\n" + "="*60)
    print("🇧🇷 HUNSRIK LANGUAGE RAG SYSTEM")
    print("="*60)
    
    rag = HunsrikRAG()
    
    # Check if vector store exists
    if not rag.vector_store:
        print("\n📋 First time setup: Processing PDFs...")
        response = input("Process PDFs now? (y/n): ").strip().lower()
        if response == 'y':
            rag.process_pdfs()
        else:
            print("⚠️  Skipping PDF processing. Run with 'reprocess' command later.")
    
    # Start interactive mode
    rag.interactive_mode()


if __name__ == "__main__":
    main()
