#!/usr/bin/env python3
"""
Hunsrik Language RAG System for Gemma 3
Extracts text from PDFs and TXT files, creates a vector store, and enables RAG queries
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
RESOURCES_DIR = "resources"  # Base directory
VECTOR_STORE_FILE = "hunsrik_vectors.json"
CHUNK_SIZE = 300  # characters per chunk (smaller for dictionary entries)
CHUNK_OVERLAP = 100  # overlap between chunks (more overlap)
TOP_K_DICT = 15  # chunks from dictionary/grammar
TOP_K_SAMPLES = 5  # chunks from Hunsrik samples
# ----------------------------

# Resource types
RESOURCE_DICT = "dictionary"
RESOURCE_GRAMMAR = "grammar"
RESOURCE_SAMPLE = "sample"


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
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from a TXT file"""
        print(f"📝 Reading text from: {os.path.basename(txt_path)}")
        text = ""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.read()
                line_count = text.count('\n') + 1
                print(f"   ✓ Read {line_count} lines")
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(txt_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                    line_count = text.count('\n') + 1
                    print(f"   ✓ Read {line_count} lines (latin-1 encoding)")
            except Exception as e:
                print(f"   ✗ Error: {e}")
        except Exception as e:
            print(f"   ✗ Error: {e}")
        return text
    
    def chunk_text(self, text: str, source: str, resource_type: str = RESOURCE_DICT) -> List[Dict]:
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
                    'resource_type': resource_type,
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
    
    def process_files(self):
        """Process all PDF and TXT files from multiple resource folders"""
        resources_dir = Path(RESOURCES_DIR)
        if not resources_dir.exists():
            print(f"❌ Directory '{RESOURCES_DIR}' not found!")
            return
        
        # Define folder mappings
        folder_types = {
            'dicts': RESOURCE_DICT,
            'grammar': RESOURCE_GRAMMAR,
            'samples': RESOURCE_SAMPLE,
        }
        
        all_chunks = []
        total_files = 0
        
        print(f"\n🚀 Processing resources from multiple folders...\n")
        
        # Process each folder type
        for folder_name, resource_type in folder_types.items():
            folder_path = resources_dir / folder_name
            
            if not folder_path.exists():
                print(f"⚠️  Folder '{folder_name}' not found, skipping...")
                continue
            
            pdf_files = list(folder_path.glob("*.pdf"))
            txt_files = list(folder_path.glob("*.txt"))
            folder_files = pdf_files + txt_files
            
            if not folder_files:
                print(f"📁 {folder_name}/: No files found")
                continue
            
            print(f"📁 {folder_name}/: Processing {len(folder_files)} file(s) ({resource_type})")
            folder_chunks = 0
            
            for file_path in folder_files:
                if file_path.suffix.lower() == '.pdf':
                    text = self.extract_text_from_pdf(str(file_path))
                elif file_path.suffix.lower() == '.txt':
                    text = self.extract_text_from_txt(str(file_path))
                else:
                    continue
                
                if text:
                    chunks = self.chunk_text(text, file_path.name, resource_type)
                    all_chunks.extend(chunks)
                    folder_chunks += len(chunks)
                    total_files += 1
            
            print(f"   → Created {folder_chunks} chunks from {folder_name}\n")
        
        if not all_chunks:
            print("❌ No files processed!")
            return
        
        print(f"📊 Total chunks: {len(all_chunks)}")
        print(f"🔮 Generating embeddings with {EMBED_MODEL}...\n")
        
        self.vector_store = []
        for i, chunk in enumerate(all_chunks):
            embedding = self.get_embedding(chunk['text'])
            if embedding:
                self.vector_store.append({
                    'text': chunk['text'],
                    'source': chunk['source'],
                    'resource_type': chunk['resource_type'],
                    'embedding': embedding
                })
                if (i + 1) % 20 == 0:
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
    
    def retrieve(self, query: str, top_k: int = 15, resource_types: List[str] = None) -> List[Dict]:
        """Retrieve most relevant chunks for a query, optionally filtered by resource type"""
        if not self.vector_store:
            print("⚠️  Vector store is empty. Run process_files() first.")
            return []
        
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        # Filter by resource type if specified
        items_to_search = self.vector_store
        if resource_types:
            items_to_search = [item for item in self.vector_store 
                             if item.get('resource_type') in resource_types]
        
        # Calculate similarities
        similarities = []
        for item in items_to_search:
            sim = self.cosine_similarity(query_embedding, item['embedding'])
            similarities.append((sim, item))
        
        # Sort by similarity and get top_k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [{'text': item['text'], 'source': item['source'], 
                'resource_type': item.get('resource_type', 'unknown'), 'similarity': sim} 
                for sim, item in similarities[:top_k]]
    
    def hybrid_retrieve(self, text: str) -> Dict[str, List[Dict]]:
        """Intelligent hybrid retrieval: dict/grammar first, then samples for context"""
        # Extract individual words for dictionary lookup
        words = text.lower().replace('?', '').replace('!', '').replace('.', '').replace(',', '').split()
        
        # Create query variations
        queries = [text, f"tradução português hunsrik: {text}"]
        for word in words:
            if len(word) > 2:
                queries.append(word)
        
        # STEP 1: Search dictionary and grammar
        dict_results = {}
        for query in queries:
            results = self.retrieve(query, top_k=TOP_K_DICT, 
                                  resource_types=[RESOURCE_DICT, RESOURCE_GRAMMAR])
            for result in results:
                text_key = result['text']
                if text_key in dict_results:
                    dict_results[text_key]['similarity'] += result['similarity'] * 0.3
                else:
                    dict_results[text_key] = result
        
        dict_sorted = sorted(dict_results.values(), key=lambda x: x['similarity'], reverse=True)[:TOP_K_DICT]
        
        # STEP 2: Extract potential Hunsrik words from dictionary results
        # Look for patterns like "word (HRX)" or lines with Hunsrik text
        hunsrik_terms = set()
        for result in dict_sorted[:5]:  # Use top 5 dictionary results
            # Simple extraction: get words that look like Hunsrik
            for word in result['text'].split():
                word_clean = word.strip('.,;:()[]"').lower()
                # Hunsrik often has: double vowels (aa, ee, oo), specific patterns
                if len(word_clean) > 3 and any(c in word_clean for c in ['aa', 'ee', 'oo', 'ä', 'ö', 'ü']):
                    hunsrik_terms.add(word_clean)
        
        # STEP 3: Search samples using Hunsrik terms found in dictionary
        sample_results = {}
        if hunsrik_terms:
            for term in list(hunsrik_terms)[:5]:  # Limit to avoid too many queries
                results = self.retrieve(term, top_k=TOP_K_SAMPLES, 
                                      resource_types=[RESOURCE_SAMPLE])
                for result in results:
                    text_key = result['text']
                    if text_key not in sample_results:
                        sample_results[text_key] = result
        
        sample_sorted = sorted(sample_results.values(), key=lambda x: x['similarity'], reverse=True)[:TOP_K_SAMPLES]
        
        return {
            'dictionary': dict_sorted,
            'samples': sample_sorted
        }
    
    def query(self, question: str, verbose: bool = True) -> str:
        """Query with hybrid retrieval: dictionary + samples for context"""
        if verbose:
            print(f"\n💬 Texto para traduzir: {question}\n")
            print("🔍 Fase 1: Buscando no dicionário e gramática...")
        
        # Use hybrid retrieval
        results = self.hybrid_retrieve(question)
        dict_chunks = results['dictionary']
        sample_chunks = results['samples']
        
        if not dict_chunks:
            return "Nenhuma informação relevante encontrada. Execute 'reprocess' primeiro."
        
        if verbose:
            print(f"   ✓ {len(dict_chunks)} entradas do dicionário/gramática")
            for i, chunk in enumerate(dict_chunks[:3], 1):
                preview = chunk['text'][:60].replace('\n', ' ')
                print(f"   [{i}] Score: {chunk['similarity']:.3f} | {preview}...")
            
            if sample_chunks:
                print(f"\n🔍 Fase 2: Buscando exemplos em textos Hunsrik...")
                print(f"   ✓ {len(sample_chunks)} exemplos de uso encontrados")
                for i, chunk in enumerate(sample_chunks[:2], 1):
                    preview = chunk['text'][:60].replace('\n', ' ')
                    print(f"   [{i}] {preview}...")
        
        # Build contexts separately
        dict_context = "\n\n".join([chunk['text'] for chunk in dict_chunks])
        
        sample_context = ""
        if sample_chunks:
            sample_context = "\n\n=== EXEMPLOS DE USO EM CONTEXTO (textos Hunsrik) ===\n"
            sample_context += "\n".join([chunk['text'][:200] for chunk in sample_chunks[:3]])
            sample_context += "\n=== FIM DOS EXEMPLOS ==="
        
        # Build prompt
        prompt = f"""Você é um tradutor especializado em Hunsrik (Hunsrückisch). Use APENAS as informações do dicionário fornecido abaixo. NÃO invente palavras.

=== DICIONÁRIO E GRAMÁTICA HUNSRIK ===
{dict_context}
=== FIM DO DICIONÁRIO ===

{sample_context}

=== EXEMPLOS DE TRADUÇÕES ===
Português: "Meu nome é Maria"
Hunsrik: "Mein Naame is Maria"

Português: "Eu tenho um cachorro"
Hunsrik: "Ich hann en Hund"

Português: "Bom dia"
Hunsrik: "Gude Dag"
=== FIM DOS EXEMPLOS ===

INSTRUÇÕES:
1. Procure cada palavra no dicionário acima
2. Use a ortografia EXATA do dicionário
3. Se não encontrar uma palavra, mantenha-a em português entre parênteses
4. Responda APENAS com a tradução, sem explicações

Português: "{question}"
Hunsrik:"""
        
        if verbose:
            print(f"\n🤖 Gerando tradução com {GEN_MODEL}...\n")
            print(f"\n🤖 Gerando tradução com {GEN_MODEL}...\n")
        
        # Generate response with lower temperature for more accurate translations
        # Generate response with lower temperature for more accurate translations
        try:
            response = ollama.generate(
                model=GEN_MODEL,
                prompt=prompt,
                options={
                    'temperature': 0.15,
                    'top_p': 0.85,
                    'top_k': 40,
                    'repeat_penalty': 1.2,
                }
            )
            return response['response'].strip()
        except Exception as e:
            return f"Erro ao gerar resposta: {e}"
    
    def interactive_mode(self):
        """Start interactive Q&A session"""
        print("\n" + "="*60)
        print("🗣️  HUNSRIK RAG SYSTEM - Interactive Mode")
        print("="*60)
        print("\nCommands:")
        print("  - Type your question to get an answer")
        print("  - 'quit' or 'exit' to stop")
        print("  - 'reprocess' to reload all files (PDFs and TXTs)")
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
                    self.process_files()
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
        print("\n📋 First time setup: Processing files...")
        response = input("Process files now? (y/n): ").strip().lower()
        if response == 'y':
            rag.process_files()
        else:
            print("⚠️  Skipping file processing. Run with 'reprocess' command later.")
    
    # Start interactive mode
    rag.interactive_mode()


if __name__ == "__main__":
    main()
