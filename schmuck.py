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
LONG_TEXT_THRESHOLD = 100  # characters - threshold for "long" text
MAX_WORDS_TO_SEARCH = 20  # maximum words to search individually
# ----------------------------

# Portuguese stopwords (common words to skip)
STOPWORDS = {
    'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas',
    'de', 'do', 'da', 'dos', 'das', 'em', 'no', 'na', 'nos', 'nas',
    'por', 'para', 'com', 'sem', 'sob', 'sobre',
    'e', 'ou', 'mas', 'que', 'se', 'quando', 'onde',
    'muito', 'muita', 'muitos', 'muitas', 'mais', 'menos',
    'é', 'está', 'estava', 'foi', 'ser', 'estar', 'ter', 'havia',
    # Dictionary technical abbreviations (metadata, not translations)
    #'pop', 'sf', 'sm', 'pl', 'sg', 'biol', 'anat', 'adj', 'geog',
    #'bot', 'zool', 'med', 'quím', 'fís', 'mat', 'gram', 'ling',
    ##'fig', 'lit', 'poet', 'arc', 'neol', 'gír', 'vulg', 'fam',
    #'form', 'colloq', 'v', 'vt', 'vi', 'adv', 'prep', 'conj',
    #'interj', 'pron', 'num', 'art', 'loc', 'expr', 'sin', 'ant'
}

# Resource types
RESOURCE_DICT = "dictionary"
RESOURCE_GRAMMAR = "grammar"
RESOURCE_SAMPLE = "sample"


class HunsrikRAG:
    def __init__(self):
        self.vector_store = []
        self.load_vector_store()
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from PDF or TXT file"""
        path = Path(file_path)
        try:
            if path.suffix.lower() == '.pdf':
                with open(file_path, 'rb') as f:
                    return '\n'.join(page.extract_text() or '' for page in PyPDF2.PdfReader(f).pages)
            else:  # TXT
                for encoding in ['utf-8', 'latin-1']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            return f.read()
                    except UnicodeDecodeError:
                        continue
        except Exception as e:
            print(f"⚠️ {path.name}: {e}")
        return ""
    
    def chunk_text(self, text: str, source: str, resource_type: str = RESOURCE_DICT) -> List[Dict]:
        """Split text into overlapping chunks"""
        chunks, start = [], 0
        text = text.strip()
        
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end]
            
            # Break at sentence boundary if possible
            if end < len(text):
                break_point = max(chunk.rfind('.'), chunk.rfind('\n'))
                if break_point > CHUNK_SIZE * 0.5:
                    end = start + break_point + 1
                    chunk = chunk[:break_point + 1]
            
            if chunk.strip():
                chunks.append({'text': chunk.strip(), 'source': source, 'resource_type': resource_type})
            start = end - CHUNK_OVERLAP
        
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text using Ollama"""
        try:
            return ollama.embeddings(model=EMBED_MODEL, prompt=text)['embedding']
        except Exception:
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag = (sum(a * a for a in vec1) * sum(b * b for b in vec2)) ** 0.5
        return dot / mag if mag else 0.0
    
    def clean_dict_text(self, text: str) -> str:
        """Remove metadata abbreviations from dictionary text"""
        # Remove common patterns like (sf), (adj), [Biol], etc.
        import re
        cleaned = text
        # Remove parentheses with abbreviations
        cleaned = re.sub(r'\s*\([^)]*(?:' + '|'.join(['sf', 'sm', 'pl', 'adj', 'adv', 'v', 'vt', 'vi']) + r')[^)]*\)', '', cleaned)
        # Remove brackets with abbreviations
        cleaned = re.sub(r'\s*\[[^\]]*(?:' + '|'.join(['Biol', 'Anat', 'Geog', 'Pop', 'Bot', 'Zool', 'Med']) + r')[^\]]*\]', '', cleaned)
        # Remove standalone abbreviations at start of lines or after punctuation
        for abbr in ['Pop', 'Biol', 'Anat', 'Geog', 'Bot', 'Zool', 'Med', 'sf', 'sm', 'pl', 'adj']:
            cleaned = cleaned.replace(abbr, '')
        return ' '.join(cleaned.split())  # Normalize whitespace
    
    def process_files(self):
        """Process all PDF and TXT files from resource folders"""
        resources_dir = Path(RESOURCES_DIR)
        if not resources_dir.exists():
            return print(f"❌ '{RESOURCES_DIR}' not found!")
        
        folder_types = {'dicts': RESOURCE_DICT, 'grammar': RESOURCE_GRAMMAR, 'samples': RESOURCE_SAMPLE}
        all_chunks = []
        
        print("\n🚀 Processing files...")
        for folder_name, resource_type in folder_types.items():
            folder_path = resources_dir / folder_name
            if not folder_path.exists():
                continue
            
            for file_path in list(folder_path.glob("*.pdf")) + list(folder_path.glob("*.txt")):
                text = self.extract_text(str(file_path))
                if text:
                    all_chunks.extend(self.chunk_text(text, file_path.name, resource_type))
        
        if not all_chunks:
            return print("❌ No files processed!")
        
        print(f"📊 Creating {len(all_chunks)} embeddings...")
        self.vector_store = []
        for i, chunk in enumerate(all_chunks):
            if emb := self.get_embedding(chunk['text']):
                self.vector_store.append({**chunk, 'embedding': emb})
                if (i + 1) % 50 == 0:
                    print(f"   {i + 1}/{len(all_chunks)}")
        
        print(f"✅ Created {len(self.vector_store)} embeddings")
        self.save_vector_store()
    
    def save_vector_store(self):
        """Save vector store to disk"""
        with open(VECTOR_STORE_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.vector_store, f, ensure_ascii=False)
        print(f"💾 Saved {len(self.vector_store)} vectors")
    
    def load_vector_store(self):
        """Load vector store from disk"""
        if os.path.exists(VECTOR_STORE_FILE):
            with open(VECTOR_STORE_FILE, 'r', encoding='utf-8') as f:
                self.vector_store = json.load(f)
            print(f"📂 Loaded {len(self.vector_store)} vectors")
    
    def retrieve(self, query: str, top_k: int = 15, resource_types: List[str] = None) -> List[Dict]:
        """Retrieve most relevant chunks for a query, optionally filtered by resource type"""
        if not self.vector_store or not (query_emb := self.get_embedding(query)):
            return []
        
        items = [i for i in self.vector_store if not resource_types or i.get('resource_type') in resource_types]
        sims = sorted([(self.cosine_similarity(query_emb, i['embedding']), i) for i in items], 
                     reverse=True, key=lambda x: x[0])[:top_k]
        
        return [{'text': i['text'], 'source': i['source'], 
                'resource_type': i.get('resource_type', 'unknown'), 'similarity': s} for s, i in sims]
    
    def hybrid_retrieve(self, text: str) -> Dict[str, List[Dict]]:
        """Intelligent hybrid retrieval: dict/grammar first, then samples for context"""
        is_long = len(text) > LONG_TEXT_THRESHOLD
        words = text.lower().replace('?', '').replace('!', '').replace('.', '').replace(',', '').split()
        meaningful = sorted([w for w in words if len(w) > 2 and w not in STOPWORDS], 
                          key=len, reverse=True)[:MAX_WORDS_TO_SEARCH]
        
        # Build queries
        queries = [text]
        if is_long:
            for delim in ['. ', '? ', '! ', ', ']:
                if delim in text:
                    queries.extend([s.strip() for s in text.split(delim) if len(s.strip()) > 10][:3])
                    break
        queries.extend(meaningful[:15])
        if len(meaningful) >= 2:
            queries.extend([f"{meaningful[i]} {meaningful[i+1]}" for i in range(min(5, len(meaningful)-1))])
        
        # Dynamic TOP_K
        k_dict = int(TOP_K_DICT * 1.5) if is_long else TOP_K_DICT
        k_samples = int(TOP_K_SAMPLES * 1.5) if is_long else TOP_K_SAMPLES
        
        # Search dictionary/grammar
        dict_results = {}
        for q in queries:
            for r in self.retrieve(q, top_k=k_dict, resource_types=[RESOURCE_DICT, RESOURCE_GRAMMAR]):
                if r['text'] in dict_results:
                    dict_results[r['text']]['similarity'] += r['similarity'] * 0.3
                else:
                    dict_results[r['text']] = r
        dict_sorted = sorted(dict_results.values(), key=lambda x: x['similarity'], reverse=True)[:k_dict]
        
        # Extract Hunsrik terms
        hunsrik_terms = set(meaningful)
        for result in dict_sorted[:12 if is_long else 8]:
            for word in result['text'].split():
                wc = word.strip('.,;:()[]"').lower()
                if len(wc) > 2 and (any(p in wc for p in ['aa','ee','oo','au','ei','eu']) or
                                   any(c in wc for c in ['ä','ö','ü','ß']) or
                                   wc.endswith(('je','che','sch','ich','lich'))):
                    hunsrik_terms.add(wc)
                    if word and word[0].isupper():
                        hunsrik_terms.add(word.strip('.,;:()[]"'))
        
        # Search samples
        sample_results = {}
        for term in list(hunsrik_terms)[:15 if is_long else 10]:
            for r in self.retrieve(term, top_k=k_samples, resource_types=[RESOURCE_SAMPLE]):
                if r['text'] not in sample_results:
                    sample_results[r['text']] = r
                else:
                    sample_results[r['text']]['similarity'] += r['similarity'] * 0.2
        
        return {
            'dictionary': dict_sorted,
            'samples': sorted(sample_results.values(), key=lambda x: x['similarity'], reverse=True)[:k_samples]
        }
    
    def query(self, question: str, verbose: bool = True) -> str:
        """Query with hybrid retrieval: dictionary + samples for context"""
        if verbose:
            print(f"\n� Buscando: {question}")
        
        results = self.hybrid_retrieve(question)
        if not (dict_chunks := results['dictionary']):
            return "Nenhuma informação encontrada. Execute 'reprocess'."
        
        sample_chunks = results['samples']
        if verbose:
            print(f"   ✓ {len(dict_chunks)} entradas | {len(sample_chunks)} exemplos\n")
            print("📚 ENTRADAS DO DICIONÁRIO:")
            for i, c in enumerate(dict_chunks[:10], 1):
                preview = c['text'][:150].replace('\n', ' ')
                print(f"   [{i}] Score: {c['similarity']:.3f} | {c['source']}")
                print(f"       {preview}...\n")
            
            if sample_chunks:
                print("📖 EXEMPLOS DE USO:")
                for i, c in enumerate(sample_chunks[:5], 1):
                    preview = c['text'][:150].replace('\n', ' ')
                    print(f"   [{i}] Score: {c['similarity']:.3f} | {c['source']}")
                    print(f"       {preview}...\n")
        
        # Build contexts - clean dictionary text to remove metadata
        dict_ctx = "\n\n".join([self.clean_dict_text(c['text']) for c in dict_chunks])
        sample_ctx = ""
        if sample_chunks:
            sample_ctx = f"\n\n=== EXEMPLOS DE USO ===\n" + \
                        "\n".join([c['text'][:200] for c in sample_chunks[:3]]) + \
                        "\n=== FIM DOS EXEMPLOS ==="
        
        prompt = f"""Você é um especialista em hunsriqueano (Hunsrickisch).

INSTRUÇÕES IMPORTANTES:
1. Use APENAS palavras que você encontrar no DICIONÁRIO abaixo
2. Se não encontrar a tradução exata, use palavras similares do dicionário
3. Mantenha a estrutura gramatical alemã (verbo em segunda posição)
4. NÃO invente palavras que não estão no dicionário
5. Use os EXEMPLOS para entender o contexto de uso

=== DICIONÁRIO E GRAMÁTICA ===
{dict_ctx}
=== FIM DO DICIONÁRIO ===

{sample_ctx}

=== EXEMPLOS DE TRADUÇÕES CORRETAS ===
Português: "Meu nome é Maria"
Hunsrickisch: "Mein Naame is Maria"

Português: "Eu tenho um cachorro"
Hunsrickisch: "Ich hon en Hund"

Português: "Bom dia"
Hunsrickisch: "Gummeuend"

Português: "Tudo bem?"
Hunsrickisch: "Alles gud?"
=== FIM DOS EXEMPLOS ===

Português: "{question}"
Hunsrickisch:"""
        
        try:
            return ollama.generate(model=GEN_MODEL, prompt=prompt, 
                                 options={'temperature': 0.15, 'top_p': 0.85, 'top_k': 35, 
                                         'repeat_penalty': 1.3, 'num_predict': 100})['response'].strip()
        except Exception as e:
            return f"Erro: {e}"
    
    def interactive_mode(self):
        """Start interactive Q&A session"""
        print("\n" + "="*60)
        print("🗣️  HUNSRIK RAG - Mode Interativo")
        print("="*60)
        print("Comandos: 'quit'/'exit' (sair) | 'reprocess' (recarregar)\n")
        
        while True:
            try:
                user_input = input("💬 Você: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Até logo!")
                    break
                if user_input.lower() == 'reprocess':
                    self.process_files()
                    continue
                
                answer = self.query(user_input)
                print(f"\n🤖 Hunsrik: {answer}\n" + "-"*60 + "\n")
            except KeyboardInterrupt:
                print("\n\n👋 Até logo!")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}\n")


def main():
    print("\n🇧🇷 HUNSRIK RAG SYSTEM")
    rag = HunsrikRAG()
    
    if not rag.vector_store:
        if input("\n� Processar arquivos? (y/n): ").lower() == 'y':
            rag.process_files()
    
    rag.interactive_mode()


if __name__ == "__main__":
    main()
