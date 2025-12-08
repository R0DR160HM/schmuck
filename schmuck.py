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
CHUNK_OVERLAP = 120  # overlap between chunks (more overlap)
TOP_K_DICT = 20 # chunks from dictionary (dynamic based on input)
TOP_K_SAMPLES = 8  # chunks from Hunsrik samples
LOG_FILE = "translation_log.jsonl"  # Log file for all translations
# ----------------------------

# Resource types
RESOURCE_DICT = "dictionary"
RESOURCE_SAMPLE = "sample"


class HunsrikRAG:
    def __init__(self):
        self.vector_store = []
        self.load_vector_store()
    
    def log_query(self, input_text: str, prompt: str, response: str, 
                  dict_chunks: List[Dict], sample_chunks: List[Dict], hunsrik_terms: List[str]):
        """Log query details to file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": input_text,
            "response": response,
            "prompt": prompt,
            "context": {
                "dictionary_entries": [
                    {
                        "text": chunk['text'],
                        "source": chunk['source'],
                        "similarity": chunk['similarity']
                    } for chunk in dict_chunks
                ],
                "sample_texts": [
                    {
                        "text": chunk['text'],
                        "source": chunk['source'],
                        "similarity": chunk['similarity']
                    } for chunk in sample_chunks
                ],
                "hunsrik_terms_extracted": hunsrik_terms
            },
            "stats": {
                "num_dict_entries": len(dict_chunks),
                "num_samples": len(sample_chunks),
                "num_hunsrik_terms": len(hunsrik_terms)
            }
        }
        
        try:
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️  Warning: Could not write to log file: {e}")
    
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
    
    def clean_dictionary_metadata(self, text: str) -> str:
        """Convert dictionary metadata to readable format"""
        import re
        
        # Convert grammatical categories to readable format
        grammar_map = {
            'sf': '[substantivo feminino]', 'sm': '[substantivo masculino]', 'sn': '[substantivo neutro]',
            'adj': '[adjetivo]', 'adv': '[advérbio]', 'v': '[verbo]',
            'vt': '[verbo transitivo]', 'vi': '[verbo intransitivo]',
            'prep': '[preposição]', 'conj': '[conjunção]',
            'interj': '[interjeição]', 'pron': '[pronome]', 'num': '[numeral]'
        }
        
        cleaned = text
        
        # Replace grammatical categories
        for abbr, full in grammar_map.items():
            cleaned = re.sub(r'\b' + abbr + r'\b', full, cleaned, flags=re.IGNORECASE)
        
        # Remove patterns that add noise
        remove_patterns = [
            r'/[^/]+/',  # Phonetic transcriptions like /ˈɔːpaˌhoːa/
            r'\b(nie|gmc|gmf|gml|gmh|gml|gmo|gmw|grc|hno|inc)\b',  # Ethymologies
            r'\b(Anat|Geog|Bot|Pop|Zool|Med|Culin|Arquit|Meteor|Agric|Relig|Econ|Pol|Hist|NP)\b',  # Domain markers
            r'\bSin\b',  # Synonym marker
            r'\b§\b',  # Example marker
            r'\(pl\s+\w+\)',  # Plural forms in parentheses like (pl Aaperhore)
        ]
        
        for pattern in remove_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove multiple spaces and clean up
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def chunk_dictionary_entries(self, text: str, source: str) -> List[Dict]:
        """Split dictionary text into individual entries intelligently"""
        import re
        chunks = []
        
        # Pattern 1: Lines starting with capital letter followed by phonetic /.../ or word definition
        # This catches entries like: "Anillblau /aˈnilˌplaw/ sn anil..."
        entry_pattern = r'^([A-ZÄËÏ][a-zäëï]*(?:[a-zäëï]+)*)\s+(?:/[^/]+/|→)'
        
        lines = text.split('\n')
        current_entry = []
        entry_word = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts a new entry
            match = re.match(entry_pattern, line, re.MULTILINE)
            
            if match:
                # Save previous entry if exists
                if current_entry:
                    entry_text = '\n'.join(current_entry)
                    if len(entry_text) > 10:  # Only save substantial entries
                        chunks.append({
                            'text': entry_text,
                            'source': source,
                            'resource_type': RESOURCE_DICT,
                            'entry_word': entry_word
                        })
                
                # Start new entry
                entry_word = match.group(1)
                current_entry = [line]
            elif current_entry:
                # Continue current entry
                # Stop if line looks like it might be a new entry without phonetic
                # (starts with capital and has Portuguese/Hunsrik pattern)
                if re.match(r'^[A-ZÄËÏ][a-zäëï]+\s+(→|sf|sm|sn|adj|adv)', line):
                    # Save previous entry
                    entry_text = '\n'.join(current_entry)
                    if len(entry_text) > 10:
                        chunks.append({
                            'text': entry_text,
                            'source': source,
                            'resource_type': RESOURCE_DICT,
                            'entry_word': entry_word
                        })
                    # Start new entry
                    entry_word = line.split()[0]
                    current_entry = [line]
                else:
                    current_entry.append(line)
            else:
                # Line doesn't match entry pattern and no current entry
                # Might be intro text, treat as separate chunk
                if len(line) > 30:  # Only substantial text
                    chunks.append({
                        'text': line,
                        'source': source,
                        'resource_type': RESOURCE_DICT,
                        'entry_word': None
                    })
        
        # Don't forget last entry
        if current_entry:
            entry_text = '\n'.join(current_entry)
            if len(entry_text) > 10:
                chunks.append({
                    'text': entry_text,
                    'source': source,
                    'resource_type': RESOURCE_DICT,
                    'entry_word': entry_word
                })
        
        return chunks
    
    def chunk_text(self, text: str, source: str, resource_type: str = RESOURCE_DICT) -> List[Dict]:
        """Split text into chunks - entry-based for dictionaries, character-based for others"""
        chunks = []
        text = text.strip()
        
        # Use entry-based chunking for dictionaries
        if resource_type == RESOURCE_DICT:
            chunks = self.chunk_dictionary_entries(text, source)
            # Clean metadata from each chunk
            for chunk in chunks:
                chunk['text'] = self.clean_dictionary_metadata(chunk['text'])
            return chunks
        
        # Use character-based chunking for samples
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
            
            # Proportional overlap: max 1/3 of chunk size
            overlap = min(CHUNK_OVERLAP, len(chunk) // 3)
            start = end - overlap
        
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
            folder_entries = 0  # Track dictionary entries
            
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
                    
                    # Count dictionary entries
                    if resource_type == RESOURCE_DICT:
                        entries_with_word = [c for c in chunks if c.get('entry_word')]
                        folder_entries += len(entries_with_word)
                    
                    total_files += 1
            
            # Show stats
            if resource_type == RESOURCE_DICT and folder_entries > 0:
                print(f"   → Created {folder_chunks} chunks ({folder_entries} dictionary entries) from {folder_name}\n")
            else:
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
    
    def retrieve_with_keyword_boost(self, query: str, top_k: int = 15, resource_types: List[str] = None) -> List[Dict]:
        """Retrieve with keyword boost for exact matches"""
        results = self.retrieve(query, top_k * 2, resource_types)
        
        # Boost chunks that contain the query word exactly
        query_lower = query.lower()
        for result in results:
            if query_lower in result['text'].lower():
                result['similarity'] *= 1.5  # 50% boost for exact keyword match
        
        # Re-sort and return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def hybrid_retrieve(self, text: str) -> Dict[str, List[Dict]]:
        """Intelligent hybrid retrieval: dictionary first, then samples for context"""
        # Extract individual words for dictionary lookup
        words = text.lower().replace('?', '').replace('!', '').replace('.', '').replace(',', '').split()
        content_words = [w for w in words if len(w) > 2]
        
        # Create query variations (more focused)
        queries = [text]  # Full text query
        for word in content_words:
            queries.append(word)
        
        # STEP 1: Search dictionary with keyword boost
        dict_results = {}
        for query in queries:
            results = self.retrieve_with_keyword_boost(query, top_k=10, 
                                  resource_types=[RESOURCE_DICT])
            for result in results:
                text_key = result['text']
                if text_key in dict_results:
                    # Increment match count and keep best similarity
                    dict_results[text_key]['match_count'] += 1
                    dict_results[text_key]['similarity'] = max(
                        dict_results[text_key]['similarity'], 
                        result['similarity']
                    )
                else:
                    result['match_count'] = 1
                    dict_results[text_key] = result
        
        # Sort by match_count first, then similarity
        # Dynamic THunsrikOP_K based on input length
        dynamic_top_k = max(TOP_K_DICT, len(content_words))
        dict_sorted = sorted(
            dict_results.values(), 
            key=lambda x: (x['match_count'], x['similarity']), 
            reverse=True
        )[:dynamic_top_k]
        
        # STEP 2: Extract Hunsrik words from ALL dictionary results
        hunsrik_terms = set()
        pt_stopwords = {'de', 'da', 'do', 'das', 'dos', 'para', 'com', 'sem', 'por', 'em', 'um', 'uma', 'ter', 'ser', 'estar'}
        
        import re
        
        # Extract from ALL dict_results, not just top k
        for result in dict_results.values():
            text_lower = result['text'].lower()
            
            # Method 1: Extract Hunsrik words based on arrow marker or first word
            if '→' in result['text'] or '->' in result['text']:
                # All words to the right of the arrow are Hunsrik
                arrow_split = re.split(r'→|->|→', result['text'])
                if len(arrow_split) > 1:
                    right_side = arrow_split[1]
                    # Extract all words from the right side (before grammatical markers)
                    # Stop at first [ or ( to avoid getting metadata
                    right_side = re.split(r'[\[\(]', right_side)[0]
                    words = re.findall(r'[\wäÄËëÏïÖöÜü]+', right_side)
                    for word in words:
                        word_clean = word.strip().lower()
                        if len(word_clean) > 1 and word_clean not in pt_stopwords:
                            hunsrik_terms.add(word_clean)
            else:
                # Just the first word of the chunk is Hunsrik
                words = re.findall(r'[\wäÄËëÏïÖöÜü]+', result['text'])
                if words:
                    first_word = words[0].strip().lower()
                    if len(first_word) > 1:
                        hunsrik_terms.add(first_word)
            
            # Method 2: Extract words with Hunsrik characteristics (with scoring)
            for word in result['text'].split():
                word_clean = word.strip('.,;:()[]"!?-–—').lower()
                if len(word_clean) > 2 and word_clean not in pt_stopwords:
                    score = 0
                    
                    # Hunsrik indicators with weights
                    if any(dv in word_clean for dv in ['aa', 'ee', 'oo', 'uu', 'ei', 'au', 'eu', 'w', 'pp', 'bb', 'tz', 'ff']):
                        score += 2  # Strong indicator
                    if 'sch' in word_clean:
                        score += 2
                    if any(u in word_clean for u in ['ä', 'ë', 'ï']):
                        score += 3  # Very strong indicator
                    if word_clean.startswith(('ge', 'ver', 'be', 'fer', 'en', 'un', 'ich', 'mein', 'dein')):
                        score += 1
                    
                    # Require score >= 2 to avoid Portuguese false positives
                    if score >= 2:
                        hunsrik_terms.add(word_clean)
        
        # STEP 3: Search samples using ALL extracted Hunsrik terms
        sample_results = {}
        
        # Fallback: if no Hunsrik terms found, use original words
        if not hunsrik_terms:
            hunsrik_terms = set(content_words)
        
        # Use all terms but prioritize longer, more specific ones
        sorted_terms = sorted(hunsrik_terms, key=len, reverse=True)[:15]  # Top 15 terms
        
        for term in sorted_terms:
            results = self.retrieve(term, top_k=TOP_K_SAMPLES, 
                                  resource_types=[RESOURCE_SAMPLE])
            for result in results:
                text_key = result['text']
                if text_key in sample_results:
                    # Boost score if same sample found with multiple terms
                    sample_results[text_key]['similarity'] += result['similarity'] * 0.2
                else:
                    sample_results[text_key] = result
        
        sample_sorted = sorted(sample_results.values(), key=lambda x: x['similarity'], reverse=True)[:TOP_K_SAMPLES]
        
        return {
            'dictionary': dict_sorted,
            'samples': sample_sorted,
            'hunsrik_terms': sorted_terms
        }
    
    def query(self, question: str, verbose: bool = True) -> str:
        """Query with hybrid retrieval: dictionary + samples for context"""
        import re
        
        if verbose:
            print(f"\n💬 Texto para traduzir: {question}\n")
            print("🔍 Fase 1: Buscando no dicionário...")
        
        # Use hybrid retrieval
        results = self.hybrid_retrieve(question)
        dict_chunks = results['dictionary']
        sample_chunks = results['samples']
        hunsrik_terms = results.get('hunsrik_terms', [])
        
        if not dict_chunks:
            return "Nenhuma informação relevante encontrada. Execute 'reprocess' primeiro."
        
        if verbose:
            print(f"   ✓ {len(dict_chunks)} entradas do dicionário")
            for i, chunk in enumerate(dict_chunks[:10], 1):
                preview = chunk['text'][:60].replace('\n', ' ')
                print(f"   [{i}] Score: {chunk['similarity']:.3f} | {preview}...")
            
            if hunsrik_terms:
                print(f"\n   🎯 Termos Hunsrik extraídos: {', '.join(hunsrik_terms[:10])}")
                if len(hunsrik_terms) > 10:
                    print(f"      ... e mais {len(hunsrik_terms) - 10} termos")
            
            if sample_chunks:
                print(f"\n🔍 Fase 2: Buscando exemplos em textos Hunsrik...")
                print(f"   ✓ {len(sample_chunks)} exemplos de uso encontrados")
                for i, chunk in enumerate(sample_chunks[:10], 1):
                    preview = chunk['text'][:60].replace('\n', ' ')
                    print(f"   [{i}] {preview}...")
        
        # Build contexts separately
        dict_context = "\n\n".join([chunk['text'] for chunk in dict_chunks])
        
        # Generate dynamic examples from samples
        sample_context = ""
        dynamic_examples = ""
        
        if sample_chunks:
            sample_context = "\n\n=== EXEMPLOS DE USO EM CONTEXTO (textos Hunsrickisch) ===\n"
            sample_context += "\n".join([chunk['text'][:200] for chunk in sample_chunks[:3]])
            sample_context += "\n=== FIM DOS EXEMPLOS ==="
            
            # Extract 2-3 sentences from samples as dynamic examples
            dynamic_examples = "\n\n=== EXEMPLOS DE FRASES REAIS ===\n"
            example_count = 0
            for chunk in sample_chunks[:5]:
                # Try to extract complete sentences
                sentences = re.split(r'[.!?]+', chunk['text'])
                for sent in sentences:
                    sent = sent.strip()
                    if 20 < len(sent) < 100 and example_count < 3:
                        dynamic_examples += f"Exemplo: {sent}\n"
                        example_count += 1
            dynamic_examples += "=== FIM DOS EXEMPLOS ==="
        
        # Build prompt
        prompt = f"""Você é um tradutor especializado em hunrisqueano (Hunsrickisch). Use APENAS as informações do dicionário fornecido abaixo. NÃO invente palavras.

=== DICIONÁRIO ===
{dict_context}
=== FIM DO DICIONÁRIO ===

{sample_context}

{dynamic_examples}

INSTRUÇÕES:
1. Procure cada palavra no dicionário acima
2. Use a ortografia do dicionário
3. Use os exemplos como guia para estrutura das frases
4. Responda APENAS com a tradução, sem explicações
5. Traduza a frase COMPLETA, mantendo a estrutura original

=== EXEMPLOS DE TRADUÇÕES ===
Português: "Meu nome é Maria"
Hunsrickisch: "Mein Naame is Maria"

Português: "Eu tenho um cachorro"
Hunsrickisch: "Ich hon en Hund"

Português: "Bom dia"
Hunsrickisch: "Gummeuend"

Português: "Tudo bem?"
Hunsrickisch: "Alles gud?"

Português: "Onde está o Pedro?"
Hunsrickisch: "Wo is de Pedro?"
=== FIM DOS EXEMPLOS ===

Português: "{question}"
Hunsrickisch:"""
        
        if verbose:
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
            result = response['response'].strip()
            
            # Log the query details
            self.log_query(
                input_text=question,
                prompt=prompt,
                response=result,
                dict_chunks=dict_chunks,
                sample_chunks=sample_chunks,
                hunsrik_terms=hunsrik_terms
            )
            
            return result
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
