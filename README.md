# Hunsrik Language RAG System

Train Gemma 3 to understand and speak Hunsrik using RAG (Retrieval-Augmented Generation) with your PDF materials.

## 📚 What This Does

- Extracts text from your Hunsrik PDFs (dictionary, grammar, stories)
- Creates embeddings using `embeddinggemma`
- Stores them in a searchable vector database
- Uses RAG to answer questions about Hunsrik with context from your materials
- Powered by Gemma 3 via Ollama

## 🚀 Quick Start

### 1. Prerequisites

Make sure you have Ollama installed and the models pulled:

```powershell
# Check if Ollama is installed
ollama --version

# Pull the required models
ollama pull gemma3n:e2b
ollama pull embeddinggemma
```

### 2. Install Python Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the System

```powershell
python schmuck.py
```

On first run, it will:
1. Extract text from all PDFs in the `pdfs/` folder
2. Create embeddings (this may take a few minutes)
3. Save the vector store to `hunsrik_vectors.json`
4. Start interactive mode

## 💬 Example Usage

```
You: How do you say "hello" in Hunsrik?

🤖 Answer: [System retrieves relevant dictionary entries and provides answer]

You: What are the grammar rules for verb conjugation?

🤖 Answer: [System finds and explains grammar rules from your materials]

You: Translate "Ich liebe dich" to Portuguese

🤖 Answer: [Uses dictionary context to provide translation]
```

## 📝 Commands

- Type any question to get an answer
- `reprocess` - Reload and reprocess all PDFs
- `quit` or `exit` - Exit the program

## 🔧 Configuration

Edit the config section in `schmuck.py`:

```python
EMBED_MODEL = "embeddinggemma"    # Embedding model
GEN_MODEL = "gemma3n:e2b"         # Generation model
CHUNK_SIZE = 500                  # Characters per chunk
TOP_K = 5                         # Number of context chunks to retrieve
```

## 📂 Project Structure

```
hunsrik-gemma/
├── schmuck.py              # Main RAG system
├── requirements.txt        # Python dependencies
├── pdfs/                   # Your Hunsrik PDF materials
│   ├── Your resources
└── hunsrik_vectors.json    # Generated vector store (created on first run)
```

## 🎯 Why RAG Over Fine-Tuning?

For your use case, RAG is superior because:

1. **Preserves Structure**: Dictionary entries and grammar rules stay intact
2. **Less Data Required**: Works great with your limited specialized materials
3. **Easy Updates**: Add new PDFs anytime without retraining
4. **Source Attribution**: Know which PDF/page the answer came from
5. **Resource Efficient**: No GPU-intensive training required
6. **Immediate Results**: Start using it right away

## 🔮 Future Enhancements

If you want to combine both approaches later:
- Use RAG for dictionary lookups and grammar rules
- Fine-tune a smaller model on conversational Hunsrik from the stories
- Use the fine-tuned model with RAG for best results

## 🐛 Troubleshooting

**"No embeddings generated"**: Make sure Ollama is running (`ollama serve`)

**"Model not found"**: Pull the models first (`ollama pull gemma3n:e2b`)

**"PDF extraction fails"**: Some PDFs may be image-based and need OCR

## 📄 License

This is your project - use it as you wish!
