# 📁 Resource Folder Structure

The system now supports organized multi-folder processing for better retrieval!

## Structure

```
resources/
├── dicts/          ← Portuguese-Hunsrik dictionaries (PDFs/TXTs)
├── grammar/        ← Grammar rules and learning materials
└── samples/        ← Pure Hunsrik texts (Wikipedia, stories, etc.)
```

## How It Works

### Phase 1: Dictionary & Grammar Search
When you translate Portuguese → Hunsrik, the system:
1. Searches `dicts/` and `grammar/` folders for relevant entries
2. Extracts Hunsrik translations and vocabulary
3. Returns TOP 12 most relevant dictionary entries

### Phase 2: Context from Samples (Intelligent!)
The system then:
1. Identifies Hunsrik words from the dictionary results
2. Searches `samples/` for those specific Hunsrik terms
3. Provides real usage examples to improve naturalness
4. Returns TOP 5 sample chunks showing words in context

## Why This Works Better

**Without samples:** Model only sees dictionary entries (isolated words)
- Result: "Mein Naame is Rodrigo" ✓ (correct but stiff)

**With samples:** Model sees dictionary + real usage in context
- Result: "Mein Naame is Rodrigo" with natural phrasing ✓✓

The samples help with:
- Word order and sentence structure
- Natural phrasing patterns
- Idiomatic expressions
- Proper grammar application

## Key Features

✅ **Smart retrieval:** Searches Portuguese in dict, then Hunsrik in samples
✅ **Prevents hallucination:** Dictionary is primary source
✅ **Contextual learning:** Samples show proper usage
✅ **Resource tagging:** Tracks which type of resource each chunk came from

## To Use

1. Organize your files into the 3 folders
2. Run `python schmuck.py`
3. Type `reprocess` to rebuild with new structure
4. Start translating!

## Example Output

```
🔍 Fase 1: Buscando no dicionário e gramática...
   ✓ 12 entradas do dicionário/gramática
   [1] Score: 0.892 | nome (POR) = Naame (HRX)...
   [2] Score: 0.856 | ser/estar (POR) = sinn/is (HRX)...
   [3] Score: 0.801 | meu/minha (POR) = mein/mei (HRX)...

🔍 Fase 2: Buscando exemplos em textos Hunsrik...
   ✓ 5 exemplos de uso encontrados
   [1] Mein Naame is Klaus unn ich wohn in Brasil...
   [2] Sei Naame waar Maria, unn sei Mann...
```

This dual-phase approach gives you accuracy (from dictionary) + naturalness (from samples)!
