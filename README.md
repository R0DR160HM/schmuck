# 🇧🇷 Hunsrik Language RAG Translation System

Sistema RAG (Retrieval-Augmented Generation) avançado para tradução Português → Hunsriqueano usando Gemma 3, com busca híbrida inteligente e extração contextual.

## 📚 O Que Este Sistema Faz

- **Processa múltiplos tipos de recursos**:
  - 📖 Dicionários Português-Hunsrik
  - 📝 Regras gramaticais
  - 📰 Textos de exemplo (Hunsrickisch Wikipedia)
- **Chunking inteligente** adaptado por tipo de recurso
- **Busca híbrida em 3 fases**:
  1. Busca no dicionário com boost de palavras-chave
  2. Extração automática de termos hunsriqueano
  3. Busca contextual em textos reais
- **Tradução com contexto dinâmico** usando Gemma 3
- **Logging completo** de todas as traduções

## 🚀 Quick Start

### 1. Pré-requisitos

Certifique-se de ter o Ollama instalado e os modelos baixados:

```bash
# Verificar se Ollama está instalado
ollama --version

# Baixar os modelos necessários
ollama pull gemma3n:e2b # Você pode mudar para um modelo maior alterando o `GEN_MODEL` dentro de `schmuck.py`, mas o uso de modelos MENORES é contra-indicado
ollama pull embeddinggemma
```

### 2. Instalar Dependências Python

```bash
pip install -r requirements.txt
```

### 3. Organizar os Recursos

Organize seus arquivos na pasta `resources/`:

```
resources/
├── dicts/          # Dicionários PT-HRX (PDFs ou TXTs)
└── samples/        # Textos em Hunsrik (ex: artigos Wikipedia)
```

### 4. Executar o Sistema

```bash
python schmuck.py
```

Na primeira execução:
1. ✅ Extrai texto de todos os PDFs e TXTs
2. ✅ Cria chunks inteligentes por tipo de recurso
3. ✅ Gera embeddings com `embeddinggemma`
4. ✅ Salva vector store em `hunsrik_vectors.json`
5. ✅ Inicia modo interativo

## 💬 Exemplo de Uso (dados inventados)

```
You: Eu tenho um cachorro grande

🔍 Fase 1: Buscando no dicionário...
   ✓ 15 entradas do dicionário
   [1] Score: 0.876 | cachorro → Hund [substantivo masculino]...
   [2] Score: 0.734 | ter → hon, hawwe [verbo]...
   [3] Score: 0.621 | grande → groos, grooss [adjetivo]...

   🎯 Termos Hunsrik extraídos: hund, hon, groos, en, ich

🔍 Fase 2: Buscando exemplos em textos Hunsrik...
   ✓ 8 exemplos de uso encontrados
   [1] Ich hon en groosse Hund. De Hund is sehr freindlich...
   [2] Mein Vadder hot en Hund unn en Katz...

🤖 Gerando tradução com gemma3n:e2b...

🤖 Answer:
Ich hon en groose Hund
```

## 📝 Comandos

- Digite qualquer frase em português para traduzir
- `reprocess` - Recarregar e reprocessar todos os recursos
- `quit` ou `exit` - Sair do programa

## 🔧 Configuração

Edite a seção de configuração em `schmuck.py`:

```python
# ---------- CONFIG ----------
EMBED_MODEL = "embeddinggemma"    # Modelo de embeddings
GEN_MODEL = "gemma3n:e2b"         # Modelo de geração
CHUNK_SIZE = 300                  # Caracteres por chunk
CHUNK_OVERLAP = 120               # Overlap entre chunks
TOP_K_DICT = 20                   # Chunks do dicionário (dinâmico)
TOP_K_SAMPLES = 8                 # Chunks de exemplos
LOG_FILE = "translation_log.jsonl" # Log de traduções
# ----------------------------
```

## 🧠 Arquitetura do Sistema

### **Fase 1: Vetorização (Indexação)**

```
📂 resources/
    ├── dicts/     → Chunking por entrada de dicionário
    └── samples/   → Chunking por caracteres (300 chars)
         ↓
    [Limpeza de metadados]
         ↓
    [Geração de embeddings com embeddinggemma]
         ↓
    💾 hunsrik_vectors.json
```

### **Fase 2: Busca Híbrida (3 Etapas)**

```
Input: "Eu tenho um cachorro"
    ↓
[ETAPA 1] Busca no Dicionário
    ├─ Query variations: ["eu", "tenho", "cachorro", "eu hunsrik"...]
    ├─ Keyword boost: +50% se palavra exata encontrada
    ├─ Match count: prioriza chunks com múltiplas palavras
    └─ Resultado: TOP 20 chunks (dinâmico por tamanho da frase)
    ↓
[ETAPA 2] Extração de Termos Hunsrik
    ├─ Método 1: Análise de setas (→)
    │   ├─ Se tem seta: todas palavras à direita
    │   └─ Se não: primeira palavra
    ├─ Método 2: Características linguísticas (scoring)
    │   ├─ Vogais duplas (aa, ee, oo): +2 pontos
    │   ├─ Umlauts (ä, ë, ï, ö, ü): +3 pontos
    │   ├─ Padrões germânicos (ge-, ver-, fer-): +1 ponto
    │   └─ Filtro de stopwords portuguesas
    └─ Resultado: ["hund", "hon", "ich", "en", "groos"]
    ↓
[ETAPA 3] Busca em Samples
    ├─ Busca com termos extraídos
    ├─ Boost: +20% se mesmo sample tem múltiplos termos
    └─ Resultado: TOP 8 textos de exemplo
    ↓
[GERAÇÃO] Prompt Dinâmico
    ├─ Contexto do dicionário (20 chunks)
    ├─ Exemplos reais extraídos dos samples
    └─ Gemma 3 gera tradução (temp=0.15)
```

## 🎯 Diferenciais Técnicos

### **1. Chunking Inteligente**
- **Dicionários**: Detecta entradas por regex, mantém estrutura completa
- **Samples**: Chunking por caracteres com overlap proporcional

### **2. Limpeza de Metadados**
- **Converte** categorias gramaticais: `sm` → `[substantivo masculino]`
- **Remove** ruído: transcrições fonéticas, etimologias, marcadores de domínio

### **3. Busca com Keyword Boost**
- +50% de score para chunks com palavra exata
- Reduz falsos positivos temáticos

### **4. Agregação por Match Count**
- Prioriza chunks que aparecem em múltiplas queries
- Identifica traduções mais relevantes automaticamente

### **5. Extração de Termos com Scoring**
- Sistema de pontuação multicritério
- Filtra palavras portuguesas automaticamente
- Extrai de **TODOS** os resultados do dicionário

### **6. Fallback Inteligente**
- Se não encontrar termos Hunsrik, usa palavras originais
- Garante que sempre haverá contexto de samples

### **7. Exemplos Dinâmicos**
- Extrai frases reais dos samples encontrados
- Adapta ao contexto da consulta específica

### **8. Logging Completo**
- Salva todas traduções em `translation_log.jsonl`
- Inclui: input, output, chunks usados, termos extraídos, scores

## 📂 Estrutura do Projeto

```
schmuck/
├── schmuck.py                  # Sistema RAG principal
├── requirements.txt            # Dependências Python
├── README.md                   # Esta documentação
├── FOLDER_STRUCTURE.md         # Estrutura detalhada
├── hunsrik_vectors.json        # Vector store (gerado automaticamente)
├── translation_log.jsonl       # Log de traduções (gerado automaticamente)
└── resources/                  # Seus materiais Hunsrik
    ├── dicts/                  # Dicionários PT-HRX
    └── samples/                # Textos de exemplo (Wikipedia, etc)
        ├── WIKI - Hunsrickisch Sproch.txt
        ├── WIKI - Brasil.txt
        └── ...
```

## 🎯 Por Que RAG em Vez de Fine-Tuning?

Para línguas de baixo recurso como Hunsrik, RAG é superior:

| Critério | RAG ✅ | Fine-Tuning ❌ |
|----------|--------|----------------|
| **Dados necessários** | Poucos documentos | Milhares de pares PT-HRX |
| **Atualização** | Adicionar PDF e reprocessar | Retreinar modelo completo |
| **Preservação** | Estrutura original intacta | Perde nuances de entrada |
| **Recursos** | CPU suficiente | GPU de alto desempenho |
| **Tempo** | Minutos para indexar | Horas/dias para treinar |
| **Rastreabilidade** | Sabe de onde veio a info | Caixa preta |
| **Custo** | Quase zero | Alto (GPU cloud) |

## 🔮 Melhorias Futuras

- [ ] **OCR integrado** para PDFs escaneados
- [ ] **Cache de embeddings** por arquivo para updates incrementais
- [ ] **Interface web** com Gradio/Streamlit
- [ ] **Suporte a voz** (speech-to-text PT → tradução → text-to-speech HRX)
- [ ] **Fine-tuning híbrido**: RAG para dicionário + modelo fino-tunado para conversação
- [ ] **Multi-direcional**: Hunsrik → Português
- [ ] **Métricas de qualidade**: BLEU score, avaliação humana

## 🐛 Troubleshooting

| Problema | Solução |
|----------|---------|
| `Error: bad character range` | Já corrigido no código (regex das setas) |
| `No embeddings generated` | Verifique se Ollama está rodando: `ollama serve` |
| `Model not found` | Baixe os modelos: `ollama pull gemma3n:e2b` |
| `PDF extraction fails` | PDF pode ser imagem. Use OCR ou converta para TXT |
| `Empty dictionary results` | Verifique se arquivos estão em `resources/dicts/` |
| `Slow on first run` | Normal. Embeddings são gerados uma vez e salvos |

## 📊 Performance

Testado com:
- **Dicionários**: ~3,000 entradas
- **Samples**: ~100 artigos Wikipedia Hunsrik
- **Vector store**: ~15,000 chunks
- **Tempo de indexação**: ~10-15 minutos (primeira vez)
- **Tempo de query**: ~2-4 segundos por tradução
- **Uso de RAM**: ~2-3 GB

## 🤝 Contribuindo

Este é um projeto pessoal, mas sugestões são bem-vindas! Abra uma issue ou PR.

## 📄 Licença

MIT License - Use como quiser!
