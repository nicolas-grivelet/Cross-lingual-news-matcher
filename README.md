# Cross-lingual News Matcher 📰

This project is a semantic search engine designed to identify similar news stories across multiple languages.

The goal is to bridge the gap between international media sources without relying on literal word-for-word translation.

## 💡 Why this project?

Monitoring global news usually requires translating every source into a single language, which is slow and often loses semantic nuance.

I built this tool to enable direct semantic matching between texts in different languages using a shared vector space:

### 🔹 Semantic Embeddings

- Uses the paraphrase-multilingual-MiniLM-L12-v2 model from Sentence-Transformers.
- Supports 50+ languages natively.
- Maps different languages into a shared mathematical space.
- Optimized for CPU execution, making it fast on any standard laptop.

### 🔹 Vector Proximity

Instead of keyword matching, the engine calculates the Cosine Similarity between article embeddings.

This allows the system to understand that a French article about "renewable energy" is contextually identical to a Spanish one about "energías limpias," even if they share zero common words.

## 🚀 Features

- **Cross-lingual:** Find matches across 50+ languages without an intermediate translation step.
- **Vector-based Relevance:** Ranks results by semantic proximity score (0 to 1).
- **CPU Optimized:** Lightweight MiniLM architecture for fast local inference.
- **Scalable:** Efficiently encodes and ranks candidate lists against a source query.

## 🛠️ Installation

```bash
git clone https://github.com/nicolas-grivelet/cross-lingual-news-matcher.git
cd cross-lingual-news-matcher
pip install -r requirements.txt
````

## 💻 Usage Example

```python
from news_matcher import NewsMatcher

matcher = NewsMatcher()

# Source text (French)
source = "La France accélère sa transition écologique."

# Candidate list (Multilingual)
candidates = [
    "France accelerates its ecological transition by investing heavily.",
    "Bayern München gewinnt in der Bundesliga.",
    "El gobierno promueve el uso de vehículos eléctricos."
]

# Rank candidates by semantic similarity
results = matcher.rank_matches(source, candidates)

for rank, match in enumerate(results, 1):
    print(f"Rank {rank}: Score {match['score']:.4f}")
    print(f"Content: {match['candidate_text']}")
```

## 📝 License

Distributed under the MIT license. See the LICENSE file for more information.
