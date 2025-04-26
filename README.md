# 🧠 AI-Powered Symptom Checker

An intelligent, retrieval-augmented medical assistant that interprets natural language symptom descriptions and outputs probable diseases, causes, and treatment strategies using a fusion of lexical, semantic, and keyword-based retrieval methods—enhanced by large language models.

---

## 🚀 Project Overview

This project builds a **retrieval-based medical diagnosis system** that processes free-form user inputs and returns interpretable, medically-informed suggestions. Unlike traditional keyword matchers, our system integrates:

- **BM25** lexical matching (with query paraphrasing)
- **Semantic similarity** retrieval (via SBERT)
- **Keyword-driven** semantic filtering
- A **score fusion** module with tunable weights
- **LLM generation** for causes and treatments

**Input**: Natural language symptom description  
**Output**: Ranked disease list + cause explanation + treatment recommendation

The full architecture is described in detail in our paper: *AI-Powered Symptom Checker (2025)*

---

## 📁 File Structure

```
key_threshold/
├── output/                  # Inference & evaluation output
│   ├── nli_baseline.json        # GPT2-based NLI consistency scores
│   ├── nli_results.json         # Model-level consistency results
│   └── output_evaluate.json     # Full test outputs
├── test/                    # Test set results from three methods
│   ├── bert.json
│   ├── bm25_false.json
│   ├── bm25_true.json
│   └── key.json
├── train/                   # Training sets for retrieval/threshold
│   ├── train_bert.json
│   ├── train_bm25.json
│   └── train_key.json
├── main.py                 # 🔑 Main entry point: run and input symptoms
├── train.py                # Generates training data for weight tuning
├── test.py                 # Runs retrieval methods and saves test results
├── baseline.py             # GPT-2 baseline consistency evaluation
├── evaluate_output.py      # Merges predictions and performs evaluation
├── obtain_key_threshold.py # Analyzes threshold impact for keyword method
├── retrieval.py            # 🔍 Retrieval logic: BM25, SBERT, keywords
├── utils.py                # Helper functions (normalization, scoring)
├── mapping.json            # Disease ID to name mapping
├── source_embeddings.pt    # Cached SBERT embeddings of keywords for fast retrieval
├── evalution.ipynb         # Notebook for test set visualization
├── README.md               # 📘 This file
```

---

## 🖥️ How to Use

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

Recommend Python 3.10+ with GPU or MPS acceleration.

### 2. Run the Symptom Checker

```bash
python main.py
```

**Example input:**

```
Please enter your symptoms: I'm having a hard time breathing. I'm sweating a lot, and the phlegm I'm coughing up is a weird color.
```

**Example output:**

```
1. Pneumonia
   Potential cause: Likely bacterial (e.g., Streptococcus)
   Treatment: Empiric antibiotics, supportive care

2. Bronchial Asthma
3. Common Cold
...
```

---

## ⚙️ Core Components

| File | Description |
|------|-------------|
| `main.py` | User-facing interface: run once, enter symptom, receive diagnosis |
| `retrieval.py` | Implements BM25 (with T5-based paraphrasing), SBERT cosine similarity, and keyword matching |
| `train.py` | Prepares training data for weight fusion |
| `test.py` | Evaluates each retrieval method and saves intermediate results |
| `evaluate_output.py` | Aggregates and analyzes model predictions |
| `obtain_key_threshold.py` | Computes keyword accuracy at different thresholds |
| `baseline.py` | NLI-based consistency check for GPT-2 outputs |
| `utils.py` | Shared utilities across pipeline |

---

## 📊 Performance & Evaluation

- **Final model accuracy**: 47.86% on the test set  
- Fusion weights (BM25: 0.5, SBERT: 0.4, Keyword: 0.1) achieved best result  
- **Top-k Accuracy** increases and converges at **k = 3 to 5**
- **Keyword similarity threshold** α = 0.9 gives optimal results for matching

Evaluation results and heatmaps are visualized in `evalution.ipynb`.

---