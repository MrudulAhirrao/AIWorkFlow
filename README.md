# AI Intent Discovery Pipeline

## Project Overview
This tool automates the discovery of "missing user intents" in customer support logs. It utilizes a **Hybrid Neuro-Symbolic Architecture**:
1.  **Unsupervised Learning (K-Means):** To cluster thousands of messages at scale without labeling.
2.  **Generative AI (Gemini 1.5):** To semantically analyze cluster centroids and propose new intent categories.

## Key Features
* **Scalable:** Uses local vector embeddings (`sentence-transformers`) to handle 10k+ messages efficiently.
* **Cost-Efficient:** Reduces LLM token usage by ~95% by analyzing clusters instead of raw messages.
* **Robust:** Dynamic JSON parsing handles complex data structures automatically.

## Setup & Usage

### Prerequisites
* Python 3.8+
* Google Gemini API Key

### Installation
```bash
pip install pandas numpy scikit-learn sentence-transformers google-generativeai python-dotenv
```
---
### Configuration
Create a .env file and add your API key:

```bash
GEMINI_API_KEY=your_api_key_here
```
---
### Run the Pipeline
Place your data file (inputs_for_assignment.json) in the root directory and run:

```bash
python intent_pipeline.py
```
---
### Methodology
The pipeline follows a strict "Reasoning" workflow:
  - Vectorization: Maps semantic meaning to 384-dimensional space.
  - Clustering: Groups semantically similar queries using K-Means.
  - Gap Analysis: The LLM compares the group against the Current Intent Map to find gaps (e.g., finding "Usage Instructions" hidden within "Product Info").
---
### Output
The script generates final_report.json containing:
  - Cluster Summaries
  - Proposed New Intents (with reasoning)
  - Representative User Quotes
