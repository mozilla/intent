# Merino Query Normalization

Exploration of query normalization techniques to improve provider matching in [Merino](https://github.com/mozilla-services/merino-py). Covers providers: **ADM, Finance, FlightAware, Sports, Weather**.

The normalization pipeline converts raw user queries into a canonical form before matching, with the goal of increasing suggestion hit rate without sacrificing precision.

## How it works

The pipeline applies normalization steps in order, short-circuiting on the first match:

1. **Tier A** — Unicode normalization, punctuation cleanup, case folding
2. **Canonical exact match** — direct lookup against known keywords
3. **Prefix index lookup** — matches partial/incomplete queries
4. **Typo correction** — spell-check with SymSpell
5. **Token reordering** — BM25-based reranking (e.g. "prime amazon" → "amazon prime")
6. **Join/split heuristics** — handles run-together or hyphenated terms (e.g. "door dash" → "doordash")
7. **Brand name normalization** — maps common brand variants to canonical forms

## Running the notebook

### Prerequisites

- Python 3.10+
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) with access to `stage-mozsoc-articles`
- Jupyter

### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Authenticate with GCP:
   ```bash
   gcloud auth login
   ```

3. Download keyword/data files from GCS (run cell 4 in the notebook, or manually):
   ```bash
   mkdir -p ./data
   gsutil -m cp \
     "gs://stage-mozsoc-articles/search_and_suggest/exploration_data/adm_keywords.json" \
     "gs://stage-mozsoc-articles/search_and_suggest/exploration_data/finance_tickers.json" \
     "gs://stage-mozsoc-articles/search_and_suggest/exploration_data/flightaware_airlines.json" \
     ./data/
   ```

4. Launch the notebook:
   ```bash
   jupyter notebook Query_Normalization_Exploration.ipynb
   ```

### Running the notebook

Run cells top-to-bottom. The notebook is organized into sections:

| Section | Description |
|---------|-------------|
| Setup | Downloads keyword files and installs dependencies |
| Normalization Pipeline Code | Full pipeline implementation |
| Pipeline Matching | Simulates provider matching (ADM, Finance, FlightAware, etc.) |
| Load Keywords / Data | Loads JSON files and builds in-memory indices |
| Trace Example Query | Step-by-step trace for a single query |
| Batch Compare | Runs a test suite of queries and checks expected match behavior |

The `trace(query)` function at the bottom is useful for debugging a specific query through each pipeline step.
