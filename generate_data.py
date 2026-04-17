"""Generate query normalization data files.

Produces three files in data/:
  - sports_teams.json: team names from SportsData.io API
  - finance_tickers.json: finance keywords from merino-py source
  - word_freq.csv: canonical-filtered word frequencies from BigQuery vocab export

Usage:
    # Generate sports teams (requires SportsData.io API key)
    python generate_data.py --sports-api-key YOUR_KEY

    # Generate word_freq from BigQuery vocab export
    python generate_data.py --vocab-csv /path/to/vocab_past_month.csv

    # Generate finance tickers from merino-py source
    python generate_data.py --merino-py /path/to/merino-py

    # Generate all
    python generate_data.py \
        --sports-api-key YOUR_KEY \
        --vocab-csv /path/to/vocab_past_month.csv \
        --merino-py /path/to/merino-py

Data sources:
    Sports teams:
        API: https://sportsdata.io
        Leagues: NFL, NBA, NHL, MLB (v3), UCL (v4 soccer)
        ~364 team names

    Finance tickers:
        Source: merino-py/merino/providers/suggest/finance/backends/polygon/
        Files: keyword_ticker_mapping.py, stock_ticker_company_mapping.py,
               etf_ticker_company_mapping.py
        ~221 keywords

    Word frequencies:
        Source: BigQuery query on merino_log_sanitized_v3
        Query:
            WITH queries AS (
                SELECT LOWER(TRIM(query)) AS query,
                       COUNT(DISTINCT session_id) AS session_count
                FROM `moz-fx-data-shared-prod.search_terms_derived.merino_log_sanitized_v3`
                WHERE TIMESTAMP_TRUNC(timestamp, DAY)
                      BETWEEN TIMESTAMP("YYYY-MM-DD") AND TIMESTAMP("YYYY-MM-DD")
                  AND query IS NOT NULL AND query != ''
                GROUP BY 1 HAVING COUNT(DISTINCT session_id) >= 2
            ),
            words AS (
                SELECT word, SUM(session_count) AS freq
                FROM queries, UNNEST(SPLIT(query, ' ')) AS word
                WHERE LENGTH(word) > 4
                GROUP BY 1
            )
            SELECT word, freq FROM words WHERE freq >= 10 ORDER BY freq DESC
        Filtered to canonical words (words appearing in ADM/finance/sports keywords)
        ~2,869 words from 1 month of Firefox logs

Upload to GCS:
    See QueryNormDataUploadFlow.py in ml-services repo:
    ml-services/jobs/metaflow/prospecting/QueryNormDataUploadFlow.py

    Stage: gs://merino-ml-data-stage/query_normalization/
    Prod:  gs://merino-ml-data-prod/query_normalization/
"""

import argparse
import csv
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
_AUTOCOMPLETE_MIN_PREFIX_LEN = 4


def generate_sports_teams(api_key: str) -> None:
    """Fetch team names from SportsData.io and save to sports_teams.json."""
    import requests

    teams: set[str] = set()

    # US leagues (v3)
    for league in ["nfl", "nba", "nhl", "mlb"]:
        url = f"https://api.sportsdata.io/v3/{league}/scores/json/teams?key={api_key}"
        resp = requests.get(url)
        if resp.status_code == 200:
            for t in resp.json():
                name = t.get("Name") or t.get("ShortName")
                if name:
                    teams.add(name.lower())
            print(f"  {league.upper()}: {len(resp.json())} teams")
        else:
            print(f"  {league.upper()}: {resp.status_code} (skipped)")

    # Soccer (v4)
    for league in ["ucl"]:
        url = f"https://api.sportsdata.io/v4/soccer/scores/json/Teams/{league.upper()}?key={api_key}"
        resp = requests.get(url)
        if resp.status_code == 200:
            for t in resp.json():
                name = t.get("Name") or t.get("ShortName")
                if name:
                    teams.add(name.lower())
            print(f"  {league.upper()}: {len(resp.json())} teams")
        else:
            print(f"  {league.upper()}: {resp.status_code} (skipped)")

    out = DATA_DIR / "sports_teams.json"
    out.write_text(json.dumps(sorted(teams), indent=2))
    print(f"  Saved {len(teams)} teams to {out}")


def generate_finance_tickers(merino_py: Path) -> None:
    """Export finance keywords from merino-py source to finance_tickers.json."""
    sys.path.insert(0, str(merino_py))

    from merino.providers.suggest.finance.backends.polygon.stock_ticker_company_mapping import (
        ALL_STOCK_TICKER_COMPANY_MAPPING,
        STOCK_TICKER_EAGER_MATCH_BLOCKLIST,
    )
    from merino.providers.suggest.finance.backends.polygon.etf_ticker_company_mapping import (
        ALL_ETF_TICKER_COMPANY_MAPPING,
        ETF_TICKER_EAGER_MATCH_BLOCKLIST,
    )
    from merino.providers.suggest.finance.backends.polygon.keyword_ticker_mapping import (
        KEYWORD_TO_STOCK_TICKER_MAPPING,
        KEYWORD_TO_ETF_TICKER_MAPPING,
    )

    data = {
        "stock_tickers": dict(ALL_STOCK_TICKER_COMPANY_MAPPING),
        "stock_blocklist": list(STOCK_TICKER_EAGER_MATCH_BLOCKLIST),
        "etf_tickers": dict(ALL_ETF_TICKER_COMPANY_MAPPING),
        "etf_blocklist": list(ETF_TICKER_EAGER_MATCH_BLOCKLIST),
        "keyword_to_stock_ticker": dict(KEYWORD_TO_STOCK_TICKER_MAPPING),
        "keyword_to_etf_tickers": {
            k: list(v) for k, v in KEYWORD_TO_ETF_TICKER_MAPPING.items()
        },
    }

    out = DATA_DIR / "finance_tickers.json"
    out.write_text(json.dumps(data, indent=2))
    n_kw = len(data["keyword_to_stock_ticker"]) + len(data["keyword_to_etf_tickers"])
    print(f"  Saved {n_kw} keywords to {out}")


def generate_word_freq(vocab_csv: Path) -> None:
    """Filter BigQuery vocab export to canonical words and save to word_freq.csv."""
    # Load canonical set
    sports_path = DATA_DIR / "sports_teams.json"
    finance_path = DATA_DIR / "finance_tickers.json"

    canonical: set[str] = set()

    if sports_path.exists():
        canonical.update(json.loads(sports_path.read_text()))

    if finance_path.exists():
        fin = json.loads(finance_path.read_text())
        canonical.update(fin.get("keyword_to_stock_ticker", {}).keys())
        canonical.update(fin.get("keyword_to_etf_tickers", {}).keys())

    adm_path = DATA_DIR / "adm_keywords.json"
    if adm_path.exists():
        adm_data = json.loads(adm_path.read_text())
        us = next(
            (
                e
                for e in adm_data
                if e["country-code"] == "US" and e["form-factor"] == "desktop"
            ),
            None,
        )
        if us:
            canonical.update(us["search-terms"].keys())

    canonical_words = {
        w
        for phrase in canonical
        for w in phrase.split()
        if len(w) > _AUTOCOMPLETE_MIN_PREFIX_LEN
    }
    print(f"  Canonical words: {len(canonical_words)}")

    # Load and filter vocab
    vocab: dict[str, int] = {}
    with open(vocab_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["word"]
            if word in canonical_words:
                vocab[word] = int(row["freq"])

    out = DATA_DIR / "word_freq.csv"
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "freq"])
        for word, freq in sorted(vocab.items(), key=lambda x: -x[1]):
            writer.writerow([word, freq])

    print(f"  Saved {len(vocab)} words to {out}")


def main() -> None:
    """Run data generation."""
    parser = argparse.ArgumentParser(description="Generate query normalization data files")
    parser.add_argument("--sports-api-key", help="SportsData.io API key")
    parser.add_argument("--vocab-csv", type=Path, help="BigQuery vocab export CSV (word, freq)")
    parser.add_argument("--merino-py", type=Path, help="Path to merino-py repo")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    if not any([args.sports_api_key, args.vocab_csv, args.merino_py]):
        parser.print_help()
        return

    if args.sports_api_key:
        print("Generating sports_teams.json...")
        generate_sports_teams(args.sports_api_key)

    if args.merino_py:
        print("Generating finance_tickers.json...")
        generate_finance_tickers(args.merino_py)

    if args.vocab_csv:
        print("Generating word_freq.csv...")
        generate_word_freq(args.vocab_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
