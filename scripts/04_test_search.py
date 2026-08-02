from __future__ import annotations

import argparse

from src.config import load_config
from src.search_engine import SearchEngine, format_search_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="a person riding a motorcycle")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    engine = SearchEngine(load_config())
    results = engine.search_by_text(args.query, top_k=args.top_k)
    print(format_search_results(results))


if __name__ == "__main__":
    main()
