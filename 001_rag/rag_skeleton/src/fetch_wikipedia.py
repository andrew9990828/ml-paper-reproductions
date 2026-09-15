# ============================================================
# RAG Reproduction — Wikipedia Data Ingestion
#
# This file is intentionally COMPLETE.
#
# Fetching Wikipedia through the MediaWiki API is plumbing rather
# than the core learning objective of this reproduction.
# ============================================================

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCES_PATH = PROJECT_ROOT / "data" / "sources.json"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "RAG-Reproduction/1.0 python-requests"
}


def get_title_from_url(url: str) -> str:
    path = urlparse(url).path
    title = path.split("/wiki/")[-1]
    return unquote(title).replace("_", " ")


def make_filename(title: str) -> str:
    filename = title.lower().replace(" ", "_")
    filename = re.sub(r"[^a-z0-9_]", "", filename)
    return f"{filename}.txt"


def fetch_article(title: str) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "redirects": True,
        "titles": title,
        "format": "json",
        "formatversion": 2,
    }

    response = requests.get(
        WIKIPEDIA_API,
        params=params,
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    page = data["query"]["pages"][0]

    if page.get("missing"):
        raise ValueError(f"Wikipedia page not found: {title}")

    return page["extract"]


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(SOURCES_PATH, "r", encoding="utf-8") as file:
        sources = json.load(file)["sources"]

    for url in sources:
        title = get_title_from_url(url)
        print(f"Fetching: {title}")

        article_text = fetch_article(title)
        output_path = RAW_DATA_DIR / make_filename(title)

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(article_text)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
