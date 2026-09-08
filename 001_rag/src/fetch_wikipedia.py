# ============================================================
# RAG Reproduction — Wikipedia Data Ingestion
#
# Author: Andrew Bieber <andrewbieber.work@gmail.com>
# Date: September 7, 2026
#
# File: fetch_wikipedia.py
#
# Description:
# Fetches the raw text from the Wikipedia articles listed in
# data/sources.json and stores each article as a separate .txt
# file in data/raw/.
#
# Delegation Note:
# This file was delegated to GPT-5.6 Sol because interacting with
# the MediaWiki API is implementation busy work and not part of
# the core learning objective of this RAG reproduction.
#
# I am intentionally implementing the chunking, embeddings,
# retrieval logic, and core RAG pipeline myself. Learning a
# one-off API just to download 10 Wikipedia articles would add
# little value when Sol can handle that plumbing in seconds.
#
# Paper:
# "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
# Lewis et al., 2020
# https://arxiv.org/abs/2005.11401
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
    "User-Agent": (
        "RAG-Reproduction/1.0 "
        "(andrewbieber.work@gmail.com) "
        "python-requests"
    )
}


def get_title_from_url(url):
    """
    Extract the Wikipedia article title from a full Wikipedia URL.

    Example:
        https://en.wikipedia.org/wiki/Infield_fly_rule

    Returns:
        Infield fly rule
    """

    path = urlparse(url).path
    title = path.split("/wiki/")[-1]

    return unquote(title).replace("_", " ")


def make_filename(title):
    """
    Convert an article title into a filesystem-safe filename.

    Example:
        "Infield fly rule" -> "infield_fly_rule.txt"
    """

    filename = title.lower().replace(" ", "_")
    filename = re.sub(r"[^a-z0-9_]", "", filename)

    return f"{filename}.txt"


def fetch_article(title):
    """
    Fetch the plain-text contents of a Wikipedia article using
    the MediaWiki API.
    """

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


def main():
    """
    Load the Wikipedia source URLs, fetch each article,
    and save the raw article text to data/raw/.
    """

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