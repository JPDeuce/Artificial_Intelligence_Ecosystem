"""Fetch a web page and save its main readable article text to Selected_Document.txt."""

import sys

import requests
from bs4 import BeautifulSoup

URL = "https://en.wikipedia.org/wiki/The_Legend_of_Zelda:_Ocarina_of_Time"
OUTPUT_FILE = "Selected_Document.txt"
TIMEOUT = 30

# Wikipedia returns HTTP 403 for the default python-requests User-Agent.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Layout blocks that are noise for a RAG corpus.
SKIP_CLASSES = (
    "navbox",
    "reflist",
    "references",
    "metadata",
    "hatnote",
    "infobox",
    "sidebar",
    "mw-editsection",
    "shortdescription",
    "thumb",
    "toc",
    "navigation-only",
)

CONTENT_TAGS = ["h2", "h3", "h4", "p", "li"]


def fetch_page(url):
    """Return the page HTML, or None if the request fails."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException as exc:
        print(f"Error: could not reach {url} -> {exc}")
        return None

    if response.status_code != 200:
        print(f"Error: request failed with HTTP status {response.status_code}.")
        return None

    return response.text


def _ancestor_classes(element):
    """Class strings of an element's ancestors, nearest first."""
    classes = []
    for parent in element.parents:
        if parent.name == "body":
            break
        classes.append(" ".join(parent.get("class") or []))
    return classes


def _is_noise(element):
    """True if the element or any ancestor is a known noise block."""
    own = " ".join(element.get("class") or [])
    if any(name in own for name in SKIP_CLASSES):
        return True
    return any(
        name in ancestor
        for ancestor in _ancestor_classes(element)
        for name in SKIP_CLASSES
    )


def _clean(text):
    """Collapse all whitespace runs into single spaces."""
    return " ".join(text.split())


def _key_facts(container):
    """Infobox rows (developer, release date, platforms...) as text."""
    facts = []
    table = container.select_one("table.infobox")
    if table is None:
        return facts

    for row in table.select("tr"):
        label, value = row.find("th"), row.find("td")
        if label is None or value is None:
            continue
        label_text = _clean(label.get_text(" ", strip=True))
        value_text = _clean(value.get_text(" ", strip=True))
        if label_text and value_text:
            facts.append(f"{label_text}: {value_text}")
    return facts


def extract_text(html):
    """Extract readable article text from page HTML."""
    soup = BeautifulSoup(html, "html.parser")

    container = soup.find("article")
    if container is None:
        container = soup.select_one("#mw-content-text .mw-parser-output")
    if container is None:
        container = soup.find("main")
    if container is None:
        print("Error: no article content container found in the page.")
        return ""

    # Drop the inline [1] [2] citation markers.
    for sup in container.select("sup.reference"):
        sup.decompose()

    blocks = []

    facts = _key_facts(container)
    if facts:
        blocks.append("Key facts")
        blocks.extend(facts)

    for element in container.find_all(CONTENT_TAGS):
        if _is_noise(element):
            continue
        text = _clean(element.get_text(" ", strip=True))
        if text:
            blocks.append(text)

    if not blocks:
        print("Error: the page was fetched but no article text was found.")
        return ""

    return "\n\n".join(blocks)


def fetch_and_extract(url):
    """Fetch, extract and save the page; return the extracted text on success."""
    print(f"Fetching {url}")
    html = fetch_page(url)
    if html is None:
        return None

    text = extract_text(html)
    if not text:
        return None

    with open(OUTPUT_FILE, "w", encoding="utf-8") as handle:
        handle.write(text)

    block_count = len(text.split("\n\n"))
    print(f"Saved {block_count} blocks ({len(text)} characters) to '{OUTPUT_FILE}'.")
    return text


def main():
    return 0 if fetch_and_extract(URL) else 1


if __name__ == "__main__":
    sys.exit(main())
