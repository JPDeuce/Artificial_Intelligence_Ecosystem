"""Fetch a web page and save its main readable article text to Selected_Document.txt.

The goal is a clean corpus for the RAG app in rag_app.py: prose only, no navigation, no
reference lists, no citation markers, and no boilerplate that would dilute the
embeddings.

Pipeline:
    HTTP GET -> locate the article body -> drop citation markers -> walk the article in
    document order -> keep the paragraphs, headings and list items that are not noise
    -> write one blank-line-separated block per element.

Run it with:
    source .venv/bin/activate
    python text_extractor_web.py
"""

import os
import sys

import requests
from bs4 import BeautifulSoup

# The source document. The 1998 Ocarina of Time article was chosen over the 2026 remake
# because it is a much richer corpus: 73 prose blocks across 21 sections, against 16
# blocks for the remake. Both pages have the same structure, so the extractor code
# handles either - only this constant changes.
URL = "https://en.wikipedia.org/wiki/The_Legend_of_Zelda:_Ocarina_of_Time"

# Where the extracted text is written. The path is resolved relative to this file, so
# the result is always written next to the script and is always found by rag_app.py,
# which resolves its own input the same way. Running the script from any directory
# therefore produces the same file in the same place.
OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Selected_Document.txt"
)

# Seconds to wait for the HTTP response before giving up.
TIMEOUT = 30

# Wikipedia returns HTTP 403 to the default python-requests User-Agent ("python-requests/2.x").
# Sending a browser User-Agent is what makes the request succeed; the course example's
# plain requests.get(url) fails against the live site for exactly this reason.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Layout blocks that are noise in a RAG corpus. Each entry matches a class name that
# Wikipedia puts on the container of a block:
#   navbox / navigation-only - the "surrounding media" boxes at the foot of the article
#   reflist / references      - the reference lists ("Notes", "Citations", "Bibliography")
#   metadata / hatnote        - the italic notices under the title
#   infobox / sidebar / thumb - the info box, side panels and image captions
#   mw-editsection            - the little "[edit]" links
#   shortdescription         - the one-line summary under the title
#   toc                      - the table of contents
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

# The element types that carry the article's actual content, in the order they appear
# in the page. Headings are kept so the chunker can see the article's structure and so
# a chunk that starts with a section title carries its topic.
CONTENT_TAGS = ["h2", "h3", "h4", "p", "li"]


def fetch_page(url):
    """Return the page HTML, or None if the request fails.

    Any failure is reported and turned into None so the caller can exit with a non-zero
    status instead of writing an empty Selected_Document.txt over a good one.
    """
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
    """Class strings of an element's ancestors, nearest first.

    This is the fix for the main extraction problem. The 115 navigation-box list items
    and 24 reference list items in this article carry no class of their own - the class
    sits on an ancestor <table> or <ol>. Filtering on the element's own class therefore
    let all of them through. Walking up the ancestor chain finds the class that marks
    the block as noise.
    """
    classes = []
    for parent in element.parents:
        # Stop at <body>: everything below it is article content, and the page chrome
        # above it is irrelevant.
        if parent.name == "body":
            break
        classes.append(" ".join(parent.get("class") or []))
    return classes


def _is_noise(element):
    """True if the element or any ancestor is a known noise block."""
    # Check the element's own class first (cheapest and most specific).
    own = " ".join(element.get("class") or [])
    if any(name in own for name in SKIP_CLASSES):
        return True

    # Then check the ancestors, which is where Wikipedia actually puts the marker.
    return any(
        name in ancestor
        for ancestor in _ancestor_classes(element)
        for name in SKIP_CLASSES
    )


def _clean(text):
    """Collapse all whitespace runs into single spaces.

    Wikipedia's HTML is full of newlines and indentation inside paragraphs. Collapsing
    them keeps the chunks compact and stops the splitter from wasting overlap on
    whitespace.
    """
    return " ".join(text.split())


def _key_facts(container):
    """Infobox rows (developer, release date, platforms...) as text.

    The info box is a table, and a plain walk over paragraphs and headings would miss
    it entirely - yet it holds some of the most quotable facts in the article
    (developer, publisher, composer, platforms, release dates). It is emitted first, as
    a "Key facts" block, so those answers are retrievable.
    """
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
    """Extract readable article text from page HTML.

    Returns the article as a single string with one block per element, separated by
    blank lines, or an empty string if nothing usable was found.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Locate the article body. The containers are tried in order of specificity:
    # <article> is the semantic wrapper, the Wikipedia-specific selector is the
    # fallback, and <main> is the last resort. Only the first match is used, so the
    # body is never extracted twice.
    container = soup.find("article")
    if container is None:
        container = soup.select_one("#mw-content-text .mw-parser-output")
    if container is None:
        container = soup.find("main")
    if container is None:
        print("Error: no article content container found in the page.")
        return ""

    # Drop the inline [1] [2] citation markers before any text is read. They are
    # sup elements with the "reference" class, and leaving them in would litter the
    # corpus with meaningless numbers that the embedder would happily index.
    for sup in container.select("sup.reference"):
        sup.decompose()

    blocks = []

    # The info box goes first, as a titled block followed by one line per fact.
    facts = _key_facts(container)
    if facts:
        blocks.append("Key facts")
        blocks.extend(facts)

    # Then the article body. find_all returns elements in document order, so the
    # extracted text reads in the same order as the page.
    for element in container.find_all(CONTENT_TAGS):
        if _is_noise(element):
            continue
        text = _clean(element.get_text(" ", strip=True))
        if text:
            blocks.append(text)

    if not blocks:
        print("Error: the page was fetched but no article text was found.")
        return ""

    # Blank line between blocks: the chunker's first separator is "\n\n", so this is
    # what makes it prefer to break on a real paragraph boundary.
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

    # Report the shape of the corpus: the number of blocks is what the chunker will
    # work from, and a sudden drop from ~87 blocks means the page layout changed.
    block_count = len(text.split("\n\n"))
    print(f"Saved {block_count} blocks ({len(text)} characters) to '{OUTPUT_FILE}'.")
    return text


def main():
    """Extract the article; return 0 on success and 1 on failure."""
    return 0 if fetch_and_extract(URL) else 1


if __name__ == "__main__":
    sys.exit(main())
