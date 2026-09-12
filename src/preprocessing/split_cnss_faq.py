"""
src/preprocessing/split_cnss_faq.py

Parses the CNSS FAQ page into structured Q&A entries, mirroring the schema
of code_du_travail_articles.json so both corpora can be consumed uniformly
by the Day 5 retriever agents.

Input:  data/raw/cnss_faq.html   (saved page source of https://www.cnss.tn/faq)
Output: data/processed/cnss_faq_entries.json
"""

import json
from pathlib import Path
from bs4 import BeautifulSoup, Tag

RAW_HTML_PATH = Path("data/raw/cnss_faq.html")
OUTPUT_PATH = Path("data/processed/cnss_faq_entries.json")
SOURCE_URL = "https://www.cnss.tn/faq"

def flatten_answer(accordion_child: Tag) -> str:
    """Convert an accordion_child block (paragraphs, lists, tables) to plain text."""
    parts = []
    for child in accordion_child.find_all(["p", "ul", "table"], recursive=False):
        if child.name == "p":
            text = child.get_text(" ", strip=True)
            if text:
                parts.append(text)
        elif child.name == "ul":
            for li in child.find_all("li"):
                parts.append(f"- {li.get_text(' ', strip=True)}")
        elif child.name == "table":
            rows = child.find_all("tr")
            handled = False
            if len(rows) == 2:
                cells = rows[1].find_all("td")
                if len(cells) == 2:
                    left_items = [p.get_text(strip=True) for p in cells[0].find_all("p")]
                    right_items = [p.get_text(strip=True) for p in cells[1].find_all("p")]
                    if len(left_items) == len(right_items) and left_items:
                        for l, r in zip(left_items, right_items):
                            parts.append(f"{l}: {r}")
                        handled = True
            if not handled:
                for row in rows:
                    cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
                    cells = [c for c in cells if c]
                    if cells:
                        parts.append(" | ".join(cells))
    if not parts:
        text = accordion_child.get_text(" ", strip=True)
        if text:
            parts.append(text)
    return "\n".join(parts)
def parse_faq(html_path: Path) -> list[dict]:
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    accordion = soup.find(id="basic-accordian")
    if accordion is None:
        raise ValueError("Could not find #basic-accordian — page structure may have changed.")

    entries = []
    current_category = None
    pending_question = None
    pending_base_id = None
    seen_questions = {}
    counter = 0

    for el in accordion.find_all(["h2", "div"], recursive=False):
        if el.name == "h2":
            current_category = el.get_text(strip=True)
            continue

        classes = el.get("class") or []
        el_id = el.get("id") or ""

        if "accordion_headings" in classes:
            pending_question = el.get_text(strip=True)
            pending_base_id = el_id.replace("-header", "")

        elif el_id.endswith("-content"):
            if pending_question is None:
                continue
            base_id = el_id.replace("-content", "")
            if base_id != pending_base_id:
                pending_question = None
                continue

            child = el.find(class_="accordion_child")
            answer = flatten_answer(child) if child else el.get_text(" ", strip=True)

            counter += 1
            entry_id = f"cnss_faq_{counter:04d}"

            if pending_question in seen_questions:
                print(f"WARNING: duplicate question text (entries {seen_questions[pending_question]} "
                      f"and {entry_id}): {pending_question!r}")
            else:
                seen_questions[pending_question] = entry_id

            entries.append({
                "id": entry_id,
                "category": current_category,
                "question": pending_question,
                "answer": answer,
                "source_url": SOURCE_URL,
            })
            pending_question = None
            pending_base_id = None

    return entries


def main():
    entries = parse_faq(RAW_HTML_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Parsed {len(entries)} Q&A entries -> {OUTPUT_PATH}")
    by_cat = {}
    for e in entries:
        by_cat[e["category"]] = by_cat.get(e["category"], 0) + 1
    for cat, n in by_cat.items():
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()