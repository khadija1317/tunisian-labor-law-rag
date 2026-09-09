import re
import json
import os

INPUT_PATH = "data/raw/code_du_travail_extracted.txt"
OUTPUT_PATH = "data/processed/code_du_travail_articles.json"

# Matches a real article header at the start of a line. Handles:
#   Article 100.-
#   Article 21-2 (Ajouté par...).-        <- numeric sub-article
#   Article 376 - ter (Ajouté par...).-   <- Latin-ordinal sub-article
#   Article 134 - 3(1) (Ajouté par...).-  <- sub-article with footnote marker
#   Article premier.-
HEADER_PATTERN = re.compile(
    r'^Article\s+(premier|\d+(?:\s*-\s*(?:\d+|bis|ter|quater|quinquies|sexies))?)'
    r'(?:\s*\(\d+\))?'
    r'\s*(\([^)]*\))?\s*\.?-',
    flags=re.IGNORECASE | re.MULTILINE
)

# Rare case: "Article N.- (note)." ordering where the note leaks into the body
# instead of being captured as the header's amendment-note group.
LEFTOVER_NOTE_PATTERN = re.compile(r'^\(([^)]*)\)\.\s*\.?-?\s*')


def split_articles(text: str) -> list[dict]:
    matches = list(HEADER_PATTERN.finditer(text))
    articles = []

    for i, match in enumerate(matches):
        article_id = match.group(1)
        if article_id.lower() == "premier":
            article_id = "1"
        else:
            article_id = re.sub(r'\s+', '', article_id)

        amendment_note = match.group(2)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        if amendment_note is None:
            leftover = LEFTOVER_NOTE_PATTERN.match(body)
            if leftover:
                amendment_note = leftover.group(1)
                body = body[leftover.end():].strip()
                amendment_note = f"({amendment_note})"  # normalize to match other notes

        articles.append({
            "article_id": article_id,
            "amendment_note": amendment_note.strip("()") if amendment_note else None,
            "text": body,
        })

    return articles


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    full_text = f.read()

full_text = re.sub(r'(\(cid:\d+\))+', '', full_text)

toc_match = re.search(r'TABLE DE MATIERES', full_text, flags=re.IGNORECASE)
if toc_match:
    full_text = full_text[:toc_match.start()]

premier_matches = list(re.finditer(r'^Article\s+premier\b', full_text, flags=re.IGNORECASE | re.MULTILINE))
if len(premier_matches) < 2:
    raise ValueError(f"Expected at least 2 'Article premier' occurrences, found {len(premier_matches)}")
code_start_pos = premier_matches[1].start()
code_text = full_text[code_start_pos:]

articles = split_articles(code_text)

print(f"Total articles extracted: {len(articles)}")

os.makedirs("data/processed", exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=2)

print(f"Saved {len(articles)} articles to {OUTPUT_PATH}")