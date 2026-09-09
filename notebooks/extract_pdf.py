import pdfplumber

PDF_PATH = "data/raw/code de travail.pdf"
OUTPUT_PATH = "data/raw/code_du_travail_extracted.txt"

def clean_page_text(page):
    chars = [c for c in page.chars if c["fontname"] != "Helvetica-Oblique"]
    from pdfplumber.utils import extract_text
    return extract_text(chars)

all_text = []
with pdfplumber.open(PDF_PATH) as pdf:
    print(f"Extracting {len(pdf.pages)} pages...")
    for i, page in enumerate(pdf.pages, start=1):
        text = clean_page_text(page)
        all_text.append(text if text else "")
        if i % 25 == 0:
            print(f"  ...page {i} done")

full_text = "\n".join(all_text)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"\nDone. Total characters: {len(full_text)}")
print(f"Saved to {OUTPUT_PATH}")



import re

# --- Anomaly scan ---
print("\n--- Anomaly scan ---")

# 1. Pages with suspiciously little text (possible scan/image page or extraction failure)
short_pages = [(i+1, len(t)) for i, t in enumerate(all_text) if len(t) < 200]
print(f"\nPages with <200 chars extracted: {len(short_pages)}")
for pnum, length in short_pages[:20]:
    print(f"  Page {pnum}: {length} chars")

# 2. Find all "Article N" occurrences in the full text, check numbering
article_matches = re.findall(r'Article\s+(\d+|premier)\b', full_text, flags=re.IGNORECASE)
print(f"\nTotal 'Article N' matches found: {len(article_matches)}")

# Convert to ints where possible, check for gaps/duplicates
nums = []
for m in article_matches:
    if m.lower() == "premier":
        nums.append(1)
    else:
        nums.append(int(m))

print(f"First 10 article numbers found: {nums[:10]}")
print(f"Last 10 article numbers found: {nums[-10:]}")
print(f"Max article number found: {max(nums)}")

# Check for gaps in sequence (rough check, duplicates from headers/refs expected)
unique_sorted = sorted(set(nums))
gaps = [unique_sorted[i+1] - unique_sorted[i] for i in range(len(unique_sorted)-1)]
big_gaps = [(unique_sorted[i], unique_sorted[i+1]) for i in range(len(unique_sorted)-1) if gaps[i] > 1]
print(f"\nGaps >1 between consecutive unique article numbers (first 15):")
for a, b in big_gaps[:15]:
    print(f"  {a} -> {b}  (gap of {b-a})")



# Find context around suspiciously high article numbers
print("\n--- Context around 'Article' matches with number > 500 ---")
for m in re.finditer(r'.{40}Article\s+(\d+)\b.{60}', full_text):
    num = int(m.group(1))
    if num > 500:
        print(f"  ...{m.group(0)}...")
        print()

# Check: how many "Article N" matches are at the start of a line
# vs buried mid-sentence (a proxy for "real header" vs "inline reference")
lines = full_text.split("\n")
header_like = 0
inline_like = 0
for line in lines:
    stripped = line.strip()
    if re.match(r'^Article\s+(\d+|premier)\b', stripped, flags=re.IGNORECASE):
        header_like += 1
    elif re.search(r'\barticle\s+\d+\b', line, flags=re.IGNORECASE):
        inline_like += 1

print(f"\n'Article N' at start of line (likely real headers): {header_like}")
print(f"'article N' buried mid-line (likely inline references): {inline_like}")


# Fix: use string slicing (index-based) instead of dotall-limited regex
print("\n--- Context around 'Article N' where N > 500 (index-based) ---")
for m in re.finditer(r'Article\s+(\d+)\b', full_text):
    num = int(m.group(1))
    if num > 500:
        start = max(0, m.start() - 80)
        end = min(len(full_text), m.end() + 80)
        snippet = full_text[start:end].replace("\n", " | ")
        print(f"  [{snippet}]")
        print()

# Check: among the HEADER-LINE matches specifically (start of line),
# are any numbers suspiciously high (>500)?
print("\n--- Header-line matches with N > 500 ---")
for line in lines:
    stripped = line.strip()
    m = re.match(r'^Article\s+(\d+)\b', stripped, flags=re.IGNORECASE)
    if m and int(m.group(1)) > 500:
        print(f"  {stripped[:100]}")

print("\n--- SANITY CHECK ---")
high_matches = [m for m in re.finditer(r'Article\s+(\d+)\b', full_text) if int(m.group(1)) > 500]
print(f"Number of 'Article N' matches with N > 500: {len(high_matches)}")

for m in high_matches[:10]:
    num = int(m.group(1))
    start = max(0, m.start() - 80)
    end = min(len(full_text), m.end() + 80)
    snippet = full_text[start:end].replace("\n", " | ")
    print(f"  N={num}: [{snippet}]")



print("\n--- Confirming with case-sensitive, start-of-line only ---")
header_pattern = re.compile(r'^Article\s+(\d+|premier)\b', flags=re.MULTILINE)
headers = header_pattern.findall(full_text)
print(f"Total header-line matches: {len(headers)}")

nums = [1 if h.lower() == "premier" else int(h) for h in headers]
print(f"Max: {max(nums)}, Min: {min(nums)}")
print(f"First 10: {nums[:10]}")
print(f"Last 10: {nums[-10:]}")

# duplicates (expected: bis/ter articles reuse a base number, e.g. "6 bis")
from collections import Counter
dupes = {n: c for n, c in Counter(nums).items() if c > 1}
print(f"\nNumbers appearing more than once: {len(dupes)}")
print(dict(list(dupes.items())[:10]))

print("\n--- All lines starting with 'Article 1' or 'Article premier' ---")
for line in lines:
    stripped = line.strip()
    if re.match(r'^Article\s+(1|premier)\b', stripped, flags=re.IGNORECASE):
        print(f"  {stripped[:90]}")

print("\n--- All lines starting with 'Article 21' ---")
for line in lines:
    stripped = line.strip()
    if re.match(r'^Article\s+21\b', stripped, flags=re.IGNORECASE):
        print(f"  {stripped[:90]}")


# Find character position of each "Article premier" occurrence
print("\n--- Locating both 'Article premier' occurrences ---")
for m in re.finditer(r'^Article\s+premier\b.{0,60}', full_text, flags=re.IGNORECASE | re.MULTILINE):
    print(f"  Position {m.start()}: {m.group(0)!r}")

# Roughly estimate which page each position falls on
# (average chars per page, rough guide only)
avg_chars_per_page = len(full_text) / 150
for m in re.finditer(r'^Article\s+premier\b', full_text, flags=re.IGNORECASE | re.MULTILINE):
    approx_page = m.start() / avg_chars_per_page
    print(f"  Position {m.start()} -> approx page {approx_page:.1f}")