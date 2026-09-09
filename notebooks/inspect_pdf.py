
"""
import pdfplumber

PDF_PATH = "data/raw/code de travail.pdf"

with pdfplumber.open(PDF_PATH) as pdf:
    print(f"Total pages: {len(pdf.pages)}")

    for page_num in [1,2,3,4,5, 50, 100]:
        page = pdf.pages[page_num - 1]  # 0-indexed
        text = page.extract_text()
        print(f"\n{'='*60}")
        print(f"PAGE {page_num}")
        print(f"{'='*60}")
        print(text[:800] if text else "[NO TEXT EXTRACTED]")

       

import pdfplumber

PDF_PATH = "data/raw/code de travail.pdf"

with pdfplumber.open(PDF_PATH) as pdf:
    page = pdf.pages[4]  # page 5, 0-indexed

    upright_chars = [c for c in page.chars if c["upright"]]
    rotated_chars = [c for c in page.chars if not c["upright"]]

    print(f"Total chars: {len(page.chars)}")
    print(f"Upright chars: {len(upright_chars)}")
    print(f"Rotated chars: {len(rotated_chars)}")

    print("\n--- Sample rotated chars ---")
    for c in rotated_chars[:15]:
        print(f"  {c['text']!r}  x0={c['x0']:.1f} top={c['top']:.1f}")

    print("\n--- Reconstructed text from upright chars only ---")
    from pdfplumber.utils import extract_text as et
    print(et(upright_chars)[:600])


import pdfplumber

PDF_PATH = "data/raw/code de travail.pdf"

with pdfplumber.open(PDF_PATH) as pdf:
    page = pdf.pages[4]  # page 5
    print(f"Page width: {page.width}, height: {page.height}")

    # Find where watermark chars sit on the x-axis
    x0s = sorted(c["x0"] for c in page.chars)
    print(f"x0 range across ALL chars: {x0s[0]:.1f} to {x0s[-1]:.1f}")

    # Try cropping out a strip on the right side (adjust after seeing width)
    # guessing watermark is in the last ~15% of page width
    crop_box = (0, 0, page.width * 0.85, page.height)
    cropped = page.crop(crop_box)
    text = cropped.extract_text()
    print("\n--- Cropped extraction ---")
    print(text[:800] if text else "[NO TEXT]")

    import pdfplumber
from collections import defaultdict

PDF_PATH = "data/raw/code de travail.pdf"

with pdfplumber.open(PDF_PATH) as pdf:
    page = pdf.pages[4]  # page 5

    # Group chars by rounded x0, collect the set of distinct letters seen at each x0
    by_x0 = defaultdict(list)
    for c in page.chars:
        by_x0[round(c["x0"])].append(c["text"])

    print("x0 -> count, distinct chars seen there")
    for x0 in sorted(by_x0):
        texts = by_x0[x0]
        distinct = sorted(set(texts))
        print(f"  x0={x0}: count={len(texts)}, distinct={distinct}")





import pdfplumber

PDF_PATH = "data/raw/code de travail.pdf"

with pdfplumber.open(PDF_PATH) as pdf:
    page = pdf.pages[4]  # page 5

    # Sort all chars by vertical position (top), then horizontal (x0)
    # so we see them in natural reading order, full detail
    chars_sorted = sorted(page.chars, key=lambda c: (round(c["top"]), c["x0"]))

    print(f"{'char':6} {'top':>6} {'x0':>6} {'size':>6}  font")
    for c in chars_sorted[:40]:
        print(f"{c['text']!r:6} {c['top']:6.1f} {c['x0']:6.1f} {c['size']:6.1f}  {c['fontname']}")






import pdfplumber
from collections import Counter

PDF_PATH = "data/raw/code de travail.pdf"

with pdfplumber.open(PDF_PATH) as pdf:
    page = pdf.pages[4]  # page 5

    font_counts = Counter((c["fontname"], round(c["size"], 1)) for c in page.chars)
    print("All (font, size) combos on this page:")
    for combo, count in font_counts.most_common():
        print(f"  {combo}: {count} chars")

    # Now show a sample of chars using each non-dominant font
    dominant_font = font_counts.most_common(1)[0][0]
    print(f"\nDominant font/size (body text, presumably): {dominant_font}")

    print("\n--- Sample chars NOT using the dominant font ---")
    other_chars = [c for c in page.chars if (c["fontname"], round(c["size"], 1)) != dominant_font]
    for c in other_chars[:30]:
        print(f"  {c['text']!r}  top={c['top']:.1f} x0={c['x0']:.1f} font={c['fontname']} size={c['size']:.1f}")
    print(f"\nTotal non-dominant-font chars: {len(other_chars)}")



    """



import pdfplumber

PDF_PATH = "data/raw/code de travail.pdf"

def clean_page_text(page):
    chars = [c for c in page.chars if c["fontname"] != "Helvetica-Oblique"]
    from pdfplumber.utils import extract_text
    return extract_text(chars)

with pdfplumber.open(PDF_PATH) as pdf:
    for page_num in [5, 50, 100]:
        page = pdf.pages[page_num - 1]
        text = clean_page_text(page)
        print(f"\n{'='*60}")
        print(f"PAGE {page_num}")
        print(f"{'='*60}")
        print(text[:800] if text else "[NO TEXT]")