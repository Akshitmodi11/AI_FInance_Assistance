"""Profile the raw KB to inform chunking. Uses pypdf/bs4 directly (loaders are still stubs).

Run from the repo root:  .venv\\Scripts\\python notebooks/profile_corpus.py .
"""

import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT))
from src.rag.loaders.markdown import load_markdown  # noqa: E402

LIMIT = 800


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else 0


def report(name, units, label):
    lens = [len(u) for u in units if u.strip()]
    if not lens:
        print(f"{name}: no text")
        return
    over = sum(n > LIMIT for n in lens)
    paras = [p for u in units for p in re.split(r"\n\s*\n", u) if p.strip()]
    plens = [len(p) for p in paras]
    lines = [ln for u in units for ln in u.splitlines() if ln.strip()]
    print(
        f"{name:38} {label}s={len(lens):4}  median={int(st.median(lens)):5}  "
        f"p90={pct(lens, 90):5}  max={max(lens):6}  >{LIMIT}={over / len(lens):5.0%}  "
        f"total={sum(lens):8,}"
    )
    print(
        f"{'':38} paragraphs/{label}={len(paras) / len(lens):5.1f}"
        f"  para median={int(st.median(plens))}  para max={max(plens)}"
        f"  avg line len={int(st.mean(len(ln) for ln in lines))}"
    )


print(f"=== chunk_size limit = {LIMIT} chars ===\n")
print("--- Markdown (sections from the real loader) ---")
for p in sorted((ROOT / "data/kb/articles").glob("*.md")):
    report(p.name, [d.text for d in load_markdown(p)], "section")

print("\n--- PDF (raw pages, before header stripping) ---")
pdf_pages = {}
for p in sorted((ROOT / "data/kb/documents").glob("*.pdf")):
    pages = [pg.extract_text() or "" for pg in PdfReader(p).pages]
    pdf_pages[p.name] = pages
    report(p.name, pages, "page")

print("\n--- PDF repeated lines (on >=50% of pages) ---")
for name, pages in pdf_pages.items():
    c = Counter(ln.strip() for pg in pages for ln in set(pg.splitlines()) if ln.strip())
    rep = [(ln, n) for ln, n in c.most_common(8) if n / len(pages) >= 0.5]
    print(f"{name}: {len(rep)} repeated")
    for ln, n in rep:
        print(f"    {n:3}/{len(pages)}  {ln[:90]!r}")
    first_lines = Counter(pg.splitlines()[0][:40] for pg in pages if pg.strip())
    print(f"    most common first line: {first_lines.most_common(1)}")

print("\n--- HTML (naive: body minus script/style/nav/header/footer, split on <h2>) ---")
for p in sorted((ROOT / "data/kb/documents").glob("*.html")):
    soup = BeautifulSoup(p.read_text(encoding="utf-8"), "html.parser")
    for t in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]):
        t.decompose()
    root = soup.find("main") or soup.find("article") or soup.body
    tags = Counter(t.name for t in root.find_all(["h1", "h2", "h3", "h4", "strong", "p", "li"]))
    sections, cur = [], []
    for el in root.find_all(["h2", "p", "li"]):
        txt = " ".join(el.get_text(" ").split())
        if el.name == "h2":
            sections.append(" ".join(cur))
            cur = []
        elif txt:
            cur.append(txt)
    sections.append(" ".join(cur))
    report(p.name, sections, "section")
    h2 = [" ".join(h.get_text(" ").split()) for h in root.find_all("h2")]
    h3 = [" ".join(h.get_text(" ").split()) for h in root.find_all("h3")][:6]
    print(f"{'':38} tag counts={dict(tags)}")
    print(f"{'':38} h2={h2}")
    print(f"{'':38} h3 (first 6)={h3}")

print("\n--- Sample: p590a page 10 (raw) ---")
print(pdf_pages["p590a.pdf"][9][:1500])
