from pathlib import Path

import pytest
import yaml

from src.core.config import Settings
from src.rag.loaders import load_corpus, load_file
from src.rag.loaders.docx import load_docx
from src.rag.loaders.html import load_html
from src.rag.loaders.markdown import load_markdown
from src.rag.loaders.pdf import clean_page_text, load_pdf, strip_repeated_lines
from tests.conftest import make_docx, make_pdf

MD = """---
title: Compound Interest
url: https://www.investor.gov/compound
source: SEC Investor.gov
category: basics
reviewed_date: 2026-10-17
level: beginner
---

# Compound Interest

Interest that earns interest.

## How it works

Your balance grows each period.

## Why time matters

Starting early helps a lot.
"""


# ---------- Markdown (implemented — the worked example) ----------


def test_markdown_splits_on_h2(tmp_path: Path):
    path = tmp_path / "compound-interest.md"
    path.write_text(MD, encoding="utf-8")
    docs = load_markdown(path)

    sections = [d.meta.section for d in docs]
    assert sections == ["Compound Interest", "How it works", "Why time matters"]
    assert [d.doc_id for d in docs] == [f"compound-interest:s{i}" for i in range(3)]
    assert docs[1].text == "Your balance grows each period."
    assert all(d.meta.file_type == "md" and d.meta.category == "basics" for d in docs)
    assert "# Compound Interest" not in docs[0].text


def test_markdown_citation_uses_section(tmp_path: Path):
    path = tmp_path / "ci.md"
    path.write_text(MD, encoding="utf-8")
    assert load_markdown(path)[1].meta.citation == "Compound Interest — How it works"


# ---------- PDF ----------

HEADER = "Publication 590-A (2025)"


def test_strip_repeated_lines_removes_headers_and_page_numbers():
    pages = [
        f"{HEADER}\nContributions to a traditional IRA.\nPage 1",
        f"{HEADER}\nRoth IRA income limits apply.\nPage 2",
        f"{HEADER}\nExcess contributions are taxed.\n3",
    ]
    cleaned = strip_repeated_lines(pages)
    assert len(cleaned) == 3
    for text in cleaned:
        assert HEADER not in text
        assert "Page" not in text
    assert "Roth IRA income limits apply." in cleaned[1]
    assert not cleaned[2].rstrip().endswith("3")


def test_strip_repeated_lines_keeps_content_on_short_docs():
    pages = ["Same line\nA", "Same line\nB"]
    assert all("Same line" in p for p in strip_repeated_lines(pages))


def test_strip_repeated_lines_removes_irs_proof_metadata():
    # Real IRS header lines (590-A): they change per page, so exact matching can't catch them.
    pages = [
        "Userid: CPM Schema: tipx Leadpct: 100% Pt. size: 10\n"
        "AH XSL/XML Fileid: … cation-969/2025/a/xml/cycle08/source (Init. & Date) _______\n"
        "IRA contributions are limited.",
        "Page 10 of 62  Fileid: … tion-590-a/2025/a/xml/cycle08/source 10:50 - 15-Jan-2026\n"
        "Spousal IRA limit applies.",
    ]
    cleaned = strip_repeated_lines(pages)  # only 2 pages: the patterns must work on their own
    assert cleaned[0].strip() == "IRA contributions are limited."
    assert cleaned[1].strip() == "Spousal IRA limit applies."


def test_clean_page_text_unwraps_and_dehyphenates():
    raw = (
        "If you invest in an annuity under an individual retire-\n"
        "ment annuity, no more than\n"
        "$7,000   can be contributed.\n"
        "\n"
        "More than one IRA.  The limit\n"
        "applies to the total."
    )
    assert clean_page_text(raw) == (
        "If you invest in an annuity under an individual retirement annuity, no more than "
        "$7,000 can be contributed.\n\n"
        "More than one IRA. The limit applies to the total."
    )


def test_clean_page_text_keeps_real_hyphens():
    # Not "letter-\nlowercase" → a real hyphenated term: keep the hyphen, drop the line break.
    assert (
        clean_page_text("File Form 1040-\nSR or Form\n8606.") == "File Form 1040-SR or Form 8606."
    )


def test_load_pdf_one_document_per_page(tmp_path: Path, manifest_meta):
    path = tmp_path / "p590a.pdf"
    path.write_bytes(
        make_pdf(
            [
                [HEADER, "Traditional IRA contributions may be deductible.", "Page 1"],
                [HEADER, "", "Page 2"],  # blank after stripping -> skipped
                [HEADER, "Roth IRA contributions are not deductible.", "Page 3"],
            ]
        )
    )
    docs = load_pdf(path, manifest_meta)

    assert [d.meta.page for d in docs] == [1, 3]
    assert [d.doc_id for d in docs] == ["p590a:p1", "p590a:p3"]
    assert "Roth IRA" in docs[1].text
    assert HEADER not in docs[0].text
    assert docs[0].meta.file_type == "pdf"
    assert docs[0].meta.tax_year == 2025
    assert docs[1].meta.citation == "IRS Publication 590-A, p. 3"
    assert docs[0].meta.section is None  # this PDF has no bookmarks


def test_load_pdf_sections_from_bookmarks(tmp_path: Path, manifest_meta):
    path = tmp_path / "p969.pdf"
    path.write_bytes(
        make_pdf(
            [
                ["Introduction text."],
                ["HSA overview."],
                ["Contribution limits are set yearly."],
                ["More about limits."],
            ],
            bookmarks=[
                ("Health Savings Accounts (HSAs)", 2, None),
                ("Contributions to an HSA", 3, "Health Savings Accounts (HSAs)"),
            ],
        )
    )
    docs = load_pdf(path, manifest_meta | {"title": "IRS Publication 969"})

    assert [d.meta.section for d in docs] == [
        None,  # before the first bookmark
        "Health Savings Accounts (HSAs)",
        "Contributions to an HSA",
        "Contributions to an HSA",  # no new bookmark on p. 4 → previous one still applies
    ]
    assert docs[2].meta.citation == "IRS Publication 969, p. 3"  # citation still uses the page
    assert docs[2].doc_id == "p969:p3"


def test_load_pdf_requires_manifest(tmp_path: Path):
    path = tmp_path / "x.pdf"
    path.write_bytes(make_pdf([["hello"]]))
    with pytest.raises(ValueError):
        load_pdf(path, None)


# ---------- DOCX ----------


def test_load_docx_splits_on_heading_styles(tmp_path: Path, manifest_meta):
    path = make_docx(
        tmp_path / "checklist.docx",
        [
            ("Normal", "A checklist for new investors."),
            ("Heading 1", "Before you invest"),
            ("Normal", "Build an emergency fund first."),
            ("Normal", ""),
            ("Normal", "Pay off high-interest debt."),
            ("Heading 2", "Opening an account"),
            ("Normal", "Compare brokerage fees."),
        ],
    )
    docs = load_docx(path, manifest_meta | {"title": "Investor Checklist"})

    assert [d.meta.section for d in docs] == [
        "Investor Checklist",
        "Before you invest",
        "Opening an account",
    ]
    assert docs[1].text == "Build an emergency fund first.\nPay off high-interest debt."
    assert [d.doc_id for d in docs] == [f"checklist:s{i}" for i in range(3)]
    assert docs[0].meta.file_type == "docx"


def test_load_docx_requires_manifest(tmp_path: Path):
    path = make_docx(tmp_path / "x.docx", [("Normal", "hi")])
    with pytest.raises(ValueError):
        load_docx(path, None)


# ---------- HTML ----------

HTML = """<html><head><title>Stocks</title><script>var x = 1;</script>
<style>.a{}</style></head><body>
<header><a href="/">Home</a> | <a href="/invest">Investing</a></header>
<nav>Menu: Glossary Calculators Alerts</nav>
<div id="cookie-banner">We use cookies. Accept?</div>
<main>
  <p>Stocks give you partial   ownership
     of a company.</p>
  <h2>Why companies issue stock</h2>
  <p>To raise money to grow.</p>
  <h2>Risks</h2>
  <p>Stock prices can fall.</p>
</main>
<footer>Contact us | Privacy policy</footer>
</body></html>"""


def test_load_html_extracts_main_content(tmp_path: Path, manifest_meta):
    path = tmp_path / "stocks.html"
    path.write_text(HTML, encoding="utf-8")
    docs = load_html(path, manifest_meta | {"title": "Stocks", "category": "investing"})

    assert [d.meta.section for d in docs] == ["Stocks", "Why companies issue stock", "Risks"]
    assert docs[0].text == "Stocks give you partial ownership of a company."
    everything = " ".join(d.text for d in docs)
    for noise in ["Menu", "cookies", "Privacy", "var x", "Home"]:
        assert noise not in everything
    assert docs[2].doc_id == "stocks:s2"
    assert docs[0].meta.file_type == "html"


HTML_H4 = """<html><body>
<header>Investor.gov</header>
<nav>Main navigation</nav>
<h1>Mutual Funds</h1>
<h2></h2>
<h4>What are mutual funds?</h4>
<p>A mutual fund pools money from many investors.</p>
<p>Each share is part ownership of the portfolio.</p>
<h4>Why do people buy mutual funds?</h4>
<ul><li>Professional management</li><li><p>Diversification</p></li><li>Liquidity</li></ul>
<footer>Privacy policy</footer>
</body></html>"""


def test_load_html_splits_on_h4_and_keeps_paragraphs(tmp_path: Path, manifest_meta):
    # Mirrors Investor.gov "Mutual Funds": no <main>, empty <h2>, questions in <h4>.
    path = tmp_path / "mutual-funds.html"
    path.write_text(HTML_H4, encoding="utf-8")
    docs = load_html(path, manifest_meta | {"title": "Mutual Funds", "category": "investing"})

    assert [d.meta.section for d in docs] == [
        "What are mutual funds?",
        "Why do people buy mutual funds?",
    ]
    assert docs[0].text == (
        "A mutual fund pools money from many investors.\n\n"
        "Each share is part ownership of the portfolio."
    )
    assert docs[1].text == "Professional management\n\nDiversification\n\nLiquidity"
    assert [d.doc_id for d in docs] == ["mutual-funds:s0", "mutual-funds:s1"]


def test_load_html_requires_manifest(tmp_path: Path):
    path = tmp_path / "x.html"
    path.write_text("<p>hi</p>", encoding="utf-8")
    with pytest.raises(ValueError):
        load_html(path, None)


# ---------- Registry ----------


def test_load_file_rejects_unknown_extension(tmp_path: Path):
    path = tmp_path / "notes.xlsx"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="No loader"):
        load_file(path)


def _settings_for(tmp_path: Path) -> Settings:
    return Settings(
        rag={
            "articles_dir": tmp_path / "articles",
            "documents_dir": tmp_path / "documents",
            "manifest_path": tmp_path / "manifest.yaml",
        }
    )


def test_load_corpus_requires_manifest_entry_for_documents(tmp_path: Path):
    (tmp_path / "articles").mkdir()
    (tmp_path / "articles" / "ci.md").write_text(MD, encoding="utf-8")
    (tmp_path / "documents").mkdir()
    (tmp_path / "documents" / "orphan.html").write_text("<p>x</p>", encoding="utf-8")

    with pytest.raises(ValueError, match="orphan.html"):
        load_corpus(_settings_for(tmp_path))


def test_load_corpus_markdown_only(tmp_path: Path):
    (tmp_path / "articles").mkdir()
    (tmp_path / "articles" / "ci.md").write_text(MD, encoding="utf-8")
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump({"documents": {}}), encoding="utf-8")

    docs = load_corpus(_settings_for(tmp_path))
    assert len(docs) == 3
