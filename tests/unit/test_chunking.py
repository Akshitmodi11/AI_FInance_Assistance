from src.rag.chunking import chunk_document, chunk_documents
from tests.conftest import make_doc


def test_short_document_is_one_chunk():
    doc = make_doc("Compound interest is interest on interest.", doc_id="ci:s0")
    chunks = chunk_document(doc, chunk_size=800, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "ci:s0:0"
    assert chunks[0].doc_id == "ci:s0"
    assert chunks[0].text == doc.text
    assert chunks[0].meta == doc.meta


def test_paragraphs_are_packed_and_size_is_bounded():
    paras = [f"Paragraph {i} " + "word " * 30 for i in range(10)]  # ~165 chars each
    doc = make_doc("\n\n".join(paras))
    chunks = chunk_document(doc, chunk_size=400, overlap=50)

    assert len(chunks) > 1
    assert all(len(c.text) <= 400 + 50 for c in chunks)
    assert [c.chunk_id for c in chunks] == [f"doc:s0:{i}" for i in range(len(chunks))]
    # Every paragraph's opening survives somewhere.
    joined = " ".join(c.text for c in chunks)
    for i in range(10):
        assert f"Paragraph {i}" in joined


def test_overlap_repeats_tail_of_previous_chunk():
    paras = [f"Para{i} " + "alpha beta gamma delta " * 8 for i in range(6)]
    chunks = chunk_document(make_doc("\n\n".join(paras)), chunk_size=300, overlap=60)
    assert len(chunks) >= 2
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        tail_words = prev.text.split()[-3:]
        assert " ".join(tail_words) in nxt.text


def test_giant_paragraph_is_split():
    sentence = "Diversification spreads risk across many investments. "
    doc = make_doc(sentence * 40)  # ~2,200 chars, no blank lines
    chunks = chunk_document(doc, chunk_size=500, overlap=0)
    assert len(chunks) >= 4
    assert all(len(c.text) <= 500 for c in chunks)


def test_no_empty_chunks():
    doc = make_doc("First.\n\n\n\n   \n\nSecond.")
    chunks = chunk_document(doc, chunk_size=10, overlap=0)
    assert chunks
    assert all(c.text.strip() for c in chunks)


def test_chunk_documents_flattens():
    docs = [make_doc("a", doc_id="x:s0"), make_doc("b", doc_id="x:s1")]
    assert [c.chunk_id for c in chunk_documents(docs, 800, 0)] == ["x:s0:0", "x:s1:0"]
