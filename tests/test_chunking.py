from code_index.chunking import chunk_text
from code_index.text_detect import is_text_bytes


def test_chunk_text_with_overlap():
    text = "a\nb\nc\nd\n"
    chunks = chunk_text(text, chunk_lines=2, overlap_lines=1)
    assert [(c.line_start, c.line_end) for c in chunks] == [(1, 2), (2, 3), (3, 4)]
    assert chunks[0].text == "a\nb\n"


def test_is_text_bytes():
    assert is_text_bytes(b"hello\nworld\n")
    assert not is_text_bytes(b"\x00\x01\x02")
