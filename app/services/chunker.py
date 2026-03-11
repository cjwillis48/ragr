import re


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[str]:
    """Split text into chunks respecting section boundaries, then paragraph boundaries."""
    if not text.strip():
        return []

    # Split on section dividers (--- on its own line) first, keeping sections intact
    sections = _split_sections(text)

    chunks: list[str] = []
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            # Section too large — sub-chunk by paragraphs
            paragraphs = _split_paragraphs(section)
            paragraphs = _force_split_long_paragraphs(paragraphs, chunk_size)
            chunks.extend(_merge_into_chunks(paragraphs, chunk_size, chunk_overlap))

    return chunks


def _split_sections(text: str) -> list[str]:
    """Split on --- dividers (horizontal rules). Falls back to treating the whole text as one section."""
    parts = re.split(r"\n---+\n", text)
    return [p.strip() for p in parts if p.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Split on double newlines, discard empties."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _force_split_long_paragraphs(paragraphs: list[str], chunk_size: int) -> list[str]:
    """Break any paragraph that exceeds chunk_size at sentence boundaries."""
    result = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            result.append(para)
            continue

        # Split oversized paragraph by sentences, falling back to hard cut
        remaining = para
        while len(remaining) > chunk_size:
            split_at = remaining.rfind(". ", 0, chunk_size)
            if split_at == -1:
                split_at = chunk_size
            else:
                split_at += 1  # Include the period
            result.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()

        if remaining:
            result.append(remaining)

    return result


def _merge_into_chunks(
    paragraphs: list[str], chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Merge paragraphs into chunks up to chunk_size, with overlap between chunks."""
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        # Adding this paragraph would exceed the limit — finalize the current chunk
        if current:
            chunks.append(current)

        # Start the next chunk with overlap from the tail of the previous
        if 0 < chunk_overlap < len(current):
            current = f"{current[-chunk_overlap:]}\n\n{para}"
        else:
            current = para

    if current:
        chunks.append(current)

    return chunks
