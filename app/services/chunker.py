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
            overlap_seed = f"{current[-chunk_overlap:]}\n\n{para}"
            # If overlap + new paragraph still exceeds chunk_size, skip overlap
            current = overlap_seed if len(overlap_seed) <= chunk_size else para
        else:
            current = para

    if current:
        chunks.append(current)

    return chunks


def chunk_code(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 100
) -> list[tuple[str, int, int]]:
    """Line-aware chunker for source code.

    Packs whole lines into chunks of at most chunk_size characters, returning
    (chunk_text, start_line, end_line) tuples where line numbers are 1-indexed
    and inclusive.

    Differences from chunk_text:
    - Never splits a line mid-line. A line longer than chunk_size becomes its
      own chunk (single-line minified files index as one chunk per line).
    - No section/paragraph awareness — code's natural unit is the line.
    - Overlap is whole-line: we repeat trailing lines from the previous chunk
      until the total overlap reaches ~chunk_overlap chars.
    """
    if not text.strip():
        return []

    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[tuple[str, int, int]] = []
    i = 0
    while i < len(lines):
        current_size = 0
        j = i
        while j < len(lines):
            line = lines[j]
            addition = len(line) + (1 if j > i else 0)  # +1 for the joining newline
            if current_size > 0 and current_size + addition > chunk_size:
                break
            current_size += addition
            j += 1

        if j == i:
            # A single line exceeds chunk_size on its own — take it anyway to guarantee progress.
            j = i + 1

        chunk_text = "\n".join(lines[i:j])
        chunks.append((chunk_text, i + 1, j))

        if j >= len(lines):
            break

        # Compute next i — back up by whole lines until overlap_size reaches chunk_overlap,
        # but never repeat the whole chunk and never go backwards.
        next_i = j
        if chunk_overlap > 0:
            overlap_size = 0
            k = j
            while k > i + 1 and overlap_size < chunk_overlap:
                k -= 1
                overlap_size += len(lines[k]) + 1
            next_i = max(k, i + 1)
        i = next_i

    return chunks
