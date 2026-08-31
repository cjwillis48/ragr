"""Structure-aware chunking over Markdown-shaped text.

Ingestion normalises HTML and PDFs to Markdown, so headings, paragraphs and
list items survive as real block structure. Chunks are built from whole blocks
within a single heading section, falling back to sentence and then word
boundaries only when a block is larger than chunk_size on its own.
"""

import re
from dataclasses import dataclass, field

_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
_FENCE = re.compile(r"^(```|~~~)")
# Permalink glyphs that survive extraction and would end up in heading_path.
_PERMALINK_SUFFIX = re.compile(r"[\s\u00b6#\u00a7]+$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkResult:
    """One chunk plus the provenance needed to order and situate it."""

    text: str
    position: int
    heading_path: list[str] = field(default_factory=list)
    start_offset: int = 0
    end_offset: int = 0


@dataclass
class _Block:
    text: str
    start: int
    heading_path: tuple[str, ...]
    is_heading: bool = False


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[ChunkResult]:
    """Split text into chunks that respect heading, paragraph and sentence boundaries."""
    if not text.strip():
        return []

    blocks = _parse_blocks(text)
    sized = _enforce_max_size(blocks, chunk_size)
    return _merge_into_chunks(sized, chunk_size, chunk_overlap)


def _parse_blocks(text: str) -> list[_Block]:
    """Split into paragraph/heading blocks, tracking the heading path and source offset."""
    blocks: list[_Block] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    buffer_start = 0
    offset = 0

    def flush() -> None:
        nonlocal buffer
        if buffer:
            joined = "\n".join(buffer).strip()
            if joined:
                blocks.append(_Block(joined, buffer_start, tuple(t for _, t in stack)))
            buffer = []

    in_fence = False

    for line in text.splitlines(keepends=True):
        bare = line.strip()

        # Inside a fenced code block nothing is markup: a shell comment is not a
        # heading, and a blank line is not a paragraph break.
        if _FENCE.match(bare):
            in_fence = not in_fence
            if not buffer:
                buffer_start = offset
            buffer.append(line.rstrip("\n"))
            offset += len(line)
            continue
        if in_fence:
            if not buffer:
                buffer_start = offset
            buffer.append(line.rstrip("\n"))
            offset += len(line)
            continue

        heading = _HEADING.match(bare)
        if heading:
            flush()
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, _PERMALINK_SUFFIX.sub("", heading.group(2)).strip()))
            blocks.append(_Block(bare, offset, tuple(t for _, t in stack), is_heading=True))
        elif not bare:
            flush()
        else:
            if not buffer:
                buffer_start = offset
            buffer.append(line.rstrip("\n"))
        offset += len(line)

    flush()
    return blocks


def _last_break(window: str) -> int:
    """Index to cut at: last sentence end, else last word boundary, else the full window."""
    last = None
    for match in _SENTENCE_END.finditer(window):
        last = match
    if last and last.start() > 0:
        return last.start()
    space = window.rfind(" ")
    return space if space > 0 else len(window)


def _enforce_max_size(blocks: list[_Block], chunk_size: int) -> list[_Block]:
    """Break any block larger than chunk_size at sentence, then word, boundaries."""
    result: list[_Block] = []
    for block in blocks:
        if len(block.text) <= chunk_size:
            result.append(block)
            continue

        text, pos = block.text, 0
        while len(text) - pos > chunk_size:
            cut = _last_break(text[pos : pos + chunk_size])
            piece = text[pos : pos + cut].strip()
            if piece:
                result.append(_Block(piece, block.start + pos, block.heading_path))
            pos += cut
            while pos < len(text) and text[pos].isspace():
                pos += 1
        tail = text[pos:].strip()
        if tail:
            result.append(_Block(tail, block.start + pos, block.heading_path))
    return result


def _overlap_tail(text: str, budget: int) -> str:
    """Trailing slice of a chunk to seed the next one, snapped to a clean boundary."""
    if budget <= 0 or not text:
        return ""
    tail = text[-budget:] if budget < len(text) else text
    match = _SENTENCE_END.search(tail)
    if match:
        return tail[match.end() :].strip()
    space = tail.find(" ")
    return tail[space + 1 :].strip() if space != -1 else ""


def _merge_into_chunks(
    blocks: list[_Block], chunk_size: int, chunk_overlap: int
) -> list[ChunkResult]:
    """Pack blocks into chunks, starting fresh at each heading and overlapping otherwise."""
    chunks: list[ChunkResult] = []
    current: list[_Block] = []
    seed = ""

    def emit() -> None:
        nonlocal current, seed
        if not current:
            return
        body = "\n\n".join(b.text for b in current)
        text = f"{seed}\n\n{body}" if seed else body
        start = current[0].start
        # The path that situates the chunk's prose, not the first heading that
        # happens to lead it — a section header followed by a subsection header
        # should report the subsection.
        situating = next((b for b in current if not b.is_heading), current[-1])
        chunks.append(ChunkResult(
            text=text,
            position=len(chunks),
            heading_path=list(situating.heading_path),
            start_offset=start,
            end_offset=current[-1].start + len(current[-1].text),
        ))
        seed = ""
        current = []

    for block in blocks:
        # A heading always begins a new chunk; carrying prose across a section
        # boundary is what mixes unrelated context. Consecutive headings with no
        # prose between them accumulate rather than emitting an orphan chunk.
        if block.is_heading and current:
            if all(b.is_heading for b in current):
                current.append(block)
                continue
            emit()
            current = [block]
            continue

        candidate = "\n\n".join(b.text for b in current + [block])
        if len(seed) + len(candidate) + 2 <= chunk_size or not current:
            current.append(block)
            continue

        previous = f"{seed}\n\n" if seed else ""
        previous += "\n\n".join(b.text for b in current)
        emit()
        # Keep as much overlap as fits alongside the block that starts the next chunk.
        seed = _overlap_tail(previous, min(chunk_overlap, chunk_size - len(block.text) - 2))
        current = [block]

    emit()
    return chunks
