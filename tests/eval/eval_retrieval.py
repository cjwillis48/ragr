"""Retrieval eval harness: score a model's retrieval against a golden query set.

Establishes the baseline that later chunking changes are measured against.

A golden is either single-turn ({"query": ...}) or multi-turn ({"turns": [...]}),
where turns are the user's messages in order and the last one is what's asked.
Retrieval runs on the last turn with the previous turn as embedding context,
mirroring the /chat endpoint; --no-context retrieves on the bare last turn
(pre-0.12.0 behaviour) for before/after comparison.

A golden marked {"expect_absent": true} asserts the corpus cannot answer it, so
correct behaviour is surfacing nothing above rerank_threshold. These measure the
precision floor: recall says what we found, abstention says what we declined to
invent.

    uv run python tests/eval/eval_retrieval.py --model hades-bot
    uv run python tests/eval/eval_retrieval.py --model hades-bot --json runs/before.json

Requires DATABASE_URL and VOYAGE_API_KEY in the environment (or a .env file).
"""

import argparse
import asyncio
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sqlalchemy import select, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.database import _init_engine
from app import database
from app.models.rag_model import RagModel
try:  # renamed in 0.14.0; the old name is what pre-0.14 checkouts export
    from app.services.retrieval import retrieve_chunks
except ImportError:
    from app.services.retrieval import retrieve_with_threshold as retrieve_chunks

GOLDENS_DIR = Path(__file__).resolve().parent / "goldens"

# A chunk ending in sentence-terminating punctuation was cut at a real boundary.
_ENDS_SENTENCE = re.compile(r"[.!?][\"')\]]?$")

# Below this mean fragment length, a chunk is mostly nav links / table cells
# rather than prose — the signature of text-node flattening in html.py.
SHREDDED_FRAGMENT_CHARS = 40


@dataclass
class QueryResult:
    query: str
    hit_rank: int | None
    retrieved: int
    top_source: str = ""
    is_absent: bool = False

    @property
    def hit(self) -> bool:
        return self.hit_rank is not None

    @property
    def abstained(self) -> bool:
        """An unanswerable query that correctly surfaced nothing above the floor."""
        return self.is_absent and self.retrieved == 0

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.hit_rank if self.hit_rank else 0.0


@dataclass
class EvalReport:
    model_slug: str
    k: int
    queries: int = 0
    recall_at_k: float = 0.0
    mrr_at_k: float = 0.0
    negatives: int = 0
    abstention_rate: float = 0.0
    mean_chunks_when_absent: float = 0.0
    misses: list[str] = field(default_factory=list)
    leaks: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    corpus: dict = field(default_factory=dict)
    per_query: list[dict] = field(default_factory=list)


def _matches(chunk, golden: dict) -> bool:
    """A chunk satisfies a golden when every stated expectation holds."""
    want_source = golden.get("expect_source")
    want_substring = golden.get("expect_substring")
    if not want_source and not want_substring:
        raise ValueError(f"golden needs expect_source or expect_substring: {golden}")

    if want_source:
        haystack = f"{chunk.source_identifier or ''} {chunk.source_url or ''}"
        if want_source.lower() not in haystack.lower():
            return False
    if want_substring:
        # Whitespace-insensitive: shredded chunks are full of spurious newlines.
        norm = " ".join((chunk.content or "").split()).lower()
        if " ".join(want_substring.split()).lower() not in norm:
            return False
    return True


async def _corpus_health(session, model_id: int) -> dict:
    row = (await session.execute(text(r"""
        SELECT count(*)                                          AS chunks,
               count(DISTINCT source_identifier)                 AS sources,
               round(avg(length(content)))                       AS avg_chars,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY length(content))::int AS p50_chars,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY length(content))::int AS p95_chars,
               max(length(content))                              AS max_chars,
               count(*) FILTER (
                   WHERE length(content)::float
                       / (1 + (length(content) - length(replace(content, E'\n\n', ''))) / 2.0)
                     < :shred
               )                                                 AS shredded
        FROM content_chunks WHERE model_id = :mid
    """), {"mid": model_id, "shred": SHREDDED_FRAGMENT_CHARS})).mappings().one()

    # Sentence-boundary rate in Python so the regex matches the chunker's own notion.
    contents = (await session.execute(text(
        "SELECT content FROM content_chunks WHERE model_id = :mid"
    ), {"mid": model_id})).scalars().all()

    total = len(contents) or 1
    clean = sum(1 for c in contents if _ENDS_SENTENCE.search((c or "").rstrip()))

    return {
        "chunks": row["chunks"],
        "sources": row["sources"],
        "avg_chars": int(row["avg_chars"] or 0),
        "p50_chars": row["p50_chars"],
        "p95_chars": row["p95_chars"],
        "max_chars": row["max_chars"],
        "pct_ends_sentence": round(100.0 * clean / total, 1),
        "pct_shredded": round(100.0 * (row["shredded"] or 0) / total, 1),
    }


async def run(slug: str, goldens_path: Path | None, k: int | None, no_context: bool = False) -> EvalReport:
    _init_engine()
    async with database.async_session() as session:
        model = (await session.execute(
            select(RagModel).where(RagModel.slug == slug, RagModel.deleted_at.is_(None))
        )).scalar_one_or_none()
        if model is None:
            sys.exit(f"no active model with slug {slug!r}")

        top_k = k or model.top_k
        report = EvalReport(model_slug=slug, k=top_k)
        report.corpus = await _corpus_health(session, model.id)

        path = goldens_path or GOLDENS_DIR / f"{slug}.json"
        if not path.exists():
            print(f"note: no golden set at {path} — reporting corpus health only\n")
            return report

        goldens = json.loads(path.read_text())
        results: list[QueryResult] = []
        ctx_chars = ctx_junk = ctx_frags = 0
        for golden in goldens:
            is_absent = bool(golden.get("expect_absent"))
            turns = golden.get("turns") or [golden["query"]]
            query = turns[-1]
            context = turns[-2] if len(turns) > 1 and not no_context else None
            # k applies inside retrieval, before neighbour expansion — slicing the
            # result here would discard neighbours (see app/api/retrieve.py).
            retrieval = await retrieve_chunks(session, model, query, limit=k, context=context)
            chunks = retrieval.chunks
            # Quality of what actually reaches the generator: rank alone can look
            # perfect while the chunk itself is navigation debris. Absent-answer
            # queries are excluded — ideally they contribute no context at all.
            if not is_absent:
                for chunk in chunks:
                    frags = [f for f in (chunk.content or "").split("\n\n") if f.strip()]
                    ctx_chars += len(chunk.content or "")
                    ctx_frags += len(frags)
                    ctx_junk += sum(len(f) for f in frags if len(f) < SHREDDED_FRAGMENT_CHARS)
            rank = None if is_absent else next(
                (i for i, c in enumerate(chunks, 1) if _matches(c, golden)), None
            )
            results.append(QueryResult(
                query=query,
                hit_rank=rank,
                retrieved=len(chunks),
                top_source=chunks[0].source_identifier if chunks else "",
                is_absent=is_absent,
            ))

        positives = [r for r in results if not r.is_absent]
        negatives = [r for r in results if r.is_absent]

        report.queries = len(positives)
        if positives:
            report.recall_at_k = round(sum(r.hit for r in positives) / len(positives), 3)
            report.mrr_at_k = round(sum(r.reciprocal_rank for r in positives) / len(positives), 3)
        report.negatives = len(negatives)
        if negatives:
            report.abstention_rate = round(sum(r.abstained for r in negatives) / len(negatives), 3)
            report.mean_chunks_when_absent = round(
                sum(r.retrieved for r in negatives) / len(negatives), 2
            )
        report.misses = [
            f"{r.query} [{g.get('expect_substring') or g.get('expect_source')}]"
            for r, g in zip(results, goldens) if not r.is_absent and not r.hit
        ]
        report.leaks = [
            f"{r.query} → {r.retrieved} chunks, top: {r.top_source}"
            for r in negatives if not r.abstained
        ]
        if ctx_chars:
            report.context = {
                "chars": ctx_chars,
                "junk_chars": ctx_junk,
                "pct_junk": round(100.0 * ctx_junk / ctx_chars, 1),
                "mean_fragment_chars": round(ctx_chars / max(ctx_frags, 1), 1),
            }
        report.per_query = [asdict(r) for r in results]
        return report


def _print(report: EvalReport) -> None:
    c = report.corpus
    print(f"\n=== corpus health: {report.model_slug} ===")
    print(f"  chunks / sources      {c['chunks']} / {c['sources']}")
    print(f"  chars avg/p50/p95/max {c['avg_chars']} / {c['p50_chars']} / {c['p95_chars']} / {c['max_chars']}")
    print(f"  ends at sentence      {c['pct_ends_sentence']}%")
    print(f"  shredded (nav/table)  {c['pct_shredded']}%")

    if not report.queries and not report.negatives:
        return
    print(f"\n=== retrieval @{report.k} over {report.queries} queries ===")
    print(f"  recall@{report.k}  {report.recall_at_k}")
    print(f"  MRR@{report.k}     {report.mrr_at_k}")
    if report.negatives:
        print(f"\n=== precision floor over {report.negatives} unanswerable queries ===")
        print(f"  abstained           {report.abstention_rate}")
        print(f"  mean chunks kept    {report.mean_chunks_when_absent}")
    if report.context:
        x = report.context
        print(f"\n  retrieved context   {x['chars']} chars, {x['pct_junk']}% junk, "
              f"mean fragment {x['mean_fragment_chars']} chars")
    if report.misses:
        print(f"\n  missed ({len(report.misses)}):")
        for q in report.misses:
            print(f"    - {q}")
    if report.leaks:
        print(f"\n  leaked ({len(report.leaks)}):")
        for q in report.leaks:
            print(f"    - {q}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="model slug")
    ap.add_argument("--goldens", type=Path, help=f"default: {GOLDENS_DIR}/<slug>.json")
    ap.add_argument("--k", type=int, help="cutoff; defaults to the model's top_k")
    ap.add_argument("--json", type=Path, help="also write the report here for before/after diffing")
    ap.add_argument("--no-context", action="store_true",
                    help="retrieve on the bare final turn, ignoring prior turns (pre-0.12.0 behaviour)")
    args = ap.parse_args()

    report = asyncio.run(run(args.model, args.goldens, args.k, no_context=args.no_context))
    _print(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(asdict(report), indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
