#!/usr/bin/env python3
"""
01_make_jsonl.py

Downloads Scryfall bulk data (default: all_cards), caches the raw JSON locally,
streams it to produce a deduplicated JSONL dataset of flavor texts.

Output format (JSONL):
{"text": "...", "oracle_id": "..."}
"""

import argparse
import hashlib
import json
from pathlib import Path

# Optional but strongly recommended for huge JSON arrays:
# pip install ijson
import ijson
import requests


def get_download_uri(session: requests.Session, bulk_type: str) -> str:
    r = session.get(f"https://api.scryfall.com/bulk-data/{bulk_type}", timeout=30)
    r.raise_for_status()
    return r.json()["download_uri"]


def download_to_file(session: requests.Session, url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    tmp.replace(path)


def ensure_bulk_file(path: Path, bulk_type: str, force: bool = False) -> Path:
    if path.exists() and not force:
        return path

    with requests.Session() as session:
        uri = get_download_uri(session, bulk_type)
        print(f"Downloading bulk '{bulk_type}' to {path} ...")
        download_to_file(session, uri, path)

    return path


def iter_cards_from_bulk_json(bulk_json_path: Path):
    # Scryfall bulk JSON is a top-level array; ijson "item" iterates array elements.
    with bulk_json_path.open("rb") as f:
        yield from ijson.items(f, "item")


def normalize_text(s: str) -> str:
    # Collapse whitespace; keep punctuation/case as-is.
    return " ".join(s.split())


def stable_text_key(text: str) -> str:
    # Hash normalized text to keep memory use down vs storing full strings in set.
    # Collisions are extremely unlikely for SHA-1 in this use case.
    n = normalize_text(text).encode("utf-8")
    return hashlib.sha1(n).hexdigest()


def write_flavor_jsonl(
    cards_iter,
    out_jsonl: Path,
    *,
    lang: str = "en",
    dedupe: str = "text",  # "text" | "oracle_id" | "pair"
    limit: int | None = None,
) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    seen = set()

    kept = 0
    total = 0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for card in cards_iter:
            total += 1

            if card.get("lang") != lang:
                continue

            ft = card.get("flavor_text")
            if not ft:
                continue

            oid = card.get("oracle_id")

            if dedupe == "text":
                k = stable_text_key(ft)
                if k in seen:
                    continue
                seen.add(k)

            elif dedupe == "oracle_id":
                if not oid or oid in seen:
                    continue
                seen.add(oid)

            elif dedupe == "pair":
                if not oid:
                    continue
                k = (oid, stable_text_key(ft))
                if k in seen:
                    continue
                seen.add(k)

            else:
                raise ValueError("dedupe must be one of: text, oracle_id, pair")

            rec = {"text": ft, "oracle_id": oid}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

            if limit is not None and kept >= limit:
                break

    print(f"Scanned: {total:,} cards | Wrote: {kept:,} examples -> {out_jsonl}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bulk-type",
        default="all_cards",
        help="Scryfall bulk type (all_cards, oracle_cards, ...)",
    )
    p.add_argument(
        "--bulk-cache",
        default="data/scryfall_bulk.json",
        help="Where to cache the raw bulk JSON",
    )
    p.add_argument(
        "--out-jsonl", default="data/flavor_texts.jsonl", help="Output JSONL path"
    )
    p.add_argument(
        "--dedupe",
        default="text",
        choices=["text", "oracle_id", "pair"],
        help="Dedup strategy",
    )
    p.add_argument(
        "--force-download", action="store_true", help="Re-download even if cache exists"
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit examples written (debug)",
    )
    args = p.parse_args()

    bulk_path = ensure_bulk_file(
        Path(args.bulk_cache), args.bulk_type, force=args.force_download
    )
    cards_iter = iter_cards_from_bulk_json(bulk_path)
    write_flavor_jsonl(
        cards_iter, Path(args.out_jsonl), dedupe=args.dedupe, limit=args.limit
    )


if __name__ == "__main__":
    main()
