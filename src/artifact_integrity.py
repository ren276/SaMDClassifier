"""
Artifact integrity: every file an endpoint loads is listed in ARTIFACT_MANIFEST.json with a
digest, and the app refuses to start when any listed artifact is missing or does not match.

Why this exists: a `model_version` string is a claim, not evidence. Binding it to the bytes that
were actually served is what lets a later incident review say which model answered.

Stdlib only, and imported before anything loads a model, so a mismatch fails the import (and the
container's healthcheck) rather than serving an unbound model.

Two entry kinds:
  file_sha256                 sha256 of the file's bytes.
  chroma_content_digest_v1    sha256 over the retrieval-defining ROWS of chroma.sqlite3.

The Chroma store cannot be byte-hashed. Opening it normally rewrites chroma.sqlite3, and the
HNSW files data_level0.bin and length.bin, without changing what a query returns (measured: a
copy of the store, opened and queried once, differed in all of these). A byte hash would pass on
first start and fail on the first container restart. The vectors' source of truth is the
`embeddings_queue` table, so the content digest covers that and the ids, documents and metadata
around it. header.bin and link_lists.bin did not change on open and are byte-hashed.
Residual: data_level0.bin and length.bin (the HNSW index derived from those rows) are not
covered.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "ARTIFACT_MANIFEST.json")
MANIFEST_SCHEMA = 1

KIND_FILE = "file_sha256"
KIND_CHROMA = "chroma_content_digest_v1"

CHROMA_SQLITE = "src/rag/vector_store/chroma.sqlite3"


class ArtifactIntegrityError(RuntimeError):
    """A listed artifact is missing or does not match the manifest."""


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# Fixed column lists and ORDER BY keys: the digest must not depend on row insertion order or on
# internal integer ids that a rewrite could renumber. Each statement selects only what defines
# retrieval: which collections exist, which ids they hold, the stored vectors, the documents and
# the metadata.
_CHROMA_QUERIES = (
    ("collections", "SELECT name, dimension FROM collections ORDER BY name"),
    (
        "embeddings",
        "SELECT c.name, e.embedding_id FROM embeddings e "
        "JOIN segments s ON s.id = e.segment_id JOIN collections c ON c.id = s.collection "
        "ORDER BY c.name, e.embedding_id",
    ),
    (
        "embeddings_queue",
        "SELECT id, operation, encoding, vector, metadata FROM embeddings_queue "
        "ORDER BY id, operation, seq_id",
    ),
    (
        "embedding_metadata",
        "SELECT c.name, e.embedding_id, m.key, m.string_value, m.int_value, m.float_value, "
        "m.bool_value FROM embedding_metadata m JOIN embeddings e ON e.id = m.id "
        "JOIN segments s ON s.id = e.segment_id JOIN collections c ON c.id = s.collection "
        "ORDER BY c.name, e.embedding_id, m.key",
    ),
)


def chroma_content_digest(sqlite_path: str) -> str:
    """sha256 over the retrieval-defining rows, read through a read-only URI connection so
    computing it can never modify the store, and before Chroma has opened it."""
    uri = "file:" + sqlite_path.replace("?", "%3f").replace("#", "%23") + "?mode=ro"
    digest = hashlib.sha256()
    con = sqlite3.connect(uri, uri=True)
    try:
        for name, sql in _CHROMA_QUERIES:
            digest.update(f"[{name}]\n".encode())
            for row in con.execute(sql):
                digest.update(
                    json.dumps(
                        [v.hex() if isinstance(v, bytes) else v for v in row],
                        separators=(",", ":"),
                    ).encode()
                    + b"\n"
                )
    finally:
        con.close()
    return digest.hexdigest()


def compute_entry(root: str, rel_path: str, kind: str) -> str:
    path = os.path.join(root, rel_path)
    if kind == KIND_FILE:
        return sha256_file(path)
    if kind == KIND_CHROMA:
        return chroma_content_digest(path)
    raise ArtifactIntegrityError(f"Unknown manifest entry kind {kind!r} for {rel_path}")


def verify_manifest(root: str = PROJECT_ROOT, manifest_path: str = MANIFEST_PATH) -> dict[str, str]:
    """Check every listed artifact. Returns {relative path: digest} for the verified set, or
    raises ArtifactIntegrityError naming EVERY failing path (not just the first)."""
    if not os.path.isfile(manifest_path):
        raise ArtifactIntegrityError(f"Artifact manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    entries = manifest.get("entries")
    if manifest.get("schema") != MANIFEST_SCHEMA or not entries:
        raise ArtifactIntegrityError("Artifact manifest has an unsupported schema or no entries")

    verified: dict[str, str] = {}
    problems: list[str] = []
    for entry in entries:
        rel_path, kind, expected = entry["path"], entry["kind"], entry["sha256"]
        if not os.path.isfile(os.path.join(root, rel_path)):
            problems.append(f"missing: {rel_path}")
            continue
        try:
            actual = compute_entry(root, rel_path, kind)
        except (OSError, sqlite3.Error) as exc:
            problems.append(f"unreadable: {rel_path} ({exc.__class__.__name__})")
            continue
        if actual != expected:
            problems.append(f"mismatch: {rel_path}")
        else:
            verified[rel_path] = actual
    if problems:
        raise ArtifactIntegrityError("Artifact integrity check failed: " + "; ".join(problems))
    return verified
