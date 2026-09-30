"""
Write ARTIFACT_MANIFEST.json from the artifacts currently on disk.

Run deliberately, by the operator, after a model or dataset change is reviewed. It records what is
there now; it does not retrain or validate anything. The app verifies the result at import
(src/artifact_integrity.py), so an unreviewed change to any listed file stops the service.

    python scripts/write_artifact_manifest.py
"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "rag"))

import artifact_integrity as ai  # noqa: E402
from embedding_pin import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_REVISION  # noqa: E402

# Everything any endpoint loads: the vitals model and its metadata, the symptom model, both
# vectorizers, the label encoder, the dataset the tier profile is derived from at import, and the
# NLEM vector store.
FILE_ARTIFACTS = (
    "models/model.json",
    "models/model_meta.json",
    "models/symptom_model.json",
    "models/symptom_model_meta.json",
    "models/symptom_vectorizer_word.joblib",
    "models/symptom_vectorizer_char.joblib",
    "models/symptom_label_encoder.joblib",
    "dataset/canonical_dataset.csv",
)
VECTOR_STORE = "src/rag/vector_store"
# The two HNSW files that do not change when Chroma opens the store (see artifact_integrity).
STABLE_HNSW_FILES = ("header.bin", "link_lists.bin")


def main() -> None:
    entries = [(rel, ai.KIND_FILE) for rel in FILE_ARTIFACTS]
    for segment_dir in sorted(glob.glob(os.path.join(ROOT, VECTOR_STORE, "*", ""))):
        seg = os.path.basename(os.path.dirname(segment_dir))
        for name in STABLE_HNSW_FILES:
            entries.append((f"{VECTOR_STORE}/{seg}/{name}", ai.KIND_FILE))
    entries.append((ai.CHROMA_SQLITE, ai.KIND_CHROMA))

    manifest = {
        "schema": ai.MANIFEST_SCHEMA,
        "embedding_model": f"{EMBEDDING_MODEL_NAME}@{EMBEDDING_MODEL_REVISION}",
        "entries": [
            {"path": rel, "kind": kind, "sha256": ai.compute_entry(ROOT, rel, kind)}
            for rel, kind in entries
        ],
    }
    with open(ai.MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {ai.MANIFEST_PATH} ({len(manifest['entries'])} entries)")


if __name__ == "__main__":
    main()
