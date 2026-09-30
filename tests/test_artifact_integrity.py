"""
The manifest is what binds a served model_version to the bytes that were served. These tests pin
that it fails closed: a wrong hash or a missing artifact raises, the real tree matches its
manifest, and the Chroma content digest tracks the retrieval rows, not the file's bytes.
Run: python -m unittest discover -s tests
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import artifact_integrity as ai  # noqa: E402

VECTOR_STORE = os.path.join(ROOT, "src", "rag", "vector_store")


def _load_manifest():
    with open(ai.MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


class ManifestVerification(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write_manifest(self, manifest):
        path = os.path.join(self.tmp, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        return path

    def test_the_real_tree_matches_the_committed_manifest(self):
        verified = ai.verify_manifest()
        self.assertEqual(len(verified), len(_load_manifest()["entries"]))
        self.assertIn("models/model.json", verified)

    def test_a_wrong_hash_raises_and_names_the_path(self):
        manifest = _load_manifest()
        manifest["entries"][0]["sha256"] = "0" * 64
        path = self._write_manifest(manifest)
        with self.assertRaises(ai.ArtifactIntegrityError) as ctx:
            ai.verify_manifest(manifest_path=path)
        self.assertIn("mismatch: " + manifest["entries"][0]["path"], str(ctx.exception))

    def test_a_missing_artifact_raises(self):
        manifest = _load_manifest()
        manifest["entries"].append(
            {"path": "models/does_not_exist.json", "kind": ai.KIND_FILE, "sha256": "0" * 64}
        )
        path = self._write_manifest(manifest)
        with self.assertRaises(ai.ArtifactIntegrityError) as ctx:
            ai.verify_manifest(manifest_path=path)
        self.assertIn("missing: models/does_not_exist.json", str(ctx.exception))

    def test_a_missing_manifest_raises(self):
        with self.assertRaises(ai.ArtifactIntegrityError):
            ai.verify_manifest(manifest_path=os.path.join(self.tmp, "nope.json"))

    def test_every_model_artifact_an_endpoint_loads_is_listed(self):
        listed = {e["path"] for e in _load_manifest()["entries"]}
        for required in (
            "models/model.json",
            "models/model_meta.json",
            "models/symptom_model.json",
            "models/symptom_model_meta.json",
            "models/symptom_vectorizer_word.joblib",
            "models/symptom_vectorizer_char.joblib",
            "models/symptom_label_encoder.joblib",
            "dataset/canonical_dataset.csv",
            ai.CHROMA_SQLITE,
        ):
            self.assertIn(required, listed)


class ChromaContentDigest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.store = os.path.join(self.tmp, "vector_store")
        shutil.copytree(VECTOR_STORE, self.store)
        self.sqlite = os.path.join(self.store, "chroma.sqlite3")

    def test_the_copy_digest_equals_the_manifest_digest(self):
        entry = next(e for e in _load_manifest()["entries"] if e["kind"] == ai.KIND_CHROMA)
        self.assertEqual(ai.chroma_content_digest(self.sqlite), entry["sha256"])

    def test_changing_one_embedding_row_changes_the_digest(self):
        before = ai.chroma_content_digest(self.sqlite)
        con = sqlite3.connect(self.sqlite)
        con.execute(
            "UPDATE embeddings_queue SET vector = X'00000000' "
            "WHERE seq_id = (SELECT MIN(seq_id) FROM embeddings_queue)"
        )
        con.commit()
        con.close()
        self.assertNotEqual(before, ai.chroma_content_digest(self.sqlite))

    def test_changing_a_document_changes_the_digest(self):
        before = ai.chroma_content_digest(self.sqlite)
        con = sqlite3.connect(self.sqlite)
        con.execute(
            "UPDATE embedding_metadata SET string_value = 'tampered' "
            "WHERE key = 'chroma:document' AND id = (SELECT MIN(id) FROM embedding_metadata "
            "WHERE key = 'chroma:document')"
        )
        con.commit()
        con.close()
        self.assertNotEqual(before, ai.chroma_content_digest(self.sqlite))

    def test_opening_the_store_through_chroma_does_not_change_the_digest(self):
        """The reason the store is content-digested and not byte-hashed: a normal open rewrites
        chroma.sqlite3 and the HNSW files, and a restarted container must still verify."""
        import chromadb

        before_digest = ai.chroma_content_digest(self.sqlite)
        before_bytes = ai.sha256_file(self.sqlite)

        client = chromadb.PersistentClient(path=self.store)
        for collection in client.list_collections():
            name = getattr(collection, "name", collection)
            client.get_collection(name).query(query_embeddings=[[0.0] * 384], n_results=2)
        del client

        self.assertEqual(before_digest, ai.chroma_content_digest(self.sqlite))
        # Guard the premise: if this stops being true, byte-hashing became possible and the
        # docstring in artifact_integrity.py is out of date.
        self.assertNotEqual(before_bytes, ai.sha256_file(self.sqlite))


if __name__ == "__main__":
    unittest.main()
