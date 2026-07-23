"""
Deterministic (no-LLM) ingestion of the NLEM 2022 PDF into a local ChromaDB
vector store.

Parsing is pure regex/state-machine based on the confirmed tabular structure
of the NLEM 2022 PDF:

    {item_number}
    {Medicine Name}
    {Level of Healthcare: P,S,T}
    {Dosage form(s) and strength(s), one or more lines}

Section/subsection headers are tracked as running state so every extracted
medicine item can be attributed to its section for citation purposes. The
"Alphabetical List of Medicines in NLEM 2022" is parsed separately into a
flat name index used by the validator for drug-existence checks.

Run standalone:
    python rag/ingest.py
"""

import re
from pathlib import Path

import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "corpus" / "nlem2022.pdf"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MEDICINES_COLLECTION = "nlem_medicines"
ALPHA_INDEX_COLLECTION = "nlem_alpha_index"

# Table body spans these 0-indexed PyMuPDF pages; page 84 is the divider
# before the alphabetical index, so the table ends there.
TABLE_START_PAGE = 3
TABLE_END_PAGE = 84  # exclusive

# Alphabetical List of Medicines in NLEM 2022 (0-indexed pages, inclusive).
ALPHA_INDEX_START_PAGE = 85
ALPHA_INDEX_END_PAGE = 101  # exclusive

SECTION_MAJOR_RE = re.compile(r"^Section\s+(\d+)\s*$")
SUBSECTION_DASH_RE = re.compile(r"^(?:Section\s+)?(\d+\.\d+)\s*-\s*(.+)$")
BARE_NUM_RE = re.compile(r"^(\d+(?:\.\d+){1,4})(?:\s+(.+))?$")
LEVEL_RE = re.compile(r"^[PST](\s*,\s*[PST]){0,2}$")
HEADER_TOKENS = {"Medicine", "Level of", "Healthcare", "Dosage form(s) and", "strength(s)"}
ALPHA_ENTRY_RE = re.compile(r"^(\d+)\.\s+(.+)$")


def _is_header(line):
    return line in HEADER_TOKENS or line.startswith("Healthcare")


def _classify_bare(lines, i, max_steps=10):
    """Look ahead from a bare numbered line to decide what it starts.

    Returns 'item' if a Level-of-Healthcare code (P/S/T) appears before any
    other structural boundary, 'subsection' if a table/section boundary is
    hit first (i.e. this number line is a subsection heading with no level
    of its own), or 'continuation' if another bare-numbered line is hit
    first (i.e. this line is wrapped dosage text, not a new entry).
    """
    j = i + 1
    n = len(lines)
    steps = 0
    while j < n and steps < max_steps:
        candidate = lines[j].strip()
        if candidate == "":
            j += 1
            continue
        if LEVEL_RE.match(candidate):
            return "item"
        if _is_header(candidate) or SECTION_MAJOR_RE.match(candidate) or SUBSECTION_DASH_RE.match(candidate):
            return "subsection"
        if BARE_NUM_RE.match(candidate):
            return "continuation"
        j += 1
        steps += 1
    return "continuation"


def _parse_page(lines, state):
    """Parse one page's lines into medicine-item dicts, mutating `state`
    (current section/subsection) as headers are encountered."""
    items = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if line == "":
            i += 1
            continue

        m = SECTION_MAJOR_RE.match(line)
        if m:
            state["section_num"] = m.group(1)
            state["subsection_num"] = None
            state["subsection_title"] = None
            i += 1
            while i < n and lines[i].strip() == "":
                i += 1
            if i < n:
                state["section_title"] = lines[i].strip()
                i += 1
            continue

        m = SUBSECTION_DASH_RE.match(line)
        if m:
            state["subsection_num"] = m.group(1)
            state["subsection_title"] = m.group(2).strip()
            i += 1
            continue

        if _is_header(line):
            i += 1
            continue

        m = BARE_NUM_RE.match(line)
        if m:
            kind = _classify_bare(lines, i)
            if kind == "subsection":
                state["subsection_num"] = m.group(1)
                state["subsection_title"] = (m.group(2) or "").strip()
                i += 1
                continue
            if kind == "continuation":
                i += 1
                continue

            item_num = m.group(1)
            rest = (m.group(2) or "").strip()
            i += 1
            name_lines = [rest] if rest else []
            steps = 0
            while i < n and not LEVEL_RE.match(lines[i].strip()) and steps < 8:
                if lines[i].strip() != "":
                    name_lines.append(lines[i].strip())
                i += 1
                steps += 1

            level = None
            if i < n and LEVEL_RE.match(lines[i].strip()):
                level = lines[i].strip()
                i += 1

            dosage_lines = []
            while i < n:
                candidate = lines[i].strip()
                if candidate == "":
                    i += 1
                    continue
                if (
                    _is_header(candidate)
                    or candidate.startswith("*")
                    or candidate.startswith("#")
                    or SECTION_MAJOR_RE.match(candidate)
                    or SUBSECTION_DASH_RE.match(candidate)
                ):
                    break
                m2 = BARE_NUM_RE.match(candidate)
                if m2 and _classify_bare(lines, i) in ("item", "subsection"):
                    break
                dosage_lines.append(candidate)
                i += 1

            items.append(
                {
                    "item_num": item_num,
                    "name": " ".join(name_lines).strip(),
                    "level": level,
                    "level_codes": [c.strip() for c in level.split(",")] if level else [],
                    "dosage_lines": dosage_lines,
                    "section_num": state.get("section_num"),
                    "section_title": state.get("section_title"),
                    "subsection_num": state.get("subsection_num"),
                    "subsection_title": state.get("subsection_title"),
                }
            )
            continue

        i += 1
    return items


def extract_medicine_items(doc):
    state = {}
    all_items = []
    for page_idx in range(TABLE_START_PAGE, TABLE_END_PAGE):
        text = doc[page_idx].get_text()
        lines = text.split("\n")
        items = _parse_page(lines, state)
        for item in items:
            item["page"] = page_idx + 1  # human-readable 1-indexed page
        all_items.extend(items)
    return all_items


def extract_alpha_index(doc):
    """Flat exact-name lookup table from the 'Alphabetical List of Medicines
    in NLEM 2022' pages."""
    entries = []
    for page_idx in range(ALPHA_INDEX_START_PAGE, ALPHA_INDEX_END_PAGE):
        text = doc[page_idx].get_text()
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            m = ALPHA_ENTRY_RE.match(line)
            if not m:
                continue
            entries.append(
                {
                    "index_number": int(m.group(1)),
                    "name": m.group(2).strip(),
                    "page": page_idx + 1,
                }
            )
    return entries


def build_chunk_text(item):
    dosage_text = "; ".join(item["dosage_lines"]) if item["dosage_lines"] else "not specified"
    subsection = f" | {item['subsection_title']}" if item.get("subsection_title") else ""
    return (
        f"Section {item['section_num']} - {item['section_title']}{subsection} | "
        f"{item['item_num']} {item['name']} | "
        f"Level of Healthcare: {item['level']} | "
        f"Dosage form(s) and strength(s): {dosage_text}"
    )


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"NLEM 2022 PDF not found at {PDF_PATH}")

    doc = fitz.open(str(PDF_PATH))
    medicine_items = extract_medicine_items(doc)
    alpha_entries = extract_alpha_index(doc)
    doc.close()

    print(f"Extracted {len(medicine_items)} medicine chunks from the NLEM table.")
    print(f"Extracted {len(alpha_entries)} entries from the alphabetical index.")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))

    for name in (MEDICINES_COLLECTION, ALPHA_INDEX_COLLECTION):
        try:
            client.delete_collection(name)
        except Exception:
            pass

    medicines_collection = client.create_collection(
        MEDICINES_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    alpha_collection = client.create_collection(
        ALPHA_INDEX_COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    chunk_texts = [build_chunk_text(item) for item in medicine_items]
    chunk_ids = [f"item-{idx}-{item['item_num']}" for idx, item in enumerate(medicine_items)]
    chunk_embeddings = model.encode(chunk_texts, show_progress_bar=True, convert_to_numpy=True).tolist()
    chunk_metadatas = [
        {
            "item_num": item["item_num"],
            "name": item["name"],
            "level": item["level"],
            "level_codes": ",".join(item["level_codes"]),
            "dosage_lines": " | ".join(item["dosage_lines"]),
            "section_num": item["section_num"] or "",
            "section_title": item["section_title"] or "",
            "subsection_num": item["subsection_num"] or "",
            "subsection_title": item["subsection_title"] or "",
            "page": item["page"],
        }
        for item in medicine_items
    ]
    medicines_collection.add(
        ids=chunk_ids,
        documents=chunk_texts,
        embeddings=chunk_embeddings,
        metadatas=chunk_metadatas,
    )

    alpha_texts = [entry["name"] for entry in alpha_entries]
    alpha_ids = [f"alpha-{entry['index_number']}" for entry in alpha_entries]
    alpha_embeddings = model.encode(alpha_texts, show_progress_bar=True, convert_to_numpy=True).tolist()
    alpha_metadatas = [
        {"index_number": entry["index_number"], "name": entry["name"], "page": entry["page"]}
        for entry in alpha_entries
    ]
    alpha_collection.add(
        ids=alpha_ids,
        documents=alpha_texts,
        embeddings=alpha_embeddings,
        metadatas=alpha_metadatas,
    )

    print(f"\nStored {medicines_collection.count()} chunks in '{MEDICINES_COLLECTION}'.")
    print(f"Stored {alpha_collection.count()} entries in '{ALPHA_INDEX_COLLECTION}'.")
    print(f"Vector store persisted at: {VECTOR_STORE_DIR}\n")

    print("Sample chunks:")
    for item in medicine_items[:3]:
        print(" -", build_chunk_text(item))


if __name__ == "__main__":
    main()
