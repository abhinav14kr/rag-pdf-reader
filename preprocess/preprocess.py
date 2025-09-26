import json
import os
from pathlib import Path


import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ------------ settings ------------
SOURCE_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_PATH = Path(__file__).resolve().parents[1] / "web" / "public" / "index.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # 384-dim, fast
CHUNK_SIZE = 1200 # chars per chunk (approx ~200-250 tokens)
CHUNK_OVERLAP = 200 # overlap to preserve context
TOP_K = 5 # used by UI; kept for metadata/reference


# ------------ utils ------------


def read_pdf_text(pdf_path: Path):
    """
    Return a list of (page_number, text) tuples for a PDF.
    - Skips files that are encrypted and cannot be opened with an empty password.
    - Prints a concise reason when skipping.
    """
    pages = []
    try:
        reader = PdfReader(str(pdf_path))

        # If encrypted, try empty password (common on some free PDFs)
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")  # may succeed for owner-locked PDFs
            except Exception:
                print(f"[skip] {pdf_path.name}: encrypted (needs password)")
                return []

        # If still marked encrypted, skip
        if getattr(reader, "is_encrypted", False):
            print(f"[skip] {pdf_path.name}: still encrypted after empty-password attempt")
            return []

        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            pages.append((i + 1, txt))

    except Exception as e:
        print(f"[warn] {pdf_path.name}: failed to parse with pypdf — {e}")
        return []

    return pages




def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    n = len(text)
    start = 0
    step = max(1, chunk_size - overlap)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        start += step
    return chunks


# ------------ main ------------
if __name__ == "__main__":
    os.makedirs(OUT_PATH.parent, exist_ok=True)

    # Load embedder (local, no network after model is cached)
    embedder = SentenceTransformer(MODEL_NAME)

    records = []
    all_texts = []

    for pdf in sorted(SOURCE_DIR.glob("*.pdf")):
        book_name = pdf.stem.replace("_", " ")
        pages = read_pdf_text(pdf)
        
        if not pages:  # Skip if PDF couldn't be processed
            continue
            
        full_text = []
        page_map = []
        for pno, ptxt in pages:
            full_text.append(ptxt)
            page_map.append((pno, len("".join(full_text))))
        combined = "\n\n".join(full_text)
        chunks = chunk_text(combined)

        # map chunk to rough page range using cumulative lengths
        cum_lens = [len(c) for c in chunks]
        cum_lens = np.cumsum([0] + cum_lens).tolist()  # start offsets

        def approx_pages(start_idx, end_idx):
            start_char = cum_lens[start_idx]
            end_char = cum_lens[end_idx+1] if end_idx+1 < len(cum_lens) else cum_lens[-1]
            # find pages containing these char positions
            def find_page(pos):
                for pno, upto in page_map:
                    if pos <= upto:
                        return pno
                return page_map[-1][0]
            return find_page(start_char), find_page(end_char)

        for i, ch in enumerate(chunks):
            p_start, p_end = approx_pages(i, i)
            rec = {
                "id": f"{book_name}__chunk_{i}",
                "book": book_name,
                "chunk_id": i,
                "text": ch,
                "pages": [p_start, p_end],
                "source": f"{pdf.name}#pp.{p_start}-{p_end}",
            }
            records.append(rec)

    # Generate embeddings for all chunks
    all_texts = [rec["text"] for rec in records]
    if all_texts:
        embeddings = embedder.encode(all_texts)

        # Add embeddings to records
        for i, rec in enumerate(records):
            rec["embedding"] = embeddings[i].tolist()

        # Write the final output
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "records": records,
                "metadata": {
                    "model": MODEL_NAME,
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                    "total_chunks": len(records),
                    "embedding_dim": len(embeddings[0]) if len(embeddings) > 0 else 0,
                }
            }, f, indent=2)

    print(f"Wrote {len(records)} chunks to {OUT_PATH}")