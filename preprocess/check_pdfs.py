# save as check_pdfs.py and run:  python check_pdfs.py
from pathlib import Path
from pypdf import PdfReader

data_dir = Path(__file__).resolve().parents[1] / "data"
for pdf in sorted(data_dir.glob("*.pdf")):
    try:
        r = PdfReader(str(pdf))
        enc = getattr(r, "is_encrypted", False)
        print(f"{pdf.name}: encrypted={enc}, pages={(len(r.pages) if not enc else 'N/A')}")
    except Exception as e:
        print(f"{pdf.name}: ERROR -> {e}")
