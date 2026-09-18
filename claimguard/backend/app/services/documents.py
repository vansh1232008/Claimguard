"""Document ingestion and field extraction.

Handles the paperwork that comes with a motor claim: repair invoices, police
reports, FIR copies, garage estimates. Text is pulled with pdfminer/Tesseract
when they are installed; structured fields are then extracted with regular
expressions that are deliberately conservative (an unparsed field is reported
as missing rather than guessed).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

AMOUNT_RE = re.compile(
    r"(?:total|amount\s*(?:due|payable)?|grand\s*total|net\s*payable)\s*[:\-]?\s*"
    r"(?:inr|rs\.?|eur|€|\$)?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
    re.IGNORECASE,
)
INVOICE_RE = re.compile(r"(?:invoice|bill|receipt)\s*(?:no\.?|number|#)\s*[:\-]?\s*([A-Z0-9\-/]+)", re.I)
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})|(\d{2}[/-]\d{2}[/-]\d{4})")
GSTIN_RE = re.compile(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1})\b")
REG_RE = re.compile(r"\b([A-Z]{2}[\s-]?\d{1,2}[\s-]?[A-Z]{1,2}[\s-]?\d{4})\b")
VENDOR_RE = re.compile(r"(?:garage|workshop|service\s*cent(?:re|er)|motors|auto)\s*[:\-]?\s*(.{3,60})", re.I)
ODOMETER_RE = re.compile(r"(?:odometer|kms?\s*reading)\s*[:\-]?\s*([0-9][0-9,]*)", re.I)


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except (TypeError, ValueError):
        return None


def _normalise_date(match: re.Match) -> str | None:
    if match.group(1):
        return match.group(1)
    if match.group(2):
        parts = re.split(r"[/-]", match.group(2))
        if len(parts) == 3:
            return f"{parts[2]}-{parts[1]}-{parts[0]}"
    return None


# ----------------------------------------------------------------- text layer
def extract_text(path: Path) -> str:
    """Best-effort text extraction across pdf / image / plain text."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".csv"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".json":
        # A structured sidecar: flatten it so the same regexes still work.
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return "\n".join(f"{k}: {v}" for k, v in data.items())
        except Exception:
            return ""
    if suffix == ".pdf":
        try:
            from pdfminer.high_level import extract_text as pdf_extract

            return pdf_extract(str(path)) or ""
        except Exception as exc:
            logger.info("pdfminer unavailable/failed for %s (%s)", path.name, exc)
            return ""
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}:
        try:
            import pytesseract
            from PIL import Image

            with Image.open(path) as im:
                return pytesseract.image_to_string(im)
        except Exception as exc:
            logger.info("OCR unavailable/failed for %s (%s)", path.name, exc)
            return ""
    return ""


# ------------------------------------------------------------ field extraction
def extract_fields(text: str, doc_type: str = "invoice") -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if not text:
        return fields

    amounts = [_to_float(m.group(1)) for m in AMOUNT_RE.finditer(text)]
    amounts = [a for a in amounts if a is not None]
    if amounts:
        fields["total_amount"] = max(amounts)
        fields["all_amounts"] = amounts[:10]

    inv = INVOICE_RE.search(text)
    if inv:
        fields["invoice_number"] = inv.group(1).strip()

    dates = [_normalise_date(m) for m in DATE_RE.finditer(text)]
    dates = [d for d in dates if d]
    if dates:
        fields["service_date"] = min(dates)
        fields["all_dates"] = dates[:6]

    gst = GSTIN_RE.search(text)
    if gst:
        fields["gstin"] = gst.group(1)

    reg = REG_RE.search(text.upper())
    if reg:
        fields["registration"] = re.sub(r"[\s-]", "", reg.group(1))

    vendor = VENDOR_RE.search(text)
    if vendor:
        fields["vendor_name"] = vendor.group(0).strip().splitlines()[0][:80]

    odo = ODOMETER_RE.search(text)
    if odo:
        fields["odometer"] = _to_float(odo.group(1))

    fields["doc_type"] = doc_type
    fields["text_length"] = len(text)
    return fields


def process_document(path: Path, doc_type: str = "invoice") -> dict[str, Any]:
    text = extract_text(path)
    fields = extract_fields(text, doc_type)
    return {
        "filename": path.name,
        "doc_type": doc_type,
        "raw_text": text[:8000],
        "extracted_fields": fields,
        "extraction_ok": bool(text),
    }
