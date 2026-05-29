"""
file_compressor.py
~~~~~~~~~~~~~~~~~~
Auto-compress uploaded study-material files that exceed Cloudinary's 10 MB
free-plan limit.

Supported strategies
---------------------
  .pdf   → re-written with pypdf (lossless page copy; removes embedded
            thumbnails and duplicate objects).
  .docx  │
  .pptx  → re-zipped with maximum ZIP compression (they are ZIP archives
  .xlsx  │ internally).
  other  → returned unchanged (caller should decide what to do).

Usage
-----
    from .file_compressor import compress_if_needed, MAX_UPLOAD_BYTES
    result = compress_if_needed(django_uploaded_file)
    # result.file      → compressed InMemoryUploadedFile (or original)
    # result.compressed → True if compression was applied
    # result.original_size_mb / result.final_size_mb
    # result.message   → human-readable status string
"""

from __future__ import annotations

import io
import os
import zipfile
import logging
from dataclasses import dataclass, field
from typing import Optional

from django.core.files.uploadedfile import InMemoryUploadedFile

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB – Cloudinary free-plan cap


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CompressionResult:
    file: object                     # original or compressed InMemoryUploadedFile
    compressed: bool = False
    original_size_mb: float = 0.0
    final_size_mb: float = 0.0
    message: str = ""
    error: Optional[str] = None      # set if compression failed


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def compress_if_needed(uploaded_file) -> CompressionResult:
    """
    Inspect *uploaded_file* (a Django UploadedFile / InMemoryUploadedFile).
    If its size exceeds MAX_UPLOAD_BYTES, attempt to compress it.
    Always returns a CompressionResult; the caller decides how to act on it.
    """
    if uploaded_file is None:
        return CompressionResult(file=uploaded_file, message="No file provided.")

    original_size = uploaded_file.size
    original_mb = original_size / (1024 * 1024)
    result = CompressionResult(
        file=uploaded_file,
        original_size_mb=round(original_mb, 2),
        final_size_mb=round(original_mb, 2),
    )

    if original_size <= MAX_UPLOAD_BYTES:
        result.message = f"File is {original_mb:.1f} MB — no compression needed."
        return result

    ext = os.path.splitext(uploaded_file.name or "")[1].lower()
    logger.info(
        "Compressing %s (%s, %.1f MB) before Cloudinary upload.",
        uploaded_file.name, ext, original_mb,
    )

    try:
        if ext == ".pdf":
            compressed = _compress_pdf(uploaded_file)
        elif ext in {".docx", ".pptx", ".xlsx"}:
            compressed = _rezip(uploaded_file, ext)
        else:
            # Cannot compress this type; return original and let caller decide
            result.message = (
                f"⚠️ File is {original_mb:.1f} MB but auto-compression is not "
                f"supported for {ext} files. Please reduce the file size manually."
            )
            result.error = "unsupported_type"
            return result

        if compressed is None:
            raise RuntimeError("Compression returned None.")

        final_size = compressed.size
        final_mb = final_size / (1024 * 1024)
        result.file = compressed
        result.compressed = True
        result.final_size_mb = round(final_mb, 2)

        if final_size > MAX_UPLOAD_BYTES:
            # Even after compression it's still too big
            ratio = (1 - final_size / original_size) * 100
            result.message = (
                f"⚠️ Compressed from {original_mb:.1f} MB → {final_mb:.1f} MB "
                f"({ratio:.0f}% smaller), but still exceeds the 10 MB limit. "
                f"Please reduce the file size further before uploading."
            )
            result.error = "still_too_large"
        else:
            ratio = (1 - final_size / original_size) * 100
            result.message = (
                f"✅ Compressed {original_mb:.1f} MB → {final_mb:.1f} MB "
                f"({ratio:.0f}% smaller) and uploaded successfully."
            )

    except Exception as exc:
        logger.exception("Compression failed for %s: %s", uploaded_file.name, exc)
        result.file = uploaded_file          # fall back to original
        result.error = "compression_failed"
        result.message = (
            f"⚠️ Auto-compression failed ({exc}). "
            f"The original file ({original_mb:.1f} MB) exceeds the 10 MB limit."
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PDF compression
# ─────────────────────────────────────────────────────────────────────────────

def _compress_pdf(uploaded_file) -> InMemoryUploadedFile:
    """
    Re-write a PDF using pypdf.
    This removes duplicate objects and embedded metadata/thumbnails that
    inflate file size.  It does NOT down-sample images (lossless only).
    """
    from pypdf import PdfReader, PdfWriter

    # Seek to beginning before reading
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    writer = PdfWriter()

    for page in reader.pages:
        writer.add_page(page)

    # Copy metadata so it isn't lost
    if reader.metadata:
        writer.add_metadata(reader.metadata)

    # Attempt to compress each page's content stream
    for page in writer.pages:
        try:
            page.compress_content_streams()
        except Exception:
            pass  # Non-fatal; skip if a page can't be compressed

    buffer = io.BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    size = buffer.getbuffer().nbytes

    return InMemoryUploadedFile(
        file=buffer,
        field_name="file",
        name=uploaded_file.name,
        content_type=getattr(uploaded_file, "content_type", "application/pdf"),
        size=size,
        charset=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DOCX / PPTX / XLSX re-compression (max ZIP deflate)
# ─────────────────────────────────────────────────────────────────────────────

def _rezip(uploaded_file, ext: str) -> InMemoryUploadedFile:
    """
    Re-zip a .docx / .pptx / .xlsx file with maximum DEFLATE compression.
    Office Open XML files are ZIP archives; many tools save them with
    compression level 0–3.  Re-zipping at level 9 often gives 20–40%
    savings on text-heavy documents.
    """
    uploaded_file.seek(0)
    original_zip = zipfile.ZipFile(uploaded_file, "r")

    out_buffer = io.BytesIO()
    with zipfile.ZipFile(out_buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as new_zip:
        for item in original_zip.infolist():
            data = original_zip.read(item.filename)
            new_zip.writestr(item, data)

    original_zip.close()
    out_buffer.seek(0)
    size = out_buffer.getbuffer().nbytes

    mime_map = {
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }

    return InMemoryUploadedFile(
        file=out_buffer,
        field_name="file",
        name=uploaded_file.name,
        content_type=mime_map.get(ext, "application/octet-stream"),
        size=size,
        charset=None,
    )
