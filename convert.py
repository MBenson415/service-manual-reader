#!/usr/bin/env python3
"""Convert PDF service manuals into structured markdown files for Claude Code.

Produces:
  - Markdown per section with hierarchy, parts tables, adjustment steps
  - PNG schematics named by assembly (e.g., awh-046-schematic.png)
  - Cross-referenced index with metadata
"""

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image

from schematic_imaging import preprocess_image

DEFAULT_OUTPUT_DIR = Path("/Users/marshallbenson/Desktop/Benchmark Audio Repair/Schematics")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")[:80]


def extract_assembly_code(title: str) -> Optional[str]:
    """Extract assembly/board code from a section title.

    Handles multiple formats:
      (AWH-046)          — Pioneer style
      (PCB 19-1360)      — Proton / generic PCB style
      (PCB-19-1361)      — variant with dash
    """
    # Try parenthesized codes first
    m = re.search(r"\(([A-Z]{2,4}[\s-]\d{2,5}(?:-\d{1,5})?)\)", title)
    if m:
        return m.group(1).strip()
    # Try unparenthesized "PCB NN-NNNN" anywhere in the title
    m = re.search(r"\b(PCB[\s-]\d{2,5}(?:-\d{1,5})?)\b", title, re.IGNORECASE)
    if m:
        return m.group(1).strip().upper()
    return None


def find_watermarks(doc: fitz.Document) -> set:
    line_counts = Counter()
    for page_num in range(doc.page_count):
        text = doc[page_num].get_text("text")
        seen = set()
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                line_counts[stripped] += 1
    threshold = max(doc.page_count * 0.3, 3)
    return {line for line, count in line_counts.items() if count >= threshold}


def strip_watermarks(text: str, watermarks: set) -> str:
    lines = text.split("\n")
    return "\n".join(l for l in lines if l.strip() not in watermarks)


def get_page_text(doc: fitz.Document, page_num: int, watermarks: set) -> str:
    return strip_watermarks(doc[page_num].get_text("text"), watermarks)


def classify_pages(doc: fitz.Document, watermarks: set) -> list:
    classes = []
    for page_num in range(doc.page_count):
        clean = get_page_text(doc, page_num, watermarks).strip()
        classes.append("text" if len(clean) >= 50 else "image")
    return classes


def clean_text(text: str) -> str:
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return "\n".join(line.rstrip() for line in text.split("\n"))


# ---------------------------------------------------------------------------
# Section detection (tried in order)
# ---------------------------------------------------------------------------

def sections_from_toc_links(doc: fitz.Document, watermarks: set) -> list:
    """Build sections from internal links on TOC pages."""
    toc_links = []
    for page_num in range(min(doc.page_count, 10)):
        page = doc[page_num]
        links = page.get_links()
        internal = [l for l in links if l.get("kind") == 1]
        if len(internal) >= 5:
            for link in internal:
                rect = fitz.Rect(link["from"])
                text = page.get_text("text", clip=rect).strip()
                dest = link.get("page", -1)
                if text and dest >= 0:
                    text = text.split("\n")[0].strip()
                    toc_links.append((text, dest))

    if not toc_links:
        return []

    seen_pages = set()
    unique = []
    for text, dest in toc_links:
        if dest not in seen_pages:
            seen_pages.add(dest)
            unique.append((text, dest))
    unique.sort(key=lambda x: x[1])

    sections = []
    for i, (title, start_page) in enumerate(unique):
        end_page = unique[i + 1][1] - 1 if i + 1 < len(unique) else doc.page_count - 1
        end_page = max(start_page, end_page)
        pages = set(range(start_page, end_page + 1))
        text_parts = [(p, get_page_text(doc, p, watermarks)) for p in range(start_page, end_page + 1)]
        sections.append({
            "title": title,
            "start_page": start_page,
            "pages": pages,
            "text_parts": text_parts,
        })
    return sections


def sections_from_outline(doc: fitz.Document, watermarks: set) -> list:
    toc = doc.get_toc()
    if not toc or len(toc) <= 1:
        return []
    entries = [(t.strip(), max(p - 1, 0)) for lv, t, p in toc if lv <= 2]
    if len(entries) <= 1:
        return []
    sections = []
    for i, (title, sp) in enumerate(entries):
        ep = entries[i + 1][1] - 1 if i + 1 < len(entries) else doc.page_count - 1
        ep = max(sp, ep)
        pages = set(range(sp, ep + 1))
        text_parts = [(p, get_page_text(doc, p, watermarks)) for p in range(sp, ep + 1)]
        sections.append({"title": title, "start_page": sp, "pages": pages, "text_parts": text_parts})
    return sections


def sections_from_fonts(doc: fitz.Document, watermarks: set) -> list:
    font_sizes = defaultdict(int)
    for pn in range(doc.page_count):
        for block in doc[pn].get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    t = span["text"].strip()
                    if t and t not in watermarks:
                        font_sizes[round(span["size"], 1)] += len(t)
    if not font_sizes:
        return []
    body_size = max(font_sizes, key=font_sizes.get)
    header_threshold = body_size * 1.4
    sections = []
    current = None
    for pn in range(doc.page_count):
        pt = get_page_text(doc, pn, watermarks)
        for block in doc[pn].get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                lt = ""
                ms = 0
                for span in line["spans"]:
                    lt += span["text"]
                    ms = max(ms, span["size"])
                lt = lt.strip()
                if not lt or lt in watermarks:
                    continue
                if ms >= header_threshold and 2 < len(lt) < 120:
                    if current:
                        sections.append(current)
                    current = {"title": lt, "start_page": pn, "pages": set(), "text_parts": []}
        if current is not None:
            current["pages"].add(pn)
            current["text_parts"].append((pn, pt))
        elif pt.strip():
            current = {"title": "Introduction", "start_page": pn, "pages": {pn}, "text_parts": [(pn, pt)]}
    if current:
        sections.append(current)
    if len(sections) > 1:
        merged = [sections[0]]
        for s in sections[1:]:
            if s["title"] == merged[-1]["title"]:
                merged[-1]["pages"].update(s["pages"])
                merged[-1]["text_parts"].extend(s["text_parts"])
            else:
                merged.append(s)
        sections = merged
    return sections if len(sections) > 1 else []


def sections_fallback(doc: fitz.Document, watermarks: set, chunk: int = 30) -> list:
    sections = []
    for start in range(0, doc.page_count, chunk):
        end = min(start + chunk, doc.page_count) - 1
        pages = set(range(start, end + 1))
        text_parts = [(p, get_page_text(doc, p, watermarks)) for p in range(start, end + 1)]
        sections.append({
            "title": "Pages {}-{}".format(start + 1, end + 1),
            "start_page": start,
            "pages": pages,
            "text_parts": text_parts,
        })
    return sections


# ---------------------------------------------------------------------------
# Vision-based section detection (for scanned / image-only PDFs)
# ---------------------------------------------------------------------------

VISION_MODEL = os.environ.get("SCHEMATIC_VISION_MODEL", "claude-haiku-4-5-20251001")
VISION_BATCH = 4          # pages per API call
VISION_DPI   = 120        # lower DPI for efficient API transfer
SPARSE_THRESHOLD = 0.80   # trigger vision when >= this fraction are image pages


def _is_sparse_text(page_classes: list) -> bool:
    """True when 80%+ of pages have no extractable text layer."""
    if not page_classes:
        return True
    return page_classes.count("image") / len(page_classes) >= SPARSE_THRESHOLD


VISION_MAX_PX = 3000      # cap longest side to stay well under API 8000px limit


def _render_page_jpeg(doc: fitz.Document, page_num: int) -> bytes:
    """Render a page to compressed grayscale JPEG for API transfer.

    Automatically scales oversized pages so the longest side never
    exceeds VISION_MAX_PX, keeping the image under API size limits.
    """
    page = doc[page_num]
    # Calculate zoom: start from VISION_DPI, then cap to max pixel limit
    zoom = VISION_DPI / 72
    width_px = page.rect.width * zoom
    height_px = page.rect.height * zoom
    longest = max(width_px, height_px)
    if longest > VISION_MAX_PX:
        zoom = zoom * (VISION_MAX_PX / longest)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    image = preprocess_image(Image.frombytes("L", (pix.width, pix.height), pix.samples))
    image.thumbnail((VISION_MAX_PX, VISION_MAX_PX), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


_VISION_PROMPT = (
    "These are pages from an electronics service manual. "
    "For each page, return a JSON array with one object per page in order. "
    "Each object MUST have exactly these keys:\n"
    '  "title": the assembly or section name printed on the page '
    '(e.g. "Power Amplifier Assembly", "Parts List of Tone Control"); empty string if unclear\n'
    '  "assembly_code": board code visible in a title block or schematic border, '
    'format like "AWH-046" or "AWR-099"; empty string if none visible\n'
    '  "section_type": one of "schematic", "parts-list", "adjustment", '
    '"specs", "exploded-view", "general"\n'
    '  "starts_new_section": true if this page starts a new assembly/section, '
    'false if it continues the previous page\n'
    "Respond with ONLY the JSON array — no markdown, no extra text."
)


def _call_vision_api(client, page_jpegs: list, page_offset: int) -> list:
    """Send a batch of JPEG bytes to Claude and return per-page metadata dicts."""
    import json

    content = []
    for i, jpeg_bytes in enumerate(page_jpegs):
        content.append({"type": "text", "text": "Page {}:".format(page_offset + i + 1)})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(jpeg_bytes).decode("ascii"),
            },
        })
    content.append({"type": "text", "text": _VISION_PROMPT})

    response = client.messages.create(
        model=VISION_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.content[0].text.strip()
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return _blank_vision_results(len(page_jpegs))
    try:
        results = json.loads(m.group())
        while len(results) < len(page_jpegs):
            results.append(_blank_vision_entry(False))
        return results[: len(page_jpegs)]
    except (json.JSONDecodeError, ValueError):
        return _blank_vision_results(len(page_jpegs))


def _blank_vision_entry(starts_new: bool) -> dict:
    return {"title": "", "assembly_code": "", "section_type": "general", "starts_new_section": starts_new}


def _blank_vision_results(n: int) -> list:
    return [_blank_vision_entry(i == 0) for i in range(n)]


def _build_sections_from_vision_metadata(
    metadata: list, doc: fitz.Document, watermarks: set,
) -> list:
    """Convert per-page vision metadata into section dicts matching the standard format."""
    sections = []
    current = None

    for page_num, meta in enumerate(metadata):
        page_text = get_page_text(doc, page_num, watermarks)
        vis_title = meta.get("title", "").strip()
        asm_code = meta.get("assembly_code", "").strip().upper()

        # Build a clean title combining vision title and assembly code
        if vis_title and asm_code and asm_code not in vis_title.upper():
            full_title = "{} ({})".format(vis_title, asm_code)
        elif vis_title:
            full_title = vis_title
        elif asm_code:
            full_title = "Assembly {}".format(asm_code)
        else:
            full_title = "Page {}".format(page_num + 1)

        starts_new = meta.get("starts_new_section", False) or current is None

        # Force a new section when the assembly code changes
        if current is not None and not starts_new:
            prev_asm = current.get("_asm_code", "")
            if prev_asm and asm_code and prev_asm != asm_code:
                starts_new = True

        if starts_new:
            if current:
                sections.append(current)
            current = {
                "title": full_title,
                "start_page": page_num,
                "pages": {page_num},
                "text_parts": [(page_num, page_text)],
                "_asm_code": asm_code,
                "vision_type": meta.get("section_type") or "general",
            }
        else:
            if current:
                current["pages"].add(page_num)
                current["text_parts"].append((page_num, page_text))

    if current:
        sections.append(current)

    return sections


def sections_from_vision(doc: fitz.Document, watermarks: set) -> list:
    """Use Claude vision API to identify sections in image-only PDFs.

    Requires ANTHROPIC_API_KEY environment variable.
    Returns sections in the same format as other detection methods.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  ANTHROPIC_API_KEY not set — skipping vision analysis.")
        return []

    try:
        import anthropic
    except ImportError:
        print("  'anthropic' package not installed — skipping vision analysis.")
        print("  Install with: pip install anthropic")
        return []

    client = anthropic.Anthropic(api_key=api_key)
    total = doc.page_count
    print("  Analyzing {} pages in batches of {} (model: {})...".format(
        total, VISION_BATCH, VISION_MODEL,
    ))

    all_metadata: list = []
    for batch_start in range(0, total, VISION_BATCH):
        batch_end = min(batch_start + VISION_BATCH, total)
        page_jpegs = [_render_page_jpeg(doc, p) for p in range(batch_start, batch_end)]
        try:
            batch_meta = _call_vision_api(client, page_jpegs, batch_start)
        except Exception as exc:
            print("    Warning: API error pages {}-{}: {}".format(batch_start + 1, batch_end, exc))
            batch_meta = _blank_vision_results(batch_end - batch_start)
        all_metadata.extend(batch_meta)
        print("  Pages {}/{} analysed".format(min(batch_end, total), total), end="\r", flush=True)

    print()  # newline after progress line

    sections = _build_sections_from_vision_metadata(all_metadata, doc, watermarks)

    # Report unique assemblies found
    assemblies = {s["_asm_code"] for s in sections if s.get("_asm_code")}
    if assemblies:
        print("  Assemblies identified: {}".format(", ".join(sorted(assemblies))))

    return sections


# ---------------------------------------------------------------------------
# Section classification (schematic, parts-list, adjustment, etc.)
# ---------------------------------------------------------------------------

# Checked in order — more specific patterns first so "Parts List of X Assembly"
# matches parts-list before the assembly regex can claim it as a schematic.
SECTION_TYPES = [
    ("parts-list", re.compile(
        r"parts?\s+list|parts?\s+location|miscellaneous\s+parts",
        re.IGNORECASE,
    )),
    ("adjustment", re.compile(
        r"adjust|tracking|idle\s+current|bias|alignment|calibrat",
        re.IGNORECASE,
    )),
    ("specs", re.compile(
        r"specification|spec\b",
        re.IGNORECASE,
    )),
    ("exploded-view", re.compile(
        r"exploded\s+view|disassembly|packing",
        re.IGNORECASE,
    )),
    ("schematic", re.compile(
        r"schematic|circuit\s+description|block\s+diagram|connection\s+diagram|"
        r"internal\s+circuitry|front\s+end|multiplex|fm\s+tun|am\s+tun|"
        r"tone\s+control|power\s+amplifier\s+assembly|power\s+supply.*assembly|"
        r"filter.*assembly|headphone.*assembly|tuner.*assembly|amplifier\s+assembly|"
        r"de-emphasis.*assembly",
        re.IGNORECASE,
    )),
]


def classify_section(title: str) -> str:
    """Return a section type label based on title keywords."""
    for stype, pattern in SECTION_TYPES:
        if pattern.search(title):
            return stype
    return "general"


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------

DPI_SCHEMATIC = 200  # Higher for legible component designators
DPI_GENERAL = 150


def extract_page_image(
    doc: fitz.Document, page_num: int, out_dir: Path,
    image_name: str, dpi: int = DPI_GENERAL,
) -> str:
    """Render a PDF page to a grayscale PNG. Returns the filename."""
    page = doc[page_num]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    filename = "{}.png".format(image_name)
    image_path = out_dir / filename
    pix.save(str(image_path))
    visible_spans = [fitz.Rect(span["bbox"]) for span in page.get_texttrace() if span["type"] != 3]
    words = []
    for word in page.get_text("words", sort=True):
        original_box = fitz.Rect(word[:4])
        center = (original_box.tl + original_box.br) / 2
        box = original_box * page.rotation_matrix
        words.append({
            "text": word[4],
            "bbox": [box.x0 / page.rect.width, box.y0 / page.rect.height,
                     box.x1 / page.rect.width, box.y1 / page.rect.height],
            "visible": any(center in span for span in visible_spans),
        })
    metadata = {
        "version": 1, "page": page_num + 1,
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "words": words,
    }
    image_path.with_suffix(".text.json").write_text(json.dumps(metadata), encoding="utf-8")
    return filename


def make_image_name(section_slug: str, page_num: int, section_type: str, page_index: int) -> str:
    """Build a descriptive image filename.

    e.g. power-amplifier-assembly-awh-046-schematic-p57
         parts-location-p28
    """
    suffix = ""
    if section_type == "schematic":
        suffix = "-schematic"
    elif section_type == "parts-list":
        suffix = "-parts"
    elif section_type == "exploded-view":
        suffix = "-exploded"
    elif section_type == "adjustment":
        suffix = "-adjustment"
    return "{}{}-p{:03d}".format(section_slug, suffix, page_num + 1)


# ---------------------------------------------------------------------------
# Parts-list table detection
# ---------------------------------------------------------------------------

def try_format_as_table(text: str) -> str:
    """If text looks like a parts list, convert to a markdown table.

    Detects lines with consistent column-like spacing (designator, value,
    part number patterns).
    """
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 3:
        return text

    # Look for patterns like: R101  4.7kΩ  RD-1234  Resistor
    # or tabular lines with 3+ whitespace-separated columns
    table_lines = []
    non_table = []
    part_pattern = re.compile(
        r"^([A-Z]{1,3}\d{1,4})\s{2,}(.+?)$"
    )

    for line in lines:
        if part_pattern.match(line.strip()):
            table_lines.append(line.strip())
        else:
            # If we had been building a table, flush it
            if table_lines:
                non_table.append(_build_table(table_lines))
                table_lines = []
            non_table.append(line)

    if table_lines:
        non_table.append(_build_table(table_lines))

    return "\n".join(non_table)


def _build_table(lines: list) -> str:
    """Convert lines that look like parts entries into a markdown table."""
    rows = []
    for line in lines:
        # Split on 2+ spaces to get columns
        cols = re.split(r"\s{2,}", line.strip())
        rows.append(cols)

    if not rows:
        return "\n".join(lines)

    # Determine column count from most common
    col_counts = Counter(len(r) for r in rows)
    target_cols = col_counts.most_common(1)[0][0]

    # Pad/trim rows to target
    normalized = []
    for r in rows:
        if len(r) < target_cols:
            r = r + [""] * (target_cols - len(r))
        elif len(r) > target_cols:
            # Merge extra columns into last
            r = r[:target_cols - 1] + [" ".join(r[target_cols - 1:])]
        normalized.append(r)

    # Build table
    headers = ["Designator", "Value", "Part Number", "Description"]
    if target_cols <= len(headers):
        header = headers[:target_cols]
    else:
        header = ["Col {}".format(i + 1) for i in range(target_cols)]

    table = "| {} |".format(" | ".join(header))
    table += "\n| {} |".format(" | ".join(["---"] * target_cols))
    for row in normalized:
        table += "\n| {} |".format(" | ".join(row))

    return table


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_section(
    section: dict, index: int, out_dir: Path, page_classes: list,
    doc: fitz.Document, section_type: str,
) -> tuple:
    title = section["title"]
    slug = slugify(title) or "section-{}".format(index)
    filename = "{:02d}-{}.md".format(index, slug)
    assembly = extract_assembly_code(title) or section.get("_asm_code") or None

    pages = sorted(section["pages"])
    if not pages:
        return filename, title, "0", section_type, assembly, []

    page_range = (
        "{}-{}".format(pages[0] + 1, pages[-1] + 1)
        if len(pages) > 1
        else str(pages[0] + 1)
    )

    # Track images produced by this section
    image_files = []

    parts = []

    # Section header with type tag and assembly code
    parts.append("# {}".format(title))
    tag_parts = []
    if section_type != "general":
        tag_parts.append("`{}`".format(section_type))
    if assembly:
        tag_parts.append("**Board:** `{}`".format(assembly))
    if tag_parts:
        parts.append(" | ".join(tag_parts))
    parts.append("<!-- pages {} -->".format(page_range))
    parts.append("")

    dpi = DPI_SCHEMATIC if section_type == "schematic" else DPI_GENERAL

    for page_idx, (page_num, text) in enumerate(section["text_parts"]):
        cleaned = clean_text(text)
        is_image = page_num < len(page_classes) and page_classes[page_num] == "image"

        if is_image or section_type in {"schematic", "parts-list", "adjustment", "exploded-view"}:
            img_name = make_image_name(slug, page_num, section_type, page_idx)
            img_file = extract_page_image(doc, page_num, out_dir, img_name, dpi)
            image_files.append(img_file)

            parts.append("---")
            parts.append("*Page {}*\n".format(page_num + 1))
            parts.append("![{} — page {}]({})".format(title, page_num + 1, img_file))
            if cleaned.strip():
                parts.append("")
                parts.append(try_format_as_table(cleaned) if section_type == "parts-list" else cleaned)
            parts.append("")
        elif cleaned.strip():
            parts.append("---")
            parts.append("*Page {}*\n".format(page_num + 1))
            # Try to format parts-list pages as tables
            if section_type == "parts-list":
                parts.append(try_format_as_table(cleaned))
            else:
                parts.append(cleaned)
            parts.append("")

    (out_dir / filename).write_text("\n".join(parts), encoding="utf-8")
    return filename, title, page_range, section_type, assembly, image_files


def write_index(
    manual_name: str, section_info: list, out_dir: Path, stats: dict,
):
    """Write _index.md with metadata, assembly cross-references, and TOC."""
    lines = []

    # Metadata header
    lines.append("# {}".format(manual_name))
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append("- **Total pages:** {} ({} text, {} diagrams/images)".format(
        stats["total"], stats["text"], stats["image"],
    ))

    # Collect assemblies
    assemblies = []
    for _, title, _, stype, asm, _ in section_info:
        if asm and asm not in [a[0] for a in assemblies]:
            assemblies.append((asm, title))
    if assemblies:
        lines.append("- **Board assemblies:**")
        for code, title in assemblies:
            lines.append("  - `{}` — {}".format(code, title))

    lines.append("")

    if stats["image"] > stats["text"]:
        lines.append(
            "> This manual is primarily diagrams/schematics. Each section's "
            "markdown file embeds its PNG images inline. Schematic images are "
            "rendered at {} DPI for legible component designators.".format(DPI_SCHEMATIC)
        )
        lines.append("")

    # TOC grouped by type
    type_order = ["specs", "schematic", "adjustment", "parts-list", "exploded-view", "general"]
    type_labels = {
        "specs": "Specifications",
        "schematic": "Schematics & Circuit Descriptions",
        "adjustment": "Adjustments & Calibration",
        "parts-list": "Parts Lists & Locations",
        "exploded-view": "Exploded Views & Disassembly",
        "general": "Other Sections",
    }

    lines.append("## Table of Contents")
    lines.append("")

    # Flat TOC (page order)
    for fname, title, pr, stype, asm, images in section_info:
        badge = ""
        if asm:
            badge = " `{}`".format(asm)
        lines.append("- [{}]({}) (pages {}){}".format(title, fname, pr, badge))
    lines.append("")

    # Cross-reference: assembly → schematic + parts list
    asm_sections = defaultdict(list)
    for fname, title, pr, stype, asm, images in section_info:
        if asm:
            asm_sections[asm].append((fname, title, stype, images))

    if asm_sections:
        lines.append("## Assembly Cross-Reference")
        lines.append("")
        for code, entries in asm_sections.items():
            lines.append("### `{}`".format(code))
            for fname, title, stype, images in entries:
                lines.append("- [{}]({}) ({})".format(title, fname, stype))
                for img in images:
                    lines.append("  - Image: `{}`".format(img))
            lines.append("")

    (out_dir / "_index.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def convert_pdf(pdf_path: str, output_dir: Optional[str] = None):
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        print("Error: File not found: {}".format(pdf_path), file=sys.stderr)
        sys.exit(1)

    manual_name = pdf_path.stem
    safe_name = slugify(manual_name) or "manual"

    if output_dir:
        out = Path(output_dir).expanduser().resolve() / safe_name
    else:
        out = DEFAULT_OUTPUT_DIR / safe_name
    out.mkdir(parents=True, exist_ok=True)

    print("Opening: {}".format(pdf_path))
    doc = fitz.open(str(pdf_path))
    print("Pages: {}".format(doc.page_count))

    # Detect watermarks
    watermarks = find_watermarks(doc)
    if watermarks:
        print("Watermarks detected ({} patterns), filtering...".format(len(watermarks)))

    # Classify pages
    page_classes = classify_pages(doc, watermarks)
    text_count = page_classes.count("text")
    image_count = page_classes.count("image")
    print("Page types: {} text, {} diagram/image".format(text_count, image_count))

    # Detect sections
    sections = sections_from_toc_links(doc, watermarks)
    if sections:
        print("Sections from TOC links: {}".format(len(sections)))
    else:
        sections = sections_from_outline(doc, watermarks)
        if sections:
            print("Sections from PDF outline: {}".format(len(sections)))
        else:
            sections = sections_from_fonts(doc, watermarks)
            if sections:
                print("Sections from font analysis: {}".format(len(sections)))
            elif _is_sparse_text(page_classes):
                print("Image-only PDF detected — attempting vision analysis...")
                sections = sections_from_vision(doc, watermarks)
                if sections:
                    print("Sections from vision analysis: {}".format(len(sections)))
                else:
                    print("Vision analysis unavailable/failed, using page-based split...")
                    sections = sections_fallback(doc, watermarks)
            else:
                print("No sections detected, using page-based split...")
                sections = sections_fallback(doc, watermarks)

    # Classify and write sections
    section_info = []
    for i, section in enumerate(sections, start=1):
        # Prefer the section_type assigned by vision analysis when available
        stype = section.get("vision_type") or classify_section(section["title"])
        info = write_section(section, i, out, page_classes, doc, stype)
        section_info.append(info)
        fname, title, pr, stype, asm, imgs = info
        asm_tag = " [{}]".format(asm) if asm else ""
        print("  [{:02d}] {} (pages {}) <{}>{}".format(i, title, pr, stype, asm_tag))

    stats = {"total": doc.page_count, "text": text_count, "image": image_count}
    write_index(manual_name, section_info, out, stats)

    print("\nOutput: {}".format(out))
    print("Index:  {}".format(out / "_index.md"))
    doc.close()


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF service manuals to structured markdown for Claude Code."
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory (default: {})".format(DEFAULT_OUTPUT_DIR),
    )
    args = parser.parse_args()
    convert_pdf(args.pdf, args.output_dir)


if __name__ == "__main__":
    main()
