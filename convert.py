#!/usr/bin/env python3
"""Convert PDF service manuals into structured markdown files for Claude Code.

Produces:
  - Markdown per section with hierarchy, parts tables, adjustment steps
  - PNG schematics named by assembly (e.g., awh-046-schematic.png)
  - Cross-referenced index with metadata
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

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
    """Extract assembly/board code like AWH-046 from a section title."""
    m = re.search(r"\(([A-Z]{2,4}-\d{2,4})\)", title)
    return m.group(1) if m else None


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
    pix.save(str(out_dir / filename))
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
    assembly = extract_assembly_code(title)

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

        if is_image:
            img_name = make_image_name(slug, page_num, section_type, page_idx)
            img_file = extract_page_image(doc, page_num, out_dir, img_name, dpi)
            image_files.append(img_file)

            parts.append("---")
            parts.append("*Page {}*\n".format(page_num + 1))
            parts.append("![{} — page {}]({})".format(title, page_num + 1, img_file))
            if cleaned.strip():
                parts.append("")
                parts.append(cleaned)
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
        out = Path(output_dir).resolve() / safe_name
    else:
        out = pdf_path.parent / "manuals" / safe_name
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
            else:
                print("No sections detected, using page-based split...")
                sections = sections_fallback(doc, watermarks)

    # Classify and write sections
    section_info = []
    for i, section in enumerate(sections, start=1):
        stype = classify_section(section["title"])
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
        help="Output directory (default: ./manuals/ next to the PDF)",
    )
    args = parser.parse_args()
    convert_pdf(args.pdf, args.output_dir)


if __name__ == "__main__":
    main()
