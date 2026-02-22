"""MCP server for service manual extraction and reading.

Tools:
  - extract_manual: Convert a local PDF to structured markdown + images
  - list_manuals: List already-extracted manuals
  - search_service_manual: Search the web for a service manual PDF
  - read_index: Read a manual's _index.md (TOC, metadata, cross-reference)
  - read_section: Read a section's markdown + inline schematic images
  - get_schematic: Get all schematic images + parts list for a board assembly
"""

import base64
import re
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MANUALS_DIR = Path.home() / "Claude-Manuals"
CONVERT_SCRIPT = Path(__file__).resolve().parent.parent / "convert.py"

mcp = FastMCP("service-manual-reader")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_manual_dir(manual_name: str) -> Path | None:
    """Resolve a manual name to its directory, with fuzzy matching."""
    if not MANUALS_DIR.exists():
        return None

    # Exact match
    exact = MANUALS_DIR / manual_name
    if exact.is_dir():
        return exact

    # Case-insensitive / partial match
    lower = manual_name.lower().replace(" ", "-")
    for d in MANUALS_DIR.iterdir():
        if d.is_dir() and (d.name.lower() == lower or lower in d.name.lower()):
            return d
    return None


def _encode_image(path: Path, max_bytes: int = 700_000) -> ImageContent:
    """Read an image, compress to JPEG, and resize to fit within max_bytes of base64 output."""
    img = PILImage.open(path)

    # Convert to mode compatible with JPEG
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")

    # base64 is ~4/3 of raw bytes
    max_raw = int(max_bytes / 1.34)

    # Try progressively smaller sizes until it fits
    for max_dim in [1400, 1100, 800, 600]:
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            resized = img.resize(
                (int(img.width * ratio), int(img.height * ratio)),
                PILImage.LANCZOS,
            )
        else:
            resized = img

        buf = BytesIO()
        resized.save(buf, format="JPEG", quality=80, optimize=True)
        if buf.tell() <= max_raw:
            data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
            return ImageContent(type="image", data=data, mimeType="image/jpeg")

    # Final fallback: very small
    ratio = 400 / max(img.size)
    resized = img.resize(
        (int(img.width * ratio), int(img.height * ratio)),
        PILImage.LANCZOS,
    )
    buf = BytesIO()
    resized.save(buf, format="JPEG", quality=60, optimize=True)
    data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return ImageContent(type="image", data=data, mimeType="image/jpeg")


def _find_section_file(manual_dir: Path, section: str) -> Path | None:
    """Find a section file by number (e.g. '37') or title substring."""
    section = section.strip()

    # Try numeric prefix match: "37" → "37-*.md"
    if section.isdigit():
        prefix = "{:02d}-".format(int(section))
        for f in manual_dir.glob("{}*.md".format(prefix)):
            return f

    # Try substring match on filename
    lower = section.lower().replace(" ", "-")
    for f in sorted(manual_dir.glob("*.md")):
        if f.name == "_index.md":
            continue
        if lower in f.name.lower():
            return f

    return None


def _extract_image_refs(md_text: str) -> list[str]:
    """Extract image filenames from markdown ![alt](filename.png) references."""
    return re.findall(r"!\[[^\]]*\]\(([^)]+\.png)\)", md_text)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def extract_manual(pdf_path: str) -> str:
    """Extract a PDF service manual into structured markdown and schematic images.

    Converts the PDF into section-based markdown files with embedded PNG
    schematics, parts list tables, and an indexed table of contents.
    Output goes to ~/Claude-Manuals/<manual-name>/.

    Args:
        pdf_path: Absolute path to the PDF file on disk.
    """
    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.exists():
        return "Error: File not found: {}".format(pdf)
    if not pdf.suffix.lower() == ".pdf":
        return "Error: Not a PDF file: {}".format(pdf)

    MANUALS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [sys.executable, str(CONVERT_SCRIPT), str(pdf), "--output-dir", str(MANUALS_DIR)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            return "Extraction failed:\n{}".format(result.stderr)
        return result.stdout
    except subprocess.TimeoutExpired:
        return "Error: Extraction timed out after 5 minutes."


@mcp.tool()
def list_manuals() -> str:
    """List all previously extracted service manuals in ~/Claude-Manuals/.

    Returns the manual names and their _index.md paths so you can read them.
    """
    if not MANUALS_DIR.exists():
        return "No manuals directory found. Extract a manual first."

    manuals = []
    for d in sorted(MANUALS_DIR.iterdir()):
        if d.is_dir():
            index = d / "_index.md"
            if index.exists():
                manuals.append("- {} → {}".format(d.name, index))
            else:
                manuals.append("- {} (no index)".format(d.name))

    if not manuals:
        return "No extracted manuals found in {}".format(MANUALS_DIR)

    return "Extracted manuals:\n" + "\n".join(manuals)


@mcp.tool()
def read_index(manual_name: str) -> str:
    """Read the full _index.md for a manual — TOC, metadata, and assembly cross-reference.

    This is the first thing to read when working with a manual. It gives you
    the complete table of contents, board assembly list, and cross-references
    between schematics and parts lists.

    Args:
        manual_name: Manual directory name (e.g. 'sx-750-service-manual').
                     Partial matches and spaces are accepted.
    """
    manual_dir = _find_manual_dir(manual_name)
    if not manual_dir:
        return "Manual '{}' not found. Use list_manuals to see available manuals.".format(manual_name)

    index = manual_dir / "_index.md"
    if not index.exists():
        return "No _index.md found in {}".format(manual_dir)

    return index.read_text(encoding="utf-8")


@mcp.tool()
def read_section(manual_name: str, section: str, text_only: bool = False) -> list:
    """Read a section's markdown text and optionally return associated schematic/diagram images.

    When text_only=True, returns just the markdown text (fast, no size limit issues).
    When text_only=False, images are compressed to JPEG and included inline.

    Args:
        manual_name: Manual directory name (e.g. 'sx-750-service-manual').
        section: Section number (e.g. '37') or title substring
                 (e.g. 'power amplifier assembly', 'adjustments').
        text_only: If True, return only the markdown text without images.
    """
    manual_dir = _find_manual_dir(manual_name)
    if not manual_dir:
        return [TextContent(
            type="text",
            text="Manual '{}' not found. Use list_manuals to see available manuals.".format(manual_name),
        )]

    section_file = _find_section_file(manual_dir, section)
    if not section_file:
        return [TextContent(
            type="text",
            text="Section '{}' not found in {}. Use read_index to see available sections.".format(
                section, manual_dir.name,
            ),
        )]

    md_text = section_file.read_text(encoding="utf-8")
    image_refs = _extract_image_refs(md_text)

    if text_only:
        note = "\n\n---\n{} image(s) not shown (text_only=True). Set text_only=False to include images.".format(
            len(image_refs),
        ) if image_refs else ""
        return [TextContent(type="text", text=md_text + note)]

    result = [TextContent(type="text", text=md_text)]

    # Calculate per-image budget to stay under ~950KB total
    text_size = len(md_text.encode("utf-8"))
    remaining = 950_000 - text_size
    num_images = sum(1 for img in image_refs if (manual_dir / img).exists())
    per_image = max(remaining // max(num_images, 1), 80_000)

    for img_name in image_refs:
        img_path = manual_dir / img_name
        if img_path.exists():
            result.append(_encode_image(img_path, max_bytes=per_image))

    return result


@mcp.tool()
def get_schematic(manual_name: str, board_id: str, page: int = 0) -> list:
    """Get schematic images and parts list for a board assembly.

    Without a page number, returns the markdown text and lists available images.
    With page=N, returns that specific image (compressed JPEG) plus the text.

    Call without page first to see what's available, then request specific pages.

    Args:
        manual_name: Manual directory name (e.g. 'sx-750-service-manual').
        board_id: Board assembly ID (e.g. 'AWH-046', 'AWR-099', 'AWE-073').
        page: Image page number (1-based). 0 = text only with image listing.
    """
    manual_dir = _find_manual_dir(manual_name)
    if not manual_dir:
        return [TextContent(
            type="text",
            text="Manual '{}' not found. Use list_manuals to see available manuals.".format(manual_name),
        )]

    board_id_upper = board_id.upper().strip()

    # Find all section files that reference this board assembly
    schematic_files = []
    parts_list_files = []

    for f in sorted(manual_dir.glob("*.md")):
        if f.name == "_index.md":
            continue
        text = f.read_text(encoding="utf-8")
        if board_id_upper not in text.upper():
            continue

        if re.search(r"parts.list", f.name, re.IGNORECASE):
            parts_list_files.append((f, text))
        else:
            schematic_files.append((f, text))

    if not schematic_files and not parts_list_files:
        return [TextContent(
            type="text",
            text="No sections found for board assembly '{}'. Check the board ID in read_index.".format(board_id),
        )]

    # Collect all images in order with metadata
    all_images = []
    for f, text in schematic_files:
        for img_name in _extract_image_refs(text):
            img_path = manual_dir / img_name
            if img_path.exists():
                all_images.append(("schematic", img_name, img_path))

    for f, text in parts_list_files:
        for img_name in _extract_image_refs(text):
            img_path = manual_dir / img_name
            if img_path.exists():
                all_images.append(("parts-list", img_name, img_path))

    result = []

    # Build text content with section markdown
    for f, text in schematic_files:
        result.append(TextContent(type="text", text="## Schematic: {}\n\n{}".format(f.stem, text)))
    for f, text in parts_list_files:
        result.append(TextContent(type="text", text="## Parts List: {}\n\n{}".format(f.stem, text)))

    # Add image index listing
    if all_images:
        listing = "\n---\n**Available images for {} ({} total):**\n".format(board_id, len(all_images))
        for i, (kind, name, _) in enumerate(all_images, 1):
            listing += "  page={}: {} ({})\n".format(i, name, kind)
        if page == 0:
            listing += "\nUse page=N to view a specific image."
        result.append(TextContent(type="text", text=listing))

    # If page requested, include that specific image
    if page > 0:
        if page > len(all_images):
            result.append(TextContent(
                type="text",
                text="Page {} not found. Valid range: 1-{}.".format(page, len(all_images)),
            ))
        else:
            _, img_name, img_path = all_images[page - 1]
            result.append(_encode_image(img_path, max_bytes=800_000))

    return result


@mcp.tool()
async def search_service_manual(brand: str, model: str) -> str:
    """Search the web for a service manual PDF given a brand and model.

    Returns download links and sources found for the service manual.
    Use extract_manual afterwards to convert a downloaded PDF.

    Args:
        brand: Equipment brand/manufacturer (e.g. Pioneer, Marantz, Sony).
        model: Model number (e.g. SX-750, 2270, STR-6800SD).
    """
    query = "{} {} service manual PDF filetype:pdf".format(brand, model)
    alt_query = "{} {} service manual download".format(brand, model)

    results = []

    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
        for q in [query, alt_query]:
            try:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": q},
                    headers={"User-Agent": "ServiceManualReader/1.0"},
                )
                if resp.status_code == 200:
                    links = re.findall(
                        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.+?)</a>',
                        resp.text,
                    )
                    for url, title in links[:10]:
                        actual = re.search(r"uddg=([^&]+)", url)
                        if actual:
                            from urllib.parse import unquote
                            url = unquote(actual.group(1))
                        title = re.sub(r"<[^>]+>", "", title).strip()
                        if title and url:
                            results.append((title, url))
            except Exception:
                pass

    if not results:
        return (
            "No results found. Try searching manually for:\n"
            '  "{} {} service manual PDF"'.format(brand, model)
        )

    seen = set()
    unique = []
    for title, url in results:
        if url not in seen:
            seen.add(url)
            unique.append((title, url))

    lines = [
        "Search results for {} {} service manual:".format(brand, model),
        "",
    ]
    for i, (title, url) in enumerate(unique[:10], 1):
        lines.append("{}. {}".format(i, title))
        lines.append("   {}".format(url))
        lines.append("")

    lines.append(
        "Download the PDF, then use extract_manual with the file path to convert it."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
