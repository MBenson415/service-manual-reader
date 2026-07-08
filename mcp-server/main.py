"""MCP server for service manual extraction and reading.

Tools:
  - extract_manual: Convert a local PDF to structured markdown + images
  - list_manuals: List already-extracted manuals
    - list_downloads: List downloaded PDFs in the Schematics directory
  - search_service_manual: Search the web for a service manual PDF
    - download_service_manual: Download a manual PDF from a direct URL
  - read_index: Read a manual's _index.md (TOC, metadata, cross-reference)
  - read_section: Read a section's markdown + inline schematic images
  - get_schematic: Get all schematic images + parts list for a board assembly
  - analyze_schematic: Use Claude vision to interpret a schematic image
  - cross_check_schematic: Compare schematic vs parts list to find unreadable components
  - generate_netlist: Trace connections on a schematic and produce a SPICE netlist
"""

import base64
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from urllib.parse import unquote, urlparse
from io import BytesIO
from pathlib import Path

import anthropic
import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MANUALS_DIR = Path.home() / "Claude-Manuals"
SCHEMATICS_DIR = Path("/Users/marshallbenson/Desktop/Benchmark Audio Repair/Schematics")
CONVERT_SCRIPT = Path(__file__).resolve().parent.parent / "convert.py"

mcp = FastMCP("service-manual-reader")

_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}

_MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024  # 200MB cap for safety


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


def _resolve_schematic_image(manual_name: str, board_id: str, page: int) -> tuple[Path | None, str]:
    """Resolve a schematic image path from manual_name, board_id, and page number.

    Returns (image_path, error_message). If image_path is None, error_message
    explains what went wrong.
    """
    manual_dir = _find_manual_dir(manual_name)
    if not manual_dir:
        return None, "Manual '{}' not found. Use list_manuals to see available manuals.".format(manual_name)

    board_id_upper = board_id.upper().strip()

    # Collect all images for this board assembly across section files
    all_images = []
    for f in sorted(manual_dir.glob("*.md")):
        if f.name == "_index.md":
            continue
        text = f.read_text(encoding="utf-8")
        if board_id_upper not in text.upper():
            continue
        kind = "parts-list" if re.search(r"parts.list", f.name, re.IGNORECASE) else "schematic"
        for img_name in _extract_image_refs(text):
            img_path = manual_dir / img_name
            if img_path.exists():
                all_images.append((kind, img_name, img_path))

    if not all_images:
        return None, "No images found for board assembly '{}'. Check the board ID in read_index.".format(board_id)

    if page < 1 or page > len(all_images):
        listing = "Available images for {} ({} total):\n".format(board_id, len(all_images))
        for i, (kind, name, _) in enumerate(all_images, 1):
            listing += "  page={}: {} ({})\n".format(i, name, kind)
        return None, "Page {} out of range (1-{}).\n\n{}".format(page, len(all_images), listing)

    _, _, img_path = all_images[page - 1]
    return img_path, ""


def _safe_filename(board_id: str) -> str:
    """Convert a board ID to a filesystem-safe filename (no extension).

    Matches the sanitisation in Swift CircuitCacheService.safeFilename(for:).
    IDs like 'HT3035:B / 25C335B' become 'HT3035-B-25C335B'.
    """
    safe = board_id
    for ch in "/:\\ ":
        safe = safe.replace(ch, "-")
    while "--" in safe:
        safe = safe.replace("--", "-")
    safe = safe.strip("-")
    return safe if safe else "unknown"


def _sanitize_name(text: str) -> str:
    """Make user/input-derived strings safe for folder/file names."""
    cleaned = text.strip()
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" .")


def _guess_download_filename(url: str, brand: str, model: str, filename: str) -> str:
    """Choose a safe PDF filename for a downloaded manual."""
    if filename.strip():
        base = _sanitize_name(filename)
    else:
        path_name = Path(unquote(urlparse(url).path)).name
        if path_name.lower().endswith(".pdf"):
            base = _sanitize_name(path_name)
        elif brand.strip() or model.strip():
            base = _sanitize_name("{} {} Service Manual.pdf".format(brand.strip(), model.strip()))
        else:
            base = "service-manual.pdf"

    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    return base


def _load_circuit_json(manual_dir: Path, board_id: str) -> dict | None:
    """Load _circuits/<board_id>.json if it exists."""
    circuits_dir = manual_dir / "_circuits"
    if not circuits_dir.is_dir():
        return None
    # Try sanitised filename first, then case-insensitive scan
    safe = _safe_filename(board_id.upper())
    target = circuits_dir / "{}.json".format(safe)
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    for f in circuits_dir.glob("*.json"):
        if f.stem.upper() == board_id.upper() or f.stem.upper() == safe.upper():
            return json.loads(f.read_text(encoding="utf-8"))
    return None


def _build_theory_context(circuit_data: dict) -> str:
    """Format circuit JSON into a concise text block for the vision prompt."""
    lines = []

    components = circuit_data.get("components", [])
    if components:
        lines.append("COMPONENTS ({}):".format(len(components)))
        for c in components:
            pins = ", ".join(p["id"] for p in c.get("pins", []))
            value = " {}".format(c["value"]) if c.get("value") else ""
            desc = " ({})".format(c["description"]) if c.get("description") else ""
            lines.append("  {}: {}{}{}  [pins: {}]".format(
                c["designator"], c.get("type", "unknown"), value, desc, pins,
            ))

    blocks = circuit_data.get("functionalBlocks", [])
    if blocks:
        lines.append("")
        lines.append("FUNCTIONAL BLOCKS:")
        for b in blocks:
            ids = ", ".join(b.get("componentIDs", []))
            lines.append("  {}: {}".format(b["name"], ids))
            if b.get("description"):
                lines.append("    -> {}".format(b["description"]))

    if circuit_data.get("description"):
        lines.append("")
        lines.append("CIRCUIT DESCRIPTION: {}".format(circuit_data["description"]))

    return "\n".join(lines)


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
def download_service_manual(url: str, brand: str = "", model: str = "", filename: str = "") -> str:
    """Download a service manual PDF from a direct URL.

    Saves the file to /Users/marshallbenson/Desktop/Benchmark Audio Repair/
    Schematics/<Brand Model>/ and verifies that the downloaded content is
    actually a PDF before keeping it.

    Args:
        url: Direct URL to a PDF file.
        brand: Optional manufacturer name (used for default filename).
        model: Optional model number (used for default filename).
        filename: Optional explicit output filename.
    """
    if not re.match(r"^https?://", url.strip(), re.IGNORECASE):
        return "Error: URL must start with http:// or https://"

    if not brand.strip() or not model.strip():
        return "Error: Both brand and model are required. Folder format is '<Brand> <Model>'."

    SCHEMATICS_DIR.mkdir(parents=True, exist_ok=True)

    folder_name = _sanitize_name("{} {}".format(brand.strip(), model.strip()))
    target_folder = SCHEMATICS_DIR / folder_name
    target_folder.mkdir(parents=True, exist_ok=True)

    output_name = _guess_download_filename(url, brand, model, filename)
    dest = target_folder / output_name

    if dest.exists():
        return "Already exists, not overwriting: {}".format(dest)

    total = 0
    try:
        with httpx.Client(follow_redirects=True, timeout=60.0, headers=_DOWNLOAD_HEADERS) as client:
            with client.stream("GET", url) as response:
                if response.status_code >= 400:
                    return "Download failed: HTTP {}".format(response.status_code)

                content_type = response.headers.get("Content-Type", "")

                first_chunk = b""
                stream = response.iter_bytes(chunk_size=262_144)
                for chunk in stream:
                    first_chunk = chunk
                    break

                if not first_chunk:
                    return "Download failed: empty response body"

                # Require PDF magic bytes in first chunk to avoid saving HTML/login pages.
                if b"%PDF" not in first_chunk[:1024]:
                    return (
                        "Not a PDF (Content-Type: {}). The URL may be a landing page "
                        "rather than a direct PDF link."
                    ).format(content_type or "unknown")

                with open(dest, "wb") as f:
                    f.write(first_chunk)
                    total += len(first_chunk)

                    for chunk in stream:
                        total += len(chunk)
                        if total > _MAX_DOWNLOAD_BYTES:
                            f.close()
                            dest.unlink(missing_ok=True)
                            return "Aborted: file exceeded {} MB cap.".format(_MAX_DOWNLOAD_BYTES // 1024 // 1024)
                        f.write(chunk)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        return "Download failed: {}: {}".format(type(exc).__name__, exc)

    size_mb = total / 1024 / 1024
    return (
        "Saved {:.1f} MB to {}\n"
        "Next step: run extract_manual(pdf_path='{}')"
    ).format(size_mb, dest, dest)


@mcp.tool()
def list_downloads() -> str:
    """List downloaded service manual PDFs in the Schematics directory."""
    if not SCHEMATICS_DIR.exists():
        return "Schematics directory not found yet. Use download_service_manual first."

    pdfs = sorted(SCHEMATICS_DIR.glob("**/*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pdfs:
        return "No downloaded PDFs found in {}".format(SCHEMATICS_DIR)

    lines = ["Downloaded PDFs:"]
    for p in pdfs:
        stat = p.stat()
        size_mb = stat.st_size / 1024 / 1024
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        try:
            rel = p.relative_to(SCHEMATICS_DIR)
        except ValueError:
            rel = p
        lines.append("- {} ({:.1f} MB, modified {})".format(rel, size_mb, modified))

    return "\n".join(lines)


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


VISION_MODEL = "claude-haiku-4-5-20251001"
VISION_MAX_PX = 2000
VISION_QUALITY = 85
COVERAGE_TARGET = 95  # % coverage before we stop cropping
CROP_GRIDS = [(2, 2), (3, 3)]  # progressive tile grids for higher-res passes

_ANALYZE_PROMPT = """\
You are an expert electronics technician analyzing a service manual schematic.
Examine this schematic image and provide a structured analysis:

1. **Circuit Overview**: What type of circuit is this? (e.g. power amplifier, \
tone control, power supply, tuner, protection circuit)

2. **Key Components**: List the major active components you can identify \
(transistors, ICs, op-amps) with their designators (e.g. Q101, IC201) and \
probable function in the circuit.

3. **Signal Path**: Describe the main signal flow through the circuit from \
input to output.

4. **Power Rails**: Identify supply voltages visible on the schematic and \
which sections they feed.

5. **Notable Design Features**: Any interesting design choices — feedback \
networks, protection circuits, bias arrangements, unusual topologies.

Be specific — reference actual component designators and values visible in \
the schematic. If parts are hard to read, note what you can make out and \
flag uncertainty. Keep the analysis concise and practical for a technician \
doing repair work."""


def _image_to_base64(img_path: Path) -> tuple[str, int, int]:
    """Load an image, resize for vision API, return (base64_data, width, height)."""
    img = PILImage.open(img_path)
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")

    # Resize if too large for the vision API
    if max(img.size) > VISION_MAX_PX:
        ratio = VISION_MAX_PX / max(img.size)
        img = img.resize(
            (int(img.width * ratio), int(img.height * ratio)),
            PILImage.LANCZOS,
        )

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=VISION_QUALITY)
    data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return data, img.width, img.height


def _crop_image_tiles(img_path: Path, grid: tuple[int, int]) -> list[tuple[str, str]]:
    """Split an image into tiles at full resolution for detailed scanning.

    Returns list of (base64_data, media_type) tuples, one per tile.
    Tiles are capped at VISION_MAX_PX per side for the API.
    """
    img = PILImage.open(img_path)
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")

    rows, cols = grid
    tile_w = img.width // cols
    tile_h = img.height // rows

    tiles = []
    for r in range(rows):
        for c in range(cols):
            left = c * tile_w
            top = r * tile_h
            right = min((c + 1) * tile_w, img.width)
            bottom = min((r + 1) * tile_h, img.height)
            tile = img.crop((left, top, right, bottom))

            if max(tile.size) > VISION_MAX_PX:
                ratio = VISION_MAX_PX / max(tile.size)
                tile = tile.resize(
                    (int(tile.width * ratio), int(tile.height * ratio)),
                    PILImage.LANCZOS,
                )

            buf = BytesIO()
            tile.save(buf, format="JPEG", quality=VISION_QUALITY)
            data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
            tiles.append((data, "image/jpeg"))

    return tiles


_TARGETED_DESIGNATOR_PROMPT = """\
Look carefully at this schematic crop for these specific component designators:
{}

List every one from the target list above that you can find, one per line.
If a designator is partially readable, include your best guess with a ? suffix.
Only list designators from the target list — ignore everything else.
If none are found, respond with: NONE_FOUND"""

_NETLIST_EXTRACTION_PROMPT = """\
You are an expert electronics technician generating a SPICE netlist from this \
schematic image. Trace every wire carefully and identify how each component \
pin connects to the circuit.

Use this circuit theory to understand the components and their roles:

{}

CRITICAL RULES FOR TRACING CONNECTIONS:
- Follow each wire from a component pin to its destination node or junction.
- Each pin of a 2-terminal component (resistor, capacitor) connects to a \
DIFFERENT node. A resistor always bridges two distinct nets.
- For transistors: collector, base, and emitter each connect to different nodes.
- When a wire runs from one component pin to another component pin, both pins \
share the SAME node name. This is how you build nets.
- Look for labeled voltage rails (VCC, VEE, +33V, -33V, GND) and use those \
exact names. Trace rails across the full width of the schematic.
- For unlabeled junctions where wires meet, assign node names (n1, n2, n3...). \
REUSE the same name for all pins that connect at that junction.
- A bias resistor typically has one end on a drive/signal rail and the other \
end on a transistor base — the two ends go to DIFFERENT nodes.

FORMAT — one component per line, nothing else:
  Resistors:     R807 node_end1 node_end2 820
  Capacitors:    C807 node1 node2 470u
  Electrolytic:  C809 node_plus node_minus 2200u
  NPN transistor: Q_HB01 collector_node base_node emitter_node NPN
  PNP transistor: Q_HB05 collector_node base_node emitter_node PNP
  Diode:         D801 anode_node cathode_node D
  Connector:     X_J801 pin1_node pin2_node CONN
  Inductor:      L801 node1 node2 value

Start with a comment line: * SPICE Netlist: <board name>
End with: .END

If a connection is uncertain, append a ? to the node name (e.g. n5?).
If you cannot determine a component's connections at all, write a comment: \
* UNRESOLVED: R816 — not enough detail visible"""

_TARGETED_NET_PROMPT = """\
Look carefully at this schematic crop. I need SPICE netlist lines for these \
specific components whose connections are incomplete:
{}

For each one you can see, trace its wire connections and write the SPICE line.

FORMAT — one component per line:
  Resistors:     R807 node1 node2 820
  Capacitors:    C807 node1 node2 470u
  NPN transistor: Q_HB01 collector base emitter NPN
  PNP transistor: Q_HB05 collector base emitter PNP
  Connector:     X_J801 pin1_node pin2_node CONN

Use named nets for labeled nodes (VCC, GND, signal names). Use n1, n2... for \
unlabeled junctions. Append ? to uncertain node names.
If none of the target components are visible, respond with: NONE_FOUND"""


@mcp.tool()
def analyze_schematic(manual_name: str, board_id: str, page: int) -> str:
    """Analyze a schematic image using Claude vision to identify components and trace signal paths.

    Sends the schematic to Claude's vision model for expert interpretation.
    Returns a structured analysis including circuit overview, key components,
    signal path, power rails, and notable design features.

    Requires ANTHROPIC_API_KEY environment variable.

    Args:
        manual_name: Manual directory name (e.g. 'sx-750-service-manual').
                     Partial matches and spaces are accepted.
        board_id: Board assembly ID (e.g. 'AWH-046', 'AWR-099', 'AWE-073').
        page: Image page number (1-based). Use get_schematic with page=0 to
              see available images first.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not set. Required for vision analysis."

    img_path, error = _resolve_schematic_image(manual_name, board_id, page)
    if img_path is None:
        return error

    try:
        img_data, w, h = _image_to_base64(img_path)
    except Exception as exc:
        return "Error loading image {}: {}".format(img_path.name, exc)

    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=VISION_MODEL,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_data,
                        },
                    },
                    {"type": "text", "text": _ANALYZE_PROMPT},
                ],
            }],
        )
    except Exception as exc:
        return "Vision API error: {}".format(exc)

    analysis = response.content[0].text

    header = "# Schematic Analysis: {} (board {}, page {})\n".format(
        img_path.name, board_id, page,
    )
    header += "Image: {} x {} px | Model: {}\n\n".format(w, h, VISION_MODEL)

    return header + analysis


_EXTRACT_PARTS_LIST_PROMPT = """\
Extract component designators from this parts list page for board "{}" ONLY.

Look for a section header matching the board ID (e.g. "P800 POWER SUPPLY BOARD").
If this page does not contain parts for that board, respond with just: NO_MATCH

IMPORTANT: Parts list pages often contain multiple board sections. STOP reading \
when you reach a different board's header or a new section (e.g. a line like \
"MISCELLANEOUS PARTS", another board ID, or a clearly different numbering series). \
Only include designators that belong to the "{}" board — these typically share \
the board's numbering series (e.g. P800 components use 8xx numbers like R801, C802).

Return one designator per line, nothing else. Example:
R801
R802
C801
Q801
D801"""

_EXTRACT_SCHEMATIC_DESIGNATORS_PROMPT = """\
List every component designator you can read on this schematic image.
Include resistors (R), capacitors (C), transistors (Q), diodes (D), \
inductors (L), ICs (IC), transformers (T), connectors (J/P), fuses (F), \
and any other components with visible reference designators.

Return one designator per line, nothing else. If a designator is partially \
readable, include your best guess with a ? suffix (e.g. R80? or C8??).
Example:
R801
R802
C801
Q801
D80?"""


def _find_parts_list_images(manual_dir: Path) -> list[Path]:
    """Find all parts list PNG images in a manual directory."""
    images = []
    for f in sorted(manual_dir.glob("*.md")):
        if f.name == "_index.md":
            continue
        if not re.search(r"parts.list", f.name, re.IGNORECASE):
            continue
        text = f.read_text(encoding="utf-8")
        for img_name in _extract_image_refs(text):
            img_path = manual_dir / img_name
            if img_path.exists():
                images.append(img_path)
    return images


def _vision_call(client: anthropic.Anthropic, images: list[tuple[str, str]], prompt: str, max_tokens: int = 4096) -> str:
    """Make a vision API call with one or more images and a text prompt.

    images: list of (base64_data, media_type) tuples.
    """
    content = []
    for data, media_type in images:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        })
    content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model=VISION_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text


def _parse_designator_lines(text: str) -> tuple[set[str], set[str]]:
    """Parse vision output into confirmed and uncertain designator sets."""
    confirmed = set()
    uncertain = set()
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("*") or "NONE_FOUND" in line:
            continue
        upper = line.upper()
        if not re.match(r"^[A-Z]+\d", upper):
            continue
        if "?" in upper:
            uncertain.add(upper)
        else:
            confirmed.add(upper)
    return confirmed, uncertain


def _compute_coverage(parts: set, found: set) -> float:
    """Return coverage percentage."""
    return len(parts & found) / len(parts) * 100 if parts else 100.0


# SPICE netlist line patterns
_SPICE_2TERM = re.compile(
    r"^([A-Z]\w*)\s+(\S+)\s+(\S+)\s+(\S+)$", re.IGNORECASE,
)
_SPICE_3TERM = re.compile(
    r"^(Q\w*)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$", re.IGNORECASE,
)


def _parse_spice_lines(text: str) -> tuple[list[dict], set[str]]:
    """Parse SPICE netlist text from vision output.

    Returns (entries, netted_designators) where each entry is a dict:
      {"raw": str, "designator": str, "nodes": list[str], "value": str}
    """
    entries = []
    netted = set()

    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("*") or line.startswith(".") or "NONE_FOUND" in line:
            continue

        # Try 3-terminal (transistor): Q_HB01 collector base emitter NPN
        m3 = _SPICE_3TERM.match(line)
        if m3:
            desig = m3.group(1).upper()
            nodes = [m3.group(2), m3.group(3), m3.group(4)]
            value = m3.group(5)
            # Strip Q_ prefix to get bare designator for coverage tracking
            bare = re.sub(r"^Q_", "", desig)
            entries.append({"raw": line, "designator": desig, "nodes": nodes, "value": value})
            netted.add(bare)
            continue

        # Try 2-terminal: R807 node1 node2 820
        m2 = _SPICE_2TERM.match(line)
        if m2:
            desig = m2.group(1).upper()
            nodes = [m2.group(2), m2.group(3)]
            value = m2.group(4)
            # Strip X_ prefix for connectors
            bare = re.sub(r"^X_", "", desig)
            entries.append({"raw": line, "designator": desig, "nodes": nodes, "value": value})
            netted.add(bare)
            continue

    return entries, netted


def _spice_entries_to_standalone_json(entries: list[dict], board_id: str, manual_name: str) -> dict:
    """Build a complete circuit JSON from SPICE entries alone (no pre-existing circuit data).

    Creates component definitions from the SPICE entries and builds a net list.
    """
    # Infer component type from designator prefix
    PREFIX_TYPE = {
        "R": "resistor", "C": "capacitor", "L": "inductor",
        "D": "diode", "Q": "transistor", "T": "transformer",
        "F": "fuse", "VR": "potentiometer", "TR": "transformer",
    }

    # Build components and node map
    components = {}
    node_map: dict[str, list[tuple[str, str]]] = {}

    for entry in entries:
        bare = re.sub(r"^[QX]_", "", entry["designator"])
        if bare not in components:
            # Determine type from prefix
            prefix = re.match(r"^[A-Z]+", bare)
            comp_type = PREFIX_TYPE.get(prefix.group() if prefix else "", "component")
            pin_ids = [str(i + 1) for i in range(len(entry["nodes"]))]
            if comp_type == "transistor" and len(pin_ids) == 3:
                pin_ids = ["C", "B", "E"]
            components[bare] = {
                "designator": bare,
                "type": comp_type,
                "value": entry.get("value", ""),
                "pins": [{"id": p, "netID": ""} for p in pin_ids],
            }

        comp = components[bare]
        pin_ids = [p["id"] for p in comp["pins"]]
        for i, node_name in enumerate(entry["nodes"]):
            pin_id = pin_ids[i] if i < len(pin_ids) else str(i + 1)
            node_map.setdefault(node_name, []).append((bare, pin_id))
            # Set netID on pin
            if i < len(comp["pins"]):
                comp["pins"][i]["netID"] = node_name

    # Build nets
    nets = []
    for node_name, connections in sorted(node_map.items()):
        nets.append({
            "id": node_name,
            "connectedPins": [
                {"componentID": cid, "pinID": pid}
                for cid, pid in connections
            ],
        })

    return {
        "name": "{} ({})".format(board_id, manual_name),
        "description": "Auto-generated from schematic vision analysis",
        "components": list(components.values()),
        "nets": nets,
        "functionalBlocks": [],
    }


def _spice_to_circuit_json(entries: list[dict], circuit_data: dict) -> dict:
    """Update a circuit JSON's nets array with vision-extracted SPICE data.

    Builds a new nets array from SPICE entries. Preserves existing component
    metadata and functional blocks from circuit_data.
    """
    # Build node -> [(componentID, pinID)] mapping from SPICE entries
    node_map: dict[str, list[tuple[str, str]]] = {}

    # Map of component designators to their pin info from circuit_data
    comp_pins = {}
    for c in circuit_data.get("components", []):
        comp_pins[c["designator"].upper()] = [p["id"] for p in c.get("pins", [])]

    for entry in entries:
        bare = re.sub(r"^[QX]_", "", entry["designator"])
        pins = comp_pins.get(bare, [])

        for i, node_name in enumerate(entry["nodes"]):
            # Determine pin ID — use known pins if available, else positional
            if i < len(pins):
                pin_id = pins[i]
            else:
                pin_id = str(i + 1)

            node_map.setdefault(node_name, []).append((bare, pin_id))

    # Build nets array
    nets = []
    for node_name, connections in sorted(node_map.items()):
        net = {
            "id": node_name,
            "connectedPins": [
                {"componentID": comp_id, "pinID": pin_id}
                for comp_id, pin_id in connections
            ],
        }
        nets.append(net)

    result = dict(circuit_data)
    result["nets"] = nets
    return result


@mcp.tool()
def cross_check_schematic(manual_name: str, board_id: str, parts_list_page: int = 0) -> str:
    """Cross-check a schematic against its parts list to find unreadable components.

    Extracts component designators from the parts list and the schematic using
    vision, compares the two, then automatically crops the schematic into
    higher-resolution tiles and re-scans for any missing designators. Repeats
    with progressively finer grids until coverage reaches 95% or no further
    improvement is found.

    Requires ANTHROPIC_API_KEY environment variable.

    Args:
        manual_name: Manual directory name (e.g. 'pdf-marantz-2250-service-manual').
                     Partial matches and spaces are accepted.
        board_id: Board assembly ID (e.g. 'P800', 'AWH-046').
        parts_list_page: PDF page number of the parts list for this board.
                         0 = auto-detect by scanning all parts list pages.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not set. Required for vision analysis."

    manual_dir = _find_manual_dir(manual_name)
    if not manual_dir:
        return "Manual '{}' not found. Use list_manuals to see available manuals.".format(manual_name)

    client = anthropic.Anthropic(api_key=api_key)
    pass_log = []  # track what happened at each pass

    # --- Step 1: Extract designators from parts list ---
    parts_list_images = _find_parts_list_images(manual_dir)
    if not parts_list_images:
        return "No parts list images found in {}. Cannot cross-check.".format(manual_dir.name)

    if parts_list_page > 0:
        match = [p for p in parts_list_images if "-p{:03d}.png".format(parts_list_page) in p.name]
        if not match:
            available = ", ".join(
                re.search(r"-p(\d+)\.png", p.name).group(1).lstrip("0") for p in parts_list_images
            )
            return "Parts list page {} not found. Available pages: {}".format(
                parts_list_page, available,
            )
        parts_list_images = match

    parts_designators = set()
    matched_pages = []

    for pl_path in parts_list_images:
        try:
            img_data, _, _ = _image_to_base64(pl_path)
        except Exception:
            continue

        prompt = _EXTRACT_PARTS_LIST_PROMPT.format(board_id.upper(), board_id.upper())
        try:
            result = _vision_call(client, [(img_data, "image/jpeg")], prompt, max_tokens=2048)
        except Exception:
            continue

        if "NO_MATCH" in result:
            continue

        matched_pages.append(pl_path.name)
        for line in result.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("*"):
                continue
            upper = line.upper()
            if board_id.upper() in upper and not re.match(r"^[A-Z]\d", upper):
                continue
            if re.match(r"^[A-Z]+\d", upper):
                parts_designators.add(upper)

    if not parts_designators:
        return "Could not find parts for board '{}' in {} parts list page(s). Try specifying parts_list_page.".format(
            board_id, len(parts_list_images),
        )

    # --- Step 2: Collect schematic images for this board ---
    board_id_upper = board_id.upper().strip()
    schematic_images = []
    for f in sorted(manual_dir.glob("*.md")):
        if f.name == "_index.md":
            continue
        text = f.read_text(encoding="utf-8")
        if board_id_upper not in text.upper():
            continue
        if re.search(r"parts.list", f.name, re.IGNORECASE):
            continue
        for img_name in _extract_image_refs(text):
            img_path = manual_dir / img_name
            if img_path.exists():
                schematic_images.append(img_path)

    if not schematic_images:
        return "No schematic images found for board '{}'. Check the board ID.".format(board_id)

    # --- Pass 1: Full image scan ---
    schematic_designators = set()
    uncertain_designators = set()

    for sch_path in schematic_images:
        try:
            img_data, _, _ = _image_to_base64(sch_path)
        except Exception:
            continue
        try:
            result = _vision_call(
                client, [(img_data, "image/jpeg")],
                _EXTRACT_SCHEMATIC_DESIGNATORS_PROMPT, max_tokens=4096,
            )
        except Exception:
            continue
        confirmed, uncertain = _parse_designator_lines(result)
        schematic_designators |= confirmed
        uncertain_designators |= uncertain

    coverage = _compute_coverage(parts_designators, schematic_designators)
    still_missing = parts_designators - schematic_designators
    pass_log.append("Pass 1 (full image): found {}/{}, coverage {:.0f}%".format(
        len(parts_designators & schematic_designators), len(parts_designators), coverage,
    ))

    # --- Pass 2+: Progressive higher-resolution tile crops ---
    for grid in CROP_GRIDS:
        if coverage >= COVERAGE_TARGET or not still_missing:
            break

        newly_found = set()
        grid_label = "{}x{}".format(grid[0], grid[1])

        for sch_path in schematic_images:
            try:
                tiles = _crop_image_tiles(sch_path, grid)
            except Exception:
                continue

            for tile_data in tiles:
                if not still_missing:
                    break

                prompt = _TARGETED_DESIGNATOR_PROMPT.format(
                    "\n".join(sorted(still_missing)),
                )
                try:
                    result = _vision_call(client, [tile_data], prompt, max_tokens=2048)
                except Exception:
                    continue

                confirmed, uncertain = _parse_designator_lines(result)
                for d in confirmed:
                    if d in still_missing:
                        schematic_designators.add(d)
                        still_missing.discard(d)
                        newly_found.add(d)
                uncertain_designators |= uncertain

        coverage = _compute_coverage(parts_designators, schematic_designators)
        pass_log.append("Pass {} ({} crop): +{} new ({}), coverage {:.0f}%".format(
            len(pass_log) + 1, grid_label, len(newly_found),
            ", ".join(sorted(newly_found)) if newly_found else "none",
            coverage,
        ))

        # Stop if no improvement — finer grid won't help
        if not newly_found:
            pass_log.append("No improvement — stopping crop passes.")
            break

    # --- Final cross-reference ---
    found = parts_designators & schematic_designators
    remaining_missing = parts_designators - schematic_designators

    maybe_found = set()
    still_missing_final = set()
    for m in remaining_missing:
        matched = False
        for u in uncertain_designators:
            pattern = u.replace("?", ".")
            if re.fullmatch(pattern, m):
                maybe_found.add("{} (possibly read as {})".format(m, u))
                matched = True
                break
        if not matched:
            still_missing_final.add(m)

    # --- Build report ---
    lines = []
    lines.append("# Cross-Check: {} schematic vs parts list".format(board_id))
    lines.append("")
    lines.append("Parts list source: {} ({} page(s))".format(
        ", ".join(matched_pages), len(matched_pages),
    ))
    lines.append("Schematic source: {} image(s)".format(len(schematic_images)))
    lines.append("")

    lines.append("## Scan Passes")
    for entry in pass_log:
        lines.append("- {}".format(entry))
    lines.append("")

    lines.append("## Summary")
    lines.append("- Parts list designators: **{}**".format(len(parts_designators)))
    lines.append("- Readable on schematic: **{}**".format(len(found)))
    lines.append("- Uncertain reads: **{}**".format(len(uncertain_designators)))
    lines.append("- Still missing: **{}**".format(len(still_missing_final)))
    lines.append("- Final coverage: **{:.0f}%**".format(coverage))
    lines.append("")

    if still_missing_final:
        by_type = {}
        for d in sorted(still_missing_final):
            prefix = re.match(r"[A-Z]+", d)
            key = prefix.group() if prefix else "OTHER"
            by_type.setdefault(key, []).append(d)

        lines.append("## Still Missing After All Passes")
        lines.append("")
        for prefix in sorted(by_type):
            lines.append("- **{}**: {}".format(prefix, ", ".join(sorted(by_type[prefix]))))
        lines.append("")

    if maybe_found:
        lines.append("## Uncertain Matches")
        for item in sorted(maybe_found):
            lines.append("- {}".format(item))
        lines.append("")

    if coverage >= COVERAGE_TARGET:
        lines.append("**Verdict**: Coverage target met ({:.0f}% >= {}%). All key components accounted for.".format(
            coverage, COVERAGE_TARGET,
        ))
    else:
        lines.append("**Verdict**: {:.0f}% coverage after all passes. {} designator(s) remain unreadable — these may require manual inspection of the original PDF.".format(
            coverage, len(still_missing_final),
        ))

    return "\n".join(lines)


@mcp.tool()
def generate_netlist(manual_name: str, board_id: str, save_json: bool = True) -> str:
    """Trace connections on a schematic and produce a SPICE netlist.

    Uses Claude vision to read the schematic image and trace wires between
    components. Circuit theory from _circuits/<board_id>.json (component types,
    functional blocks, descriptions) is provided as context to improve accuracy.

    Automatically crops the schematic into higher-resolution tiles and re-scans
    for components whose connections could not be determined on the first pass.

    Requires ANTHROPIC_API_KEY environment variable.

    Args:
        manual_name: Manual directory name (e.g. 'pdf-marantz-2250-service-manual').
                     Partial matches and spaces are accepted.
        board_id: Board assembly ID (e.g. 'P800', 'P700').
        save_json: If True, write the updated circuit JSON to _circuits/<board_id>.json.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not set. Required for vision analysis."

    manual_dir = _find_manual_dir(manual_name)
    if not manual_dir:
        return "Manual '{}' not found. Use list_manuals to see available manuals.".format(manual_name)

    def _progress(msg):
        """Emit progress to stderr so the Swift app can display it."""
        print(msg, file=sys.stderr, flush=True)

    client = anthropic.Anthropic(api_key=api_key)
    pass_log = []

    # --- Step 1: Gather context ---
    _progress("Step 1/4: Loading circuit context for {}...".format(board_id))
    circuit_data = _load_circuit_json(manual_dir, board_id)
    theory_context = _build_theory_context(circuit_data) if circuit_data else ""

    # Known component designators from circuit JSON
    known_components = set()
    if circuit_data:
        for c in circuit_data.get("components", []):
            known_components.add(c["designator"].upper())
        _progress("Found {} known components from circuit data".format(len(known_components)))
    else:
        _progress("No existing circuit data — will create from scratch")

    # Collect schematic images for this board
    board_id_upper = board_id.upper().strip()
    schematic_images = []
    for f in sorted(manual_dir.glob("*.md")):
        if f.name == "_index.md":
            continue
        text = f.read_text(encoding="utf-8")
        if board_id_upper not in text.upper():
            continue
        if re.search(r"parts.list", f.name, re.IGNORECASE):
            continue
        for img_name in _extract_image_refs(text):
            img_path = manual_dir / img_name
            if img_path.exists():
                schematic_images.append(img_path)

    if not schematic_images:
        return "No schematic images found for board '{}'. Check the board ID.".format(board_id)

    _progress("Found {} schematic image(s) for {}".format(len(schematic_images), board_id))

    # --- Pass 1: Full image netlist extraction ---
    _progress("Step 2/4: Pass 1 — analyzing full schematic image(s)...")
    all_entries = []
    netted_designators = set()

    for i, sch_path in enumerate(schematic_images):
        _progress("Pass 1: Analyzing image {}/{} ({})...".format(i + 1, len(schematic_images), sch_path.name))
        try:
            img_data, _, _ = _image_to_base64(sch_path)
        except Exception:
            continue

        prompt = _NETLIST_EXTRACTION_PROMPT.format(theory_context)
        try:
            result = _vision_call(client, [(img_data, "image/jpeg")], prompt, max_tokens=4096)
        except Exception as exc:
            pass_log.append("Pass 1: Vision error on {}: {}".format(sch_path.name, exc))
            _progress("Pass 1: Vision error on {}: {}".format(sch_path.name, exc))
            continue

        entries, netted = _parse_spice_lines(result)
        all_entries.extend(entries)
        netted_designators |= netted
        _progress("Pass 1: Found {} connections so far".format(len(all_entries)))

    coverage = _compute_coverage(known_components, netted_designators) if known_components else 0
    still_missing = known_components - netted_designators if known_components else set()
    pass1_msg = "Pass 1 (full image): netted {}/{} components, coverage {:.0f}%".format(
        len(known_components & netted_designators) if known_components else len(netted_designators),
        len(known_components) if known_components else "?",
        coverage,
    )
    pass_log.append(pass1_msg)
    _progress(pass1_msg)

    # --- Pass 2+: Progressive tile crop refinement ---
    _progress("Step 3/4: Tile refinement passes ({} missing)...".format(len(still_missing)))
    for grid_idx, grid in enumerate(CROP_GRIDS):
        if coverage >= COVERAGE_TARGET or not still_missing:
            _progress("Coverage target met or no missing components — skipping tile passes")
            break

        newly_found = set()
        grid_label = "{}x{}".format(grid[0], grid[1])
        _progress("Pass {} ({} crop): scanning for {} missing components...".format(
            grid_idx + 2, grid_label, len(still_missing)))

        for sch_path in schematic_images:
            try:
                tiles = _crop_image_tiles(sch_path, grid)
            except Exception:
                continue

            for tile_idx, tile_data in enumerate(tiles):
                if not still_missing:
                    break

                _progress("Pass {} tile {}/{}: {} still missing...".format(
                    grid_idx + 2, tile_idx + 1, len(tiles), len(still_missing)))
                prompt = _TARGETED_NET_PROMPT.format(
                    "\n".join(sorted(still_missing)),
                )
                try:
                    result = _vision_call(client, [tile_data], prompt, max_tokens=2048)
                except Exception:
                    continue

                entries, netted = _parse_spice_lines(result)
                for entry in entries:
                    bare = re.sub(r"^[QX]_", "", entry["designator"])
                    if bare in still_missing:
                        all_entries.append(entry)
                        netted_designators.add(bare)
                        still_missing.discard(bare)
                        newly_found.add(bare)

        coverage = _compute_coverage(known_components, netted_designators)
        pass_msg = "Pass {} ({} crop): +{} new ({}), coverage {:.0f}%".format(
            len(pass_log) + 1, grid_label, len(newly_found),
            ", ".join(sorted(newly_found)) if newly_found else "none",
            coverage,
        )
        pass_log.append(pass_msg)
        _progress(pass_msg)

        if not newly_found:
            pass_log.append("No improvement — stopping crop passes.")
            _progress("No improvement — stopping crop passes.")
            break

    # --- Save updated JSON ---
    _progress("Step 4/4: Saving circuit JSON...")
    json_saved = False
    if save_json and all_entries:
        try:
            if circuit_data:
                updated = _spice_to_circuit_json(all_entries, circuit_data)
            else:
                # No pre-existing circuit — build a minimal one from SPICE entries
                updated = _spice_entries_to_standalone_json(all_entries, board_id, manual_dir.name)
            circuits_dir = manual_dir / "_circuits"
            circuits_dir.mkdir(exist_ok=True)
            out_path = circuits_dir / "{}.json".format(_safe_filename(board_id.upper()))
            out_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")
            json_saved = True
            _progress("Saved circuit JSON: {}".format(out_path.name))
        except Exception as exc:
            _progress("Error saving JSON: {}".format(exc))

    # --- Build SPICE output ---
    spice_lines = ["* SPICE Netlist: {} ({})".format(board_id, manual_dir.name)]
    spice_lines.append("* Generated by service-manual-reader vision analysis")
    spice_lines.append("*")
    for entry in all_entries:
        spice_lines.append(entry["raw"])
    spice_lines.append(".END")
    spice_text = "\n".join(spice_lines)

    # --- Build report ---
    report = []
    report.append("# Netlist: {} ({})".format(board_id, manual_dir.name))
    report.append("")
    report.append("## Scan Passes")
    for entry in pass_log:
        report.append("- {}".format(entry))
    report.append("")

    report.append("## Summary")
    report.append("- Components in circuit data: **{}**".format(len(known_components) if known_components else "N/A"))
    report.append("- Components with connections traced: **{}**".format(len(netted_designators)))
    report.append("- Final coverage: **{:.0f}%**".format(coverage))
    if json_saved:
        report.append("- Circuit JSON updated: `_circuits/{}.json`".format(_safe_filename(board_id.upper())))
    report.append("")

    if still_missing:
        report.append("## Unresolved Components")
        for d in sorted(still_missing):
            report.append("- {}".format(d))
        report.append("")

    report.append("## SPICE Netlist")
    report.append("```spice")
    report.append(spice_text)
    report.append("```")

    return "\n".join(report)


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
