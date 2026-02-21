"""MCP server for service manual extraction and search.

Tools:
  - extract_manual: Convert a local PDF to structured markdown + images
  - list_manuals: List already-extracted manuals
  - search_service_manual: Search the web for a service manual PDF
"""

import os
import sys
import subprocess
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MANUALS_DIR = Path.home() / "Claude-Manuals"
CONVERT_SCRIPT = Path(__file__).resolve().parent.parent / "convert.py"

mcp = FastMCP("service-manual-reader")

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
        # Try DuckDuckGo HTML search (no API key needed)
        for q in [query, alt_query]:
            try:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": q},
                    headers={"User-Agent": "ServiceManualReader/1.0"},
                )
                if resp.status_code == 200:
                    # Parse result links from the HTML
                    import re
                    links = re.findall(
                        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.+?)</a>',
                        resp.text,
                    )
                    for url, title in links[:10]:
                        # DuckDuckGo wraps URLs in a redirect
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

    # Deduplicate by URL
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
