"""
Drop-in tools for the service-manual-reader MCP server.

Adds:
  - list_schematics()   : see existing equipment folders + PDFs in Schematics/
  - download_manual()   : download a manual PDF into Schematics/<folder>/

Assumes FastMCP style (matches the Args-docstring pattern of your existing
tools). Paste these into your server module and reuse your existing `mcp`
instance — delete the placeholder instance below if you already have one.
"""

import re
import importlib
import urllib.request
from pathlib import Path

if "mcp" not in globals():
    try:
        # Use FastMCP when available for local/standalone execution.
        FastMCP = importlib.import_module("mcp.server.fastmcp").FastMCP
        mcp = FastMCP("service-manual-reader")
    except Exception:
        # Fallback keeps @mcp.tool() decorators valid without MCP installed.
        class _MCPStub:
            def tool(self):
                def _decorator(fn):
                    return fn

                return _decorator

        mcp = _MCPStub()

SCHEMATICS_DIR = Path(
    "/Users/marshallbenson/Desktop/Benchmark Audio Repair/Schematics"
)

# Some manual-hosting sites reject the default urllib user agent.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}

_MAX_BYTES = 200 * 1024 * 1024  # 200 MB safety cap


def _sanitize(name: str) -> str:
    """Make a string safe for use as a file/folder name."""
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip(" .")


@mcp.tool()
def list_schematics() -> str:
    """List equipment folders and PDFs in the Schematics directory.

    Returns one line per folder with its PDF filenames, so you can check
    whether a manual already exists before downloading, and match the
    existing folder-naming convention.
    """
    if not SCHEMATICS_DIR.is_dir():
        return f"Schematics directory not found: {SCHEMATICS_DIR}"

    lines = []
    for folder in sorted(SCHEMATICS_DIR.iterdir()):
        if folder.name.startswith(".") or not folder.is_dir():
            continue
        pdfs = sorted(p.name for p in folder.glob("*.pdf"))
        lines.append(f"{folder.name}/  ->  {', '.join(pdfs) if pdfs else '(no PDFs)'}")

    # Loose PDFs sitting at the top level, if any
    loose = sorted(p.name for p in SCHEMATICS_DIR.glob("*.pdf"))
    if loose:
        lines.append(f"(top level)  ->  {', '.join(loose)}")

    return "\n".join(lines) if lines else "Schematics directory is empty."


@mcp.tool()
def download_manual(
    url: str,
    brand: str,
    model: str,
    folder_name: str = "",
    filename: str = "",
) -> str:
    """Download a service manual PDF into the Schematics directory.

    Saves to Schematics/<folder_name>/<filename>. Creates the folder if it
    doesn't exist. Verifies the downloaded file is actually a PDF (magic
    bytes) and refuses to overwrite an existing file.

    Args:
        url: Direct link to the PDF (from search_service_manual).
        brand: Equipment brand, e.g. 'Pioneer'.
        model: Model number, e.g. 'SX-1050'.
        folder_name: Target folder under Schematics/. Defaults to
            '<Brand> <Model>'. Pass an existing folder name to match
            your current convention (check with list_schematics first).
        filename: Output filename. Defaults to
            '<Brand> <Model> Service Manual.pdf'.
    """
    brand, model = _sanitize(brand), _sanitize(model)
    if not brand or not model:
        return "Error: Both brand and model are required. Folder format is '<Brand> <Model>'."

    folder = SCHEMATICS_DIR / _sanitize(folder_name or f"{brand} {model}")
    fname = _sanitize(filename or f"{brand} {model} Service Manual.pdf")
    if not fname.lower().endswith(".pdf"):
        fname += ".pdf"
    dest = folder / fname

    if dest.exists():
        return f"Already exists, not overwriting: {dest}"

    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            first = resp.read(1024)
            if not first.startswith(b"%PDF"):
                ctype = resp.headers.get("Content-Type", "unknown")
                return (
                    f"Not a PDF (Content-Type: {ctype}). The link is probably a "
                    "landing/login page rather than a direct file link — open it "
                    "in a browser or find a direct .pdf URL."
                )
            folder.mkdir(parents=True, exist_ok=True)
            total = 0
            with open(dest, "wb") as f:
                f.write(first)
                total += len(first)
                while chunk := resp.read(1024 * 256):
                    total += len(chunk)
                    if total > _MAX_BYTES:
                        f.close()
                        dest.unlink(missing_ok=True)
                        return f"Aborted: file exceeded {_MAX_BYTES // 1024 // 1024} MB cap."
                    f.write(chunk)
    except Exception as e:  # noqa: BLE001 - report any failure to the model
        dest.unlink(missing_ok=True)
        return f"Download failed: {type(e).__name__}: {e}"

    mb = total / 1024 / 1024
    return f"Saved {mb:.1f} MB to {dest}\nNext step: extract_manual(pdf_path='{dest}')"