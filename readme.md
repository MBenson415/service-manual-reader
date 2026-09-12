# Service Manual Reader

Converts PDF service manuals into structured markdown + PNG schematics optimized for Claude. Three ways to use it:

1. **Claude Desktop (MCP)** — Claude can search for, extract, and read manuals directly
2. **Finder right-click** — Right-click any PDF → "Extract for Claude"
3. **CLI** — `python3 convert.py manual.pdf`

## Quick Install

```bash
./install.sh
```

This installs everything: Quick Action, MCP server, and dependencies. Restart Claude Desktop after.

## What It Does

Given a service manual PDF, the converter produces:

- **Structured markdown** per section (specs, schematics, adjustments, parts lists)
- **PNG schematics** per board/assembly at 200 DPI (legible component designators)
- **Assembly cross-reference** linking schematics ↔ parts lists ↔ adjustments
- **Metadata** with model info and board assembly numbers

```
/Users/marshallbenson/Desktop/Benchmark Audio Repair/Schematics/sx-750-service-manual/
├── _index.md                                    # TOC + metadata + cross-reference
├── 02-specifications.md                         # Specs section
├── 07-circuit-descriptions.md                   # With embedded schematic PNG
├── 12-adjustments.md                            # Bias, tracking procedures
├── 19-parts-location.md                         # Board layout diagrams
├── 37-power-amplifier-assembly-awh-046.md       # Schematic with board tag
├── 38-parts-lists-of-power-amplifier-assembly-awh-046.md
├── power-amplifier-assembly-awh-046-schematic-p057.png
├── power-amplifier-assembly-awh-046-schematic-p058.png
└── ...
```

## Claude Desktop (MCP)

After install, Claude Desktop has these manual-management tools:

- **search_service_manual** — "Find me the Pioneer SX-750 service manual"
- **download_service_manual** — "Download this direct PDF URL into a Brand+Model folder in Schematics"
- **extract_manual** — "Extract the PDF at ~/Downloads/SX-750.pdf"
- **list_manuals** — "What manuals do I have extracted?"
- **list_downloads** — "Show all downloaded PDFs in Schematics"

## CLI Usage

```bash
python3 convert.py "SX-750 Service Manual.pdf"
python3 convert.py manual.pdf --output-dir "/Users/marshallbenson/Desktop/Benchmark Audio Repair/Schematics"
```

## Manual Setup (if not using install.sh)

**Dependencies:**
```bash
pip3 install -r requirements.txt
```

**MCP server** (requires [uv](https://astral.sh/uv)):
```bash
cd mcp-server && uv sync
```

Then add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "service-manual-reader": {
      "command": "/path/to/uv",
      "args": ["--directory", "/path/to/service-manual-reader/mcp-server", "run", "main.py"]
    }
  }
}
```

## Reading Difficult Scans

- Small-angle deskew, adaptive local contrast, and light sharpening are applied to vision inputs. Original PNGs are never overwritten; analysis and hard retries include original views for comparison. Cleanup cannot reconstruct missing details.
- Crop passes overlap by 10%, include all edge pixels, and prioritize regions located by earlier readings. Unresolved labels/connections get a bounded stronger-model pass even when cheaper passes make no progress.
- Extraction keeps schematic images on text-bearing pages and writes positioned PDF words in adjacent `.text.json` files. Visible text seeds label identification; hidden OCR is advisory. Re-extract older manuals to enable this feature.
- `cross_check_schematic` saves per-designator `confirmed`, `uncertain`, or `missing` readings in `_circuits/<BOARD_ID>.json`, with image/crop evidence and source hashes. Unchanged sources reuse those readings; `refresh=true` forces a new check. Partial parts-list reads are retried.
- `generate_netlist` uses those labels to target unresolved connections. Label confidence is not electrical verification. Uncertain connections remain commented-out drafts and are excluded from saved connectivity.

Set these environment variables in the MCP server's `env` configuration or CLI environment:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SCHEMATIC_PREPROCESS` | `1` | Set to `0` to disable scan cleanup. |
| `SCHEMATIC_VISION_MODEL` | `claude-haiku-4-5-20251001` | Base vision model, including section detection. |
| `SCHEMATIC_HARD_MODEL` | `claude-sonnet-4-5-20250929` | Stronger model for unresolved crop retries. |
| `SCHEMATIC_MAX_HARD_CALLS` | `4` | Maximum hard-model calls per cross-check/netlist invocation; `0` disables escalation, maximum `20`. |

Vision requests use your Anthropic account and incur API charges. Tests mock those requests and do not spend API credits. The hard-call cap does not limit base-model calls. Restart the MCP client after changing configuration.

Downloads and extracted manuals now share `/Users/marshallbenson/Desktop/Benchmark Audio Repair/Schematics`. PDFs retain their `<Brand Model>` folders; extracted assets go under `<manual-name>`. Existing files in older locations are not moved automatically. `--output-dir` still overrides CLI extraction storage.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 mcp-server/.venv/bin/python -m unittest test_vision -v
```
