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
~/Claude-Manuals/sx-750-service-manual/
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

After install, Claude Desktop has three tools:

- **search_service_manual** — "Find me the Pioneer SX-750 service manual"
- **extract_manual** — "Extract the PDF at ~/Downloads/SX-750.pdf"
- **list_manuals** — "What manuals do I have extracted?"

## CLI Usage

```bash
python3 convert.py "SX-750 Service Manual.pdf"
python3 convert.py manual.pdf --output-dir ~/Claude-Manuals
```

## Manual Setup (if not using install.sh)

**Dependencies:**
```bash
pip3 install PyMuPDF
```

**MCP server** (requires [uv](https://astral.sh/uv)):
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
