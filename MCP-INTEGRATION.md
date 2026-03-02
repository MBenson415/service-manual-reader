# Service Manual Reader — MCP Server Integration Guide

## Overview

A local MCP (Model Context Protocol) server that extracts electronics service manual PDFs into structured data, analyzes schematic images via Claude vision, and generates SPICE netlists with full circuit connectivity. Designed to be consumed by desktop applications over stdio JSON-RPC.

## Transport

- **Protocol:** MCP over stdio (JSON-RPC 2.0 with `Content-Length` framing)
- **Launch command:**
  ```
  ~/.local/bin/uv --directory /path/to/service-manual-reader/mcp-server run main.py
  ```
- **Environment:** Requires `ANTHROPIC_API_KEY` for vision-based tools (`analyze_schematic`, `cross_check_schematic`, `generate_netlist`)

### Swift Integration

```swift
let process = Process()
process.executableURL = URL(fileURLWithPath: "\(HOME)/.local/bin/uv")
process.arguments = ["--directory", "/path/to/mcp-server", "run", "main.py"]
process.environment = ["ANTHROPIC_API_KEY": apiKey]

let stdin = Pipe()
let stdout = Pipe()
process.standardInput = stdin
process.standardOutput = stdout
process.launch()

// Send:  "Content-Length: N\r\n\r\n{JSON-RPC request}"
// Read:  "Content-Length: N\r\n\r\n{JSON-RPC response}"
```

## Tools

### Manual Management

| Tool | Parameters | Returns |
|------|-----------|---------|
| `extract_manual` | `pdf_path: String` | Status message. Extracts PDF → `~/Claude-Manuals/<name>/` |
| `list_manuals` | — | List of extracted manual names |
| `search_service_manual` | `brand: String, model: String` | Web search results with download links |

### Reading

| Tool | Parameters | Returns |
|------|-----------|---------|
| `read_index` | `manual_name: String` | Full `_index.md` — TOC, board assemblies, cross-references |
| `read_section` | `manual_name: String, section: String, text_only: Bool = false` | Markdown text + JPEG images (as base64 `ImageContent`) |
| `get_schematic` | `manual_name: String, board_id: String, page: Int = 0` | Schematic/parts list images for a board. `page=0` lists available images. |

### Vision Analysis (requires ANTHROPIC_API_KEY)

| Tool | Parameters | Returns |
|------|-----------|---------|
| `analyze_schematic` | `manual_name: String, board_id: String, page: Int` | Natural language circuit analysis (components, signal path, power rails) |
| `cross_check_schematic` | `manual_name: String, board_id: String, parts_list_page: Int = 0` | Coverage report — compares parts list vs schematic, auto-crops for missing components |
| `generate_netlist` | `manual_name: String, board_id: String, save_json: Bool = true` | SPICE netlist + updates `_circuits/<board_id>.json` with traced connectivity |

## Data Model — `_circuits/<board_id>.json`

This is the primary data file for 3D assembly rendering. Located at:
```
~/Claude-Manuals/<manual-name>/_circuits/<BOARD_ID>.json
```

### Schema

```json
{
  "id": "marantz_2250_p800",
  "name": "Marantz 2250 P800 Power Supply",
  "description": "Power supply circuit for Marantz 2250 receiver",

  "components": [
    {
      "designator": "HB01",
      "id": "HB01",
      "type": "transistor_npn",
      "value": "",
      "description": "Power transistor",
      "functionalBlock": "power_stage",
      "position": [-0.4, 0.3, 0],
      "pins": [
        { "id": "B", "label": "", "netID": "n1" },
        { "id": "C", "label": "", "netID": "VCC" },
        { "id": "E", "label": "", "netID": "n2" }
      ]
    }
  ],

  "functionalBlocks": [
    {
      "id": "power_stage",
      "name": "Power Transistors",
      "description": "Main power amplification stage",
      "color": "red",
      "componentIDs": ["HB01", "HB02", "HB03", "HB04", "HB05", "HB06", "HB07"]
    }
  ],

  "nets": [
    {
      "id": "VCC",
      "connectedPins": [
        { "componentID": "HB01", "pinID": "C" },
        { "componentID": "C807", "pinID": "+" },
        { "componentID": "C810", "pinID": "1" }
      ]
    }
  ]
}
```

### Field Reference

**Component:**
| Field | Type | Description |
|-------|------|-------------|
| `designator` | String | Reference designator (R807, C807, HB01, J801) |
| `type` | String | `resistor`, `capacitor`, `capacitor_electrolytic`, `transistor_npn`, `transistor_pnp`, `connector`, `diode`, `inductor` |
| `value` | String | Component value ("820Ω", "470µF", "2200µF", "0.1µF", or empty) |
| `description` | String | Functional description ("Base resistor", "Filter capacitor") |
| `functionalBlock` | String | ID of the parent functional block |
| `position` | [Float, Float, Float] | 3D position hint (x, y, z) — normalized coordinates |
| `pins` | [Pin] | Ordered pin list. Pin order matters for SPICE mapping. |

**Pin:**
| Field | Type | Description |
|-------|------|-------------|
| `id` | String | Pin identifier. Transistors: `B`, `C`, `E`. Passives: `1`, `2`. Electrolytic: `+`, `-`. |
| `netID` | String | Net this pin connects to (matches a `nets[].id`) |

**Functional Block:**
| Field | Type | Description |
|-------|------|-------------|
| `id` | String | Block identifier (e.g. `power_stage`, `bias_circuit`) |
| `name` | String | Human-readable name ("Power Transistors", "Bias Network") |
| `description` | String | What this block does in the circuit |
| `color` | String | Suggested render color (red, green, blue, orange, purple, yellow) |
| `componentIDs` | [String] | Designators of components in this block |

**Net:**
| Field | Type | Description |
|-------|------|-------------|
| `id` | String | Net name. Named: `VCC`, `VEE`, `GND`, signal names. Anonymous: `n1`, `n2`, `n_B01`, `n_drive01` |
| `connectedPins` | [ConnectedPin] | All component pins on this net |

## Typical Workflow for 3D Assembly

1. **`list_manuals`** → get available manual names
2. **`read_index(manual_name)`** → get board assembly IDs (`P800`, `P700`, `PE11`, etc.)
3. **`generate_netlist(manual_name, board_id)`** → traces schematic, produces SPICE netlist, writes `_circuits/<board_id>.json`
4. **Read `_circuits/<board_id>.json` directly from disk** → parse components, positions, functional blocks, and net connectivity for 3D rendering

### Rendering Hints

- **`position`** values are normalized ~[-1, 1] and represent approximate schematic layout. Use as initial 3D placement seeds.
- **`functionalBlock` + `color`** group related components visually. Use block colors for grouping or highlighting in the 3D scene.
- **`nets`** define electrical connectivity — render as wires/traces between the connected pins.
- **Pin order** in `pins` array maps directly to SPICE node order:
  - Transistors: `[collector, base, emitter]`
  - 2-terminal: `[pin1, pin2]`
  - Electrolytic: `[+, -]`

## Available Boards (Marantz 2250)

| Board ID | Name | Components |
|----------|------|------------|
| `P800` | Power Supply | 25 (7 transistors, 7 bias resistors, 4 caps, 7 connectors) |
| `P700` | Main Amplifier | Amplifier stages with feedback network |
| `PE11` | Pre-Tone Amplifier | Cascaded amplifier stages |
| `PHO1` | Filter | RIAA equalization network |
| `PC01` | Dolby FM Assembly | Dolby noise reduction |
| `MODEL 2528` | Mechanical | U.S. market model |
