#!/bin/bash
# Install script for Service Manual Reader
# Sets up: Quick Action (Finder right-click) + MCP server (Claude Desktop)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICES_DIR="$HOME/Library/Services"
CLAUDE_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
UV_PATH="$HOME/.local/bin/uv"

echo "=== Service Manual Reader Installer ==="
echo ""

# --- 1. Install Quick Action ---
echo "[1/3] Installing Finder Quick Action..."
mkdir -p "$SERVICES_DIR"
cp -R "$SCRIPT_DIR/Extract for Claude.workflow" "$SERVICES_DIR/"
echo "  ✓ Quick Action installed. Right-click any PDF → Quick Actions → Extract for Claude"
echo ""

# --- 2. Install Python dependencies ---
echo "[2/3] Installing Python dependencies..."
if command -v "$UV_PATH" &>/dev/null; then
    echo "  Using uv..."
    cd "$SCRIPT_DIR/mcp-server"
    "$UV_PATH" sync 2>&1 | tail -1
else
    echo "  uv not found, using pip..."
    pip3 install -r "$SCRIPT_DIR/requirements.txt" 2>&1 | tail -1
fi
echo "  ✓ Dependencies installed"
echo ""

# --- 3. Configure Claude Desktop MCP ---
echo "[3/3] Configuring Claude Desktop MCP server..."
if [ -f "$CLAUDE_CONFIG" ]; then
    # Check if mcpServers key already exists
    if python3 -c "
import json, sys
with open('$CLAUDE_CONFIG') as f:
    config = json.load(f)
if 'mcpServers' not in config:
    config['mcpServers'] = {}
config['mcpServers']['service-manual-reader'] = {
    'command': '$UV_PATH',
    'args': ['--directory', '$SCRIPT_DIR/mcp-server', 'run', 'main.py']
}
with open('$CLAUDE_CONFIG', 'w') as f:
    json.dump(config, f, indent=2)
print('  ✓ MCP server added to Claude Desktop config')
" 2>/dev/null; then
        :
    else
        echo "  ⚠ Could not update Claude Desktop config automatically."
        echo "  Add this to $CLAUDE_CONFIG manually:"
        echo ""
        echo '  "mcpServers": {'
        echo '    "service-manual-reader": {'
        echo "      \"command\": \"$UV_PATH\","
        echo "      \"args\": [\"--directory\", \"$SCRIPT_DIR/mcp-server\", \"run\", \"main.py\"]"
        echo '    }'
        echo '  }'
    fi
else
    echo "  ⚠ Claude Desktop config not found at $CLAUDE_CONFIG"
    echo "  Install Claude Desktop first, then re-run this script."
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Usage:"
echo "  Finder:  Right-click PDF → Quick Actions → Extract for Claude"
echo "  Claude:  Ask Claude to 'search for the Pioneer SX-750 service manual'"
echo "  CLI:     python3 $SCRIPT_DIR/convert.py <path-to-pdf>"
echo ""
echo "Extracted manuals go to: ~/Claude-Manuals/"
echo ""
echo "⚠ Restart Claude Desktop to pick up the MCP server."
