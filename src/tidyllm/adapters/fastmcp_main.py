"""FastMCP main entry point with startup information."""

import json
import os
import sys
from pathlib import Path

from tidyllm.adapters.fastmcp_adapter import create_tidyllm_mcp_server
from tidyllm.registry import REGISTRY

# Print startup information
print("🚀 TidyLLM FastMCP Server Starting...")
print(f"📁 Current directory: {os.getcwd()}")
print(f"🐍 Python executable: {sys.executable}")
print(f"📦 TidyLLM package location: {Path(__file__).parent.parent}")

# Print Claude Desktop configuration
config_json = {
    "mcpServers": {
        "tidyllm": {
            "command": "uv",
            "args": ["run", "tidyllm-mcp"],
            "env": {
                "TIDYLLM_NOTES_DIR": os.environ.get("TIDYLLM_NOTES_DIR", str(Path.home() / "Documents" / "Notes")),
                "TIDYLLM_USER_DB": os.environ.get("TIDYLLM_USER_DB", str(Path.home() / ".config" / "tidyllm" / "user.db"))
            }
        }
    }
}

print("\n📋 Claude Desktop Configuration:")
print("Add this to your claude_desktop_config.json:")
print(json.dumps(config_json, indent=2))

# Alternative configuration for fastmcp run
alt_config = {
    "mcpServers": {
        "tidyllm": {
            "command": "uv",
            "args": ["run", "fastmcp", "run", f"{os.getcwd()}/src/tidyllm/adapters/fastmcp_main.py"],
            "env": config_json["mcpServers"]["tidyllm"]["env"]
        }
    }
}

print("\n🔧 Alternative configuration (using fastmcp run):")
print(json.dumps(alt_config, indent=2))

print(f"\n🛠️  Available tools: {len(REGISTRY.functions)} registered")
for tool_desc in REGISTRY.functions:
    print(f"  - {tool_desc.name}")

print("\n🎯 Server starting on STDIO transport...")

# Create the default server instance that fastmcp can find
mcp = create_tidyllm_mcp_server()

if __name__ == "__main__":
    # Run the server when executed directly
    mcp.run()