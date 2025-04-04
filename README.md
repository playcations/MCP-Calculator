# MCP-Calculator

A Python-based calculator service using MCP (Managed Compute Protocol) that provides various mathematical operations through a containerized API.

## Features

tons of math from scipy and numpy


## Quick Start

### Local Server

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Install calculator.py to MCP
```bash
MCP install calculator.py
```

## Claude Integration

To use this calculator with Claude, add the following configuration to your Claude settings:

Either 

add directly to docker:

```json
{
  "mcpServers": {
     "Calculator": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "mcp[cli]",
        "mcp",
        "run",
        "/PathTo/calculator.py"
      ]
    }
  }
}
```
