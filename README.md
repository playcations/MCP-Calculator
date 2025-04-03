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
'''

3. run the server
'''bash
MCP dev calculator.py
'''


### Docker

You can run the calculator service directly from Docker Hub:

```bash
docker run -p 6277:6277 shooshmashta/calculatormcp:latest
```

Or build it locally:

1. Build the Docker container:
```bash
docker build -t calculator-service .
```

2. Run the container:
```bash
docker run -p 6277:6277 calculator-service
```

The service will be available on port 6277.

## Claude Integration

To use this calculator with Claude, add the following configuration to your Claude settings:

```json
{
  "mcpServers": {
    "calculatormcp": {
      "command": "curl",
      "args": [
        "-s",
        "-X",
        "POST",
        "-H",
        "Content-Type: application/json",
        "-d",
        "@-",
        "http://IP.ADDRESS:PORT"
      ]
    }
  }
}
```
