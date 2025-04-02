# MCP-Calculator

A Python-based calculator service using MCP (Managed Compute Protocol) that provides various mathematical operations through a containerized API.

## Features

- Basic arithmetic operations (add, subtract, multiply, divide)
- Advanced mathematical functions:
  - Power calculations
  - Factorial
  - Fibonacci sequence
  - Prime number checking
  - GCD and LCM
  - Square root
  - Logarithms
  - Trigonometric functions
- Numerical calculus operations

## Requirements

- Docker
- Python 3.12
- MCP
- NumPy
- SciPy

## Quick Start

1. Build the Docker container:
```bash
docker build -t calculator-service .
```

2. Run the container:
```bash
docker run -p 6277:6277 calculator-service
```

The service will be available on port 6277.

## Project Structure

- `calculator.py` - Main calculator implementation with MCP tools
- `Dockerfile` - Container configuration
- `requirements.txt` - Python dependencies
