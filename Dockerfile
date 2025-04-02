# Use Python 3.9 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the calculator code
COPY calculator.py .

# Install the calculator module
RUN mcp install calculator.py

# Expose the port that MCP server runs on
EXPOSE 6277

# Command to run the MCP server
CMD ["python", "-m", "mcp.cmd.server", "calculator"]