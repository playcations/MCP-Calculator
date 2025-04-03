# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the calculator code
COPY calculator.py .

# Expose the port that MCP server runs on
EXPOSE 6277

# Use the Python module version of the command instead of the executable
CMD ["python", "-m", "mcp.server", "dev", "calculator.py"]
