# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install uv
RUN apt-get update && apt-get install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    apt-get remove -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the project definition file
COPY pyproject.toml ./

# Install project dependencies using uv
# --system installs to the system Python, suitable for containers
RUN uv pip install --system .

# Copy the rest of the application code
COPY main.py .
# Assuming model.py contains necessary logic used by main.py
COPY model.py .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable for the host (optional, can be overridden)
ENV HOST=0.0.0.0
ENV PORT=8000

# Run the application using uv when the container launches
# Runs `python main.py serve --host 0.0.0.0 --port 8000` via uv
CMD ["uv", "run", "python", "main.py", "serve", "--host", "${HOST}", "--port", "${PORT}"]
