# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory in the container
WORKDIR /app

# Copy the project definition file
COPY pyproject.toml ./

# Install project dependencies using uv
RUN uv sync

# Copy the rest of the application code
COPY main.py .
# Assuming model.py contains necessary logic used by main.py
COPY model.py .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application using uv when the container launches
# Runs `python main.py serve --host 0.0.0.0 --port 8000` via uv
CMD ["uv", "run", "python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
