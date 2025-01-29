# FROM python:3.11.5-slim

# WORKDIR /app

# COPY requirements.txt ./requirements.txt
# COPY app.py ./app.py
# COPY templates/ ./templates

# RUN pip install --no-cache-dir -r requirements.txt
# EXPOSE 8501
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8501"]

# Use a lightweight Python image
FROM python:3.11.5-slim

# Set the working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
