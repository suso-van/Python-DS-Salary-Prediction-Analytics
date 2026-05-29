# 1. Use an official, lightweight Python runtime as a parent image
FROM python:3.11-slim

# 2. Set systemic environment variables to optimize Python inside Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Establish the operational directory inside the container
WORKDIR /app

# 4. Copy only dependency metrics first to leverage Docker's caching layer
COPY requirements.txt /app/

# 5. Upgrade pip and install core system requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy the local application layers into the container workspace
COPY ./src /app/src
COPY ./models /app/models
COPY ./app.py /app/app.py

# 7. Expose the standard FastAPI network port boundary
EXPOSE 8000

# 8. Launch the Uvicorn application server bound to all incoming network interfaces
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
