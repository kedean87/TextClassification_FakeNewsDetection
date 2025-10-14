# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app/ ./app/

WORKDIR /app/app

# Expose port
EXPOSE 8080

CMD ["python", "app.py"]

