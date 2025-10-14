# Use lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first for better Docker caching
COPY app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy the rest of the app
COPY app/ .

# Expose port 80 (Azure expects this)
EXPOSE 80

# Use Gunicorn to serve the app
# The format is: gunicorn --bind 0.0.0.0:PORT module_name:app_variable
CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:app"]

