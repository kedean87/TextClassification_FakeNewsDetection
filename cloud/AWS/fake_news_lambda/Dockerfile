# Dockerfile
FROM public.ecr.aws/lambda/python:3.10

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install dependencies
COPY app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ${LAMBDA_TASK_ROOT}/

# Lambda handler
CMD ["lambda_function.lambda_handler"]

