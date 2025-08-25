# Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the API-specific requirements file
COPY api/requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API source code into the container
COPY api/ /app/

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run your app using uvicorn
# The app will listen on all available interfaces (0.0.0.0)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]