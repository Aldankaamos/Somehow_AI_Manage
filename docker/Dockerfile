# Use an official Python runtime as a parent image
FROM python:3.10.12-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY app/ .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Set environment variable to use file-based tracking
ENV MLFLOW_TRACKING_URI http://127.0.0.1:5000

# Expose ports for Streamlit and MLflow
EXPOSE 8501 8000 5001 5000

# Run the app
CMD ["streamlit", "run", "run.py"]
