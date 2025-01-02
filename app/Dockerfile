FROM mcr.microsoft.com/playwright/python:v1.31.1-jammy

WORKDIR /app

# Install requirements
COPY requirements.txt /app
RUN pip install -r /app/requirements.txt

# Install project package
COPY src /app/src
COPY pyproject.toml /app
RUN pip install /app

# Copy files which are not contained in the package
COPY data /app/data
COPY house_features_validation_data /app/house_features_validation_data
COPY tests /app/tests