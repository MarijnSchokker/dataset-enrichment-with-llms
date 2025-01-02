FROM mcr.microsoft.com/playwright/python:v1.31.1-jammy

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r /app/requirements.txt

COPY src /app/src
COPY data /app/data
COPY house_features_validation_data /app/house_features_validation_data
COPY tests /app/tests

COPY pyproject.toml /app
RUN pip install /app
