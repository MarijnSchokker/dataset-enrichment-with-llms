FROM mcr.microsoft.com/playwright/python:v1.31.1-jammy

WORKDIR /app

# copy project
COPY pyproject.toml /app
COPY requirements.txt /app
RUN pip install /app[quality_checks,tests]

COPY src /app/src
COPY data /app/data
COPY house_features_validation_data /app/house_features_validation_data
COPY tests /app/tests
