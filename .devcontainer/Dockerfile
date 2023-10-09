FROM mcr.microsoft.com/playwright/python:v1.31.1-jammy

WORKDIR /app

# copy project
COPY pyproject.toml /app
COPY requirements.txt /app
COPY src /app/src

# install project
RUN pip install /app[quality_checks,tests]
